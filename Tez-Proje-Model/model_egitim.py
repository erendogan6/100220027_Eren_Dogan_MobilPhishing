import gc
import tensorflow as tf
import os
import json
import logging
import sys
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers.legacy import Adam
import numpy as np
import glob
import random
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
import time
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve

# Logging formatını ve seviyesini ayarla
logging.basicConfig(
    level=logging.DEBUG,
    format='%(levelname)s - %(name)s - %(message)s',
    handlers=[
        # Konsola yazma
        logging.StreamHandler(sys.stdout)
    ]
)

# Root logger'ı al
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)

# Matplotlib'in diğer verbose loglarını da kapatmak istersen:
logging.getLogger('matplotlib').setLevel(logging.WARNING)

# Başlangıç mesajı
logger.info("=== Model Training Started ===")


def _calculate_specificity(y_true, y_pred):
    """Specificity (True Negative Rate) hesapla."""
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0


class PerformanceAnalyzer:
    """Model performansını analiz eden ve görselleştiren sınıf."""

    def __init__(self, export_dir):
        self.export_dir = export_dir
        self.logger = logging.getLogger(__name__)

    def evaluate_model(self, model, test_dataset, history=None):
        """Kapsamlı model performans analizi."""
        self.logger.info("Starting comprehensive performance evaluation")

        # Test verilerini topla
        y_true = []
        y_pred_probs = []

        for batch_x, batch_y in test_dataset:
            predictions = model.predict(batch_x, verbose=0)
            y_pred_probs.extend(tf.nn.sigmoid(predictions).numpy().flatten())
            y_true.extend(batch_y.numpy().flatten())

        y_true = np.array(y_true)
        y_pred_probs = np.array(y_pred_probs)
        y_pred = (y_pred_probs > 0.5).astype(int)

        # Temel metrikler
        metrics = self._calculate_metrics(y_true, y_pred, y_pred_probs)

        # Görselleştirmeler
        self._plot_confusion_matrix(y_true, y_pred)
        self._plot_roc_curve(y_true, y_pred_probs)
        self._plot_precision_recall_curve(y_true, y_pred_probs)
        self._plot_prediction_distribution(y_pred_probs, y_true)

        if history is not None:
            self._plot_training_history(history)

        # Rapor kaydet
        self._save_evaluation_report(metrics, y_true, y_pred)

        return metrics

    def _calculate_metrics(self, y_true, y_pred, y_pred_probs):
        """Detaylı performans metrikleri hesapla."""
        from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                     f1_score, matthews_corrcoef, log_loss)

        # ROC AUC
        fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
        roc_auc = auc(fpr, tpr)

        # Precision-Recall AUC
        precision, recall, _ = precision_recall_curve(y_true, y_pred_probs)
        pr_auc = auc(recall, precision)

        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred),
            'specificity': _calculate_specificity(y_true, y_pred),
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'matthews_corrcoef': matthews_corrcoef(y_true, y_pred),
            'log_loss': log_loss(y_true, y_pred_probs),
            'total_samples': len(y_true),
            'positive_samples': np.sum(y_true),
            'negative_samples': len(y_true) - np.sum(y_true)
        }

        # Sonuçları logla
        self.logger.info("=== PERFORMANCE METRICS ===")
        for metric, value in metrics.items():
            if isinstance(value, float):
                self.logger.info(f"{metric.upper()}: {value:.4f}")
            else:
                self.logger.info(f"{metric.upper()}: {value}")

        return metrics

    def _plot_confusion_matrix(self, y_true, y_pred):
        """Confusion Matrix görselleştirme."""
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_true, y_pred)

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Legal', 'Phishing'],
                    yticklabels=['Legal', 'Phishing'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')

        # Yüzde değerlerini ekle
        total = np.sum(cm)
        for i in range(2):
            for j in range(2):
                percentage = (cm[i, j] / total) * 100
                plt.text(j + 0.5, i + 0.7, f'({percentage:.1f}%)',
                         ha='center', va='center', fontsize=10, color='red')

        plt.tight_layout()
        plt.savefig(os.path.join(self.export_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info("Confusion matrix saved")

    def _plot_roc_curve(self, y_true, y_pred_probs):
        """ROC Curve görselleştirme."""
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_probs)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                 label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
                 label='Random Classifier')

        # Optimal threshold noktasını işaretle
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        plt.plot(fpr[optimal_idx], tpr[optimal_idx], 'ro', markersize=8,
                 label=f'Optimal Threshold = {optimal_threshold:.3f}')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.export_dir, 'roc_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info("ROC curve saved")

    def _plot_precision_recall_curve(self, y_true, y_pred_probs):
        """Precision-Recall Curve görselleştirme."""
        precision, recall, _ = precision_recall_curve(y_true, y_pred_probs)
        pr_auc = auc(recall, precision)

        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2,
                 label=f'PR curve (AUC = {pr_auc:.4f})')

        # Baseline (random classifier) için pozitif sınıf oranı
        baseline = np.sum(y_true) / len(y_true)
        plt.axhline(y=baseline, color='red', linestyle='--',
                    label=f'Random Classifier (Baseline = {baseline:.3f})')

        # F1-score maksimum noktasını bul
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        max_f1_idx = np.argmax(f1_scores)
        plt.plot(recall[max_f1_idx], precision[max_f1_idx], 'go', markersize=8,
                 label=f'Max F1-Score = {f1_scores[max_f1_idx]:.3f}')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.export_dir, 'precision_recall_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info("Precision-Recall curve saved")

    def _plot_prediction_distribution(self, y_pred_probs, y_true):
        """Tahmin dağılımı görselleştirme."""
        plt.figure(figsize=(12, 8))

        # Alt grafik 1: Histogram
        plt.subplot(2, 2, 1)
        plt.hist(y_pred_probs[y_true == 0], bins=50, alpha=0.7, label='Legal', color='blue', density=True)
        plt.hist(y_pred_probs[y_true == 1], bins=50, alpha=0.7, label='Phishing', color='red', density=True)
        plt.xlabel('Prediction Probability')
        plt.ylabel('Density')
        plt.title('Prediction Probability Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Alt grafik 2: Box plot
        plt.subplot(2, 2, 2)
        data_to_plot = [y_pred_probs[y_true == 0], y_pred_probs[y_true == 1]]
        plt.boxplot(data_to_plot, tick_labels=['Legal', 'Phishing'])  # labels yerine tick_labels
        plt.ylabel('Prediction Probability')
        plt.title('Prediction Probability Box Plot')
        plt.grid(True, alpha=0.3)

        # Alt grafik 3: Threshold analizi
        plt.subplot(2, 2, 3)
        thresholds = np.linspace(0, 1, 100)
        accuracies = []
        f1_scores = []

        for threshold in thresholds:
            y_pred_thresh = (y_pred_probs > threshold).astype(int)
            acc = np.mean(y_true == y_pred_thresh)

            # F1 score hesapla (sıfır bölme hatası kontrolü ile)
            tp = np.sum((y_true == 1) & (y_pred_thresh == 1))
            fp = np.sum((y_true == 0) & (y_pred_thresh == 1))
            fn = np.sum((y_true == 1) & (y_pred_thresh == 0))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            accuracies.append(acc)
            f1_scores.append(f1)

        plt.plot(thresholds, accuracies, label='Accuracy', color='blue')
        plt.plot(thresholds, f1_scores, label='F1-Score', color='red')
        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.title('Performance vs Threshold')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Alt grafik 4: Calibration plot
        plt.subplot(2, 2, 4)
        from sklearn.calibration import calibration_curve
        try:
            fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_pred_probs, n_bins=10)
            plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Model")
            plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
            plt.xlabel('Mean Predicted Probability')
            plt.ylabel('Fraction of Positives')
            plt.title('Calibration Plot')
            plt.legend()
            plt.grid(True, alpha=0.3)
        except:
            plt.text(0.5, 0.5, 'Calibration plot\nnot available', ha='center', va='center')
            plt.title('Calibration Plot')

        plt.tight_layout()
        plt.savefig(os.path.join(self.export_dir, 'prediction_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info("Prediction analysis plots saved")

    def _plot_training_history(self, history):
        """Eğitim geçmişi görselleştirme."""
        plt.figure(figsize=(15, 5))

        # Loss grafiği
        plt.subplot(1, 3, 1)
        plt.plot(history.history['loss'], label='Training Loss', color='blue')
        plt.plot(history.history['val_loss'], label='Validation Loss', color='red')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Accuracy grafiği
        plt.subplot(1, 3, 2)
        plt.plot(history.history['accuracy'], label='Training Accuracy', color='blue')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='red')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Learning curve
        plt.subplot(1, 3, 3)
        epochs = range(1, len(history.history['loss']) + 1)
        plt.plot(epochs, history.history['loss'], 'bo-', label='Training Loss')
        plt.plot(epochs, history.history['val_loss'], 'ro-', label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.export_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info("Training history plots saved")

    def _save_evaluation_report(self, metrics, y_true, y_pred):
        """Detaylı değerlendirme raporu kaydet."""
        report_path = os.path.join(self.export_dir, 'evaluation_report.txt')

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("MODEL PERFORMANCE EVALUATION REPORT\n")
            f.write("=" * 60 + "\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Dataset bilgileri
            f.write("DATASET INFORMATION:\n")
            f.write(f"Total samples: {metrics['total_samples']}\n")
            f.write(f"Positive samples (Phishing): {metrics['positive_samples']}\n")
            f.write(f"Negative samples (Legal): {metrics['negative_samples']}\n")
            f.write(
                f"Class distribution: {metrics['positive_samples'] / metrics['total_samples']:.3f} positive, {metrics['negative_samples'] / metrics['total_samples']:.3f} negative\n\n")

            # Performans metrikleri
            f.write("PERFORMANCE METRICS:\n")
            f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
            f.write(f"Precision: {metrics['precision']:.4f}\n")
            f.write(f"Recall (Sensitivity): {metrics['recall']:.4f}\n")
            f.write(f"Specificity: {metrics['specificity']:.4f}\n")
            f.write(f"F1-Score: {metrics['f1_score']:.4f}\n")
            f.write(f"ROC AUC: {metrics['roc_auc']:.4f}\n")
            f.write(f"PR AUC: {metrics['pr_auc']:.4f}\n")
            f.write(f"Matthews Correlation Coefficient: {metrics['matthews_corrcoef']:.4f}\n")
            f.write(f"Log Loss: {metrics['log_loss']:.4f}\n\n")

            # Confusion Matrix
            cm = confusion_matrix(y_true, y_pred)
            f.write("CONFUSION MATRIX:\n")
            f.write(f"True Negatives: {cm[0, 0]}\n")
            f.write(f"False Positives: {cm[0, 1]}\n")
            f.write(f"False Negatives: {cm[1, 0]}\n")
            f.write(f"True Positives: {cm[1, 1]}\n\n")

            # Classification Report
            f.write("DETAILED CLASSIFICATION REPORT:\n")
            f.write(classification_report(y_true, y_pred, target_names=['Legal', 'Phishing']))

        self.logger.info(f"Evaluation report saved to {report_path}")


class TimingAnalyzer:
    """Eğitim süre analizi için sınıf."""

    def __init__(self):
        self.times = {}
        self.logger = logging.getLogger(__name__)

    def start_timer(self, name):
        """Timer başlat."""
        self.times[name] = {'start': time.time()}
        self.logger.info(f"Started timing: {name}")

    def end_timer(self, name):
        """Timer durdur."""
        if name in self.times and 'start' in self.times[name]:
            self.times[name]['end'] = time.time()
            self.times[name]['duration'] = self.times[name]['end'] - self.times[name]['start']
            self.logger.info(f"Finished timing: {name} - Duration: {self.times[name]['duration']:.2f} seconds")
        else:
            self.logger.warning(f"Timer {name} was not started")

    def get_timing_report(self):
        """Zamanlama raporunu döndür."""
        report = {}
        total_time = 0
        for name, data in self.times.items():
            if 'duration' in data:
                report[name] = data['duration']
                total_time += data['duration']
        report['total_time'] = total_time
        return report

    def save_timing_report(self, export_dir):
        """Zamanlama raporunu kaydet."""
        report = self.get_timing_report()
        report_path = os.path.join(export_dir, 'timing_report.json')

        # JSON format
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        readable_path = os.path.join(export_dir, 'timing_report.txt')
        with open(readable_path, 'w') as f:
            f.write("TIMING ANALYSIS REPORT\n")
            f.write("=" * 40 + "\n")
            for name, duration in report.items():
                if name != 'total_time':
                    minutes = duration // 60
                    seconds = duration % 60
                    f.write(f"{name}: {duration:.2f}s ({int(minutes)}m {seconds:.1f}s)\n")
            f.write("-" * 40 + "\n")
            total_minutes = report['total_time'] // 60
            total_seconds = report['total_time'] % 60
            f.write(f"TOTAL TIME: {report['total_time']:.2f}s ({int(total_minutes)}m {total_seconds:.1f}s)\n")

        self.logger.info(f"Timing reports saved to {export_dir}")


# Orijinal sınıflarınız (MemoryEfficientDataLoader, FLModel, InitialModelTrainer) burada devam eder...
# (Bu kısımları aynen koruyarak, sadece InitialModelTrainer sınıfında bazı değişiklikler yapacağız)

class MemoryEfficientDataLoader:
    """Büyük veri setleri için bellek verimli veri yükleme sınıfı."""

    def __init__(self, logger, max_sequence_length=500, batch_size=32):
        self.logger = logger
        self.max_sequence_length = max_sequence_length
        self.batch_size = batch_size

    def find_files(self, directory, file_extension):
        """Belirtilen dizindeki dosya yollarını bulur, ama içeriklerini yüklemez."""
        self.logger.info(f"Finding files in {directory} with extension {file_extension}")
        pattern = os.path.join(directory, f'**/*{file_extension}')
        files = glob.glob(pattern, recursive=True)

        if not files:
            self.logger.warning(f"No files found in {directory} with extension {file_extension}")
            # Alternatif uzantılar dene
            for alt_ext in ['.html', '.htm', '.txt']:
                if alt_ext != file_extension:
                    pattern = os.path.join(directory, f'**/*{alt_ext}')
                    files = glob.glob(pattern, recursive=True)
                    if files:
                        self.logger.info(f"Found {len(files)} files with extension {alt_ext} instead")
                        break

        self.logger.info(f"Found {len(files)} files in {directory}")
        return files

    def load_file_content(self, filepath):
        """Tek bir dosyanın içeriğini yükler."""
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                return file.read()
        except UnicodeDecodeError:
            try:
                with open(filepath, 'r', encoding='latin1') as file:
                    return file.read()
            except Exception as e:
                self.logger.error(f"Error reading file {filepath}: {str(e)}")
                return ""

    def create_tf_dataset(self, file_paths, labels, tokenizer):
        """Create TensorFlow dataset with preprocessing done outside TF."""
        self.logger.info(f"Creating dataset from {len(file_paths)} files with preprocessing outside TF")

        # Process in smaller chunks to manage memory
        chunk_size = 20000
        all_sequences = []
        all_labels = []

        for i in range(0, len(file_paths), chunk_size):
            chunk_paths = file_paths[i:i + chunk_size]
            chunk_labels = labels[i:i + chunk_size]

            self.logger.info(
                f"Processing chunk {i // chunk_size + 1}/{(len(file_paths) + chunk_size - 1) // chunk_size}")

            for filepath, label in tqdm(zip(chunk_paths, chunk_labels), total=len(chunk_paths)):
                # Load and preprocess text
                content = self.load_file_content(filepath)

                # Basic text preprocessing
                content = content.lower()
                # Clean special characters (similar to your regex in the original code)
                for char in '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n':
                    content = content.replace(char, ' ')
                # Remove extra spaces
                content = ' '.join(content.split())

                # Tokenize
                sequence = tokenizer.texts_to_sequences([content])[0]

                # Padding
                if len(sequence) > self.max_sequence_length:
                    padded = sequence[:self.max_sequence_length]
                else:
                    padded = sequence + [0] * (self.max_sequence_length - len(sequence))

                all_sequences.append(padded)
                all_labels.append(label)

            # Force garbage collection after each chunk
            gc.collect()

        self.logger.info(f"Finished processing {len(all_sequences)} examples")

        # Convert to numpy arrays
        sequences_array = np.array(all_sequences, dtype=np.int32)
        labels_array = np.array(all_labels, dtype=np.float32)

        # Create TensorFlow dataset
        dataset = tf.data.Dataset.from_tensor_slices((sequences_array, labels_array))

        # Shuffle and batch
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        self.logger.info("Dataset creation complete")
        return dataset


class FLModel(tf.Module):
    def __init__(self, keras_model, learning_rate=0.001):
        super().__init__(name='fl_model')
        self.model = keras_model
        self.optimizer = Adam(learning_rate=learning_rate)
        self.loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    @tf.function(input_signature=[
        tf.TensorSpec([1, 500], tf.int32, name="train_input"),
        tf.TensorSpec([1], tf.float32, name="train_label")
    ])
    def train(self, train_input, train_label):
        with tf.GradientTape() as tape:
            logits = self.model(train_input, training=True)
            loss = self.loss_fn(train_label, logits)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        prediction = tf.nn.sigmoid(logits)

        loss_tensor = tf.reshape(loss, [1])
        prediction_tensor = tf.reshape(prediction, [1])

        return {
            'train_loss': tf.identity(loss_tensor, name="train_loss"),
            'train_prediction': tf.identity(prediction_tensor, name="train_prediction")
        }

    @tf.function(input_signature=[
        tf.TensorSpec([1, 500], tf.int32, name="infer_input")
    ])
    def infer(self, infer_input):
        logits = self.model(infer_input, training=False)
        prediction = tf.nn.sigmoid(logits)
        return {"infer_output": tf.identity(prediction, name="infer_output")}

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[], dtype=tf.string, name="checkpoint_path")
    ])
    def save(self, checkpoint_path):
        tensor_names = [weight.name for weight in self.model.weights]
        tensors_to_save = [weight.read_value() for weight in self.model.weights]

        tf.raw_ops.Save(
            filename=checkpoint_path,
            tensor_names=tensor_names,
            data=tensors_to_save,
            name='save'
        )

        return {
            "saved_path": tf.identity(checkpoint_path, name="saved_path"),
            "version": tf.identity(tf.constant(1.0), name="version"),
            "timestamp": tf.identity(tf.constant(datetime.now().isoformat()), name="timestamp")
        }


class InitialModelTrainer:
    def __init__(self, params):
        self.params = params
        self.tokenizer = None
        self.model = None
        self.word_freq = None
        self.logger = logging.getLogger(__name__)
        self.logger.info("InitialModelTrainer initialized with params: %s", params)

        # Performance tracking için
        self.training_metrics = {}

    def load_data_efficiently(self, train_dir, test_dir, train_sample_ratio=1.0, test_sample_ratio=1.0):
        """Bellek dostu ve dengeli veri yükleme metodu."""
        self.logger.info(f"Starting memory-efficient data loading with balanced classes")
        self.logger.info(f"Sample ratios - Train: {train_sample_ratio}, Test: {test_sample_ratio}")

        try:
            # MemoryEfficientDataLoader oluştur
            data_loader = MemoryEfficientDataLoader(
                logger=self.logger,
                max_sequence_length=self.params["max_sequence_length"],
                batch_size=self.params["batch_size"]
            )

            # Tüm dosya yollarını bul
            phishing_train_files = data_loader.find_files(os.path.join(train_dir, 'phishing'), '.html')
            legal_train_files = data_loader.find_files(os.path.join(train_dir, 'legal'), '.txt')
            phishing_test_files = data_loader.find_files(os.path.join(test_dir, 'phishingTest'), '.html')
            legal_test_files = data_loader.find_files(os.path.join(test_dir, 'legalTest'), '.html')

            # Alternatif klasörleri de kontrol et
            if not legal_test_files:
                legal_test_files = data_loader.find_files(os.path.join(test_dir, 'legal'), '.txt')

            # Orijinal dosya sayılarını log
            original_phishing_train_count = len(phishing_train_files)
            original_legal_train_count = len(legal_train_files)
            original_phishing_test_count = len(phishing_test_files)
            original_legal_test_count = len(legal_test_files)

            # EĞİTİM VERİSİ İÇİN SAMPLE RATIO UYGULA
            if train_sample_ratio < 1.0:
                self.logger.info(f"Directly sampling train data: {train_sample_ratio * 100}% of files")

                # Her sınıftan doğrudan sample_ratio kadar örnek seç
                phishing_train_size = max(10, int(len(phishing_train_files) * train_sample_ratio))
                legal_train_size = max(10, int(len(legal_train_files) * train_sample_ratio))

                # Dosyaları örnekle
                phishing_train_files = random.sample(phishing_train_files, phishing_train_size)
                legal_train_files = random.sample(legal_train_files, legal_train_size)

                self.logger.info(
                    f"After train sampling: {len(phishing_train_files)} phishing, {len(legal_train_files)} legal")

            # TEST VERİSİ İÇİN SAMPLE RATIO UYGULA
            if test_sample_ratio < 1.0:
                self.logger.info(f"Directly sampling test data: {test_sample_ratio * 100}% of files")

                # Her sınıftan doğrudan sample_ratio kadar örnek seç
                phishing_test_size = max(10, int(len(phishing_test_files) * test_sample_ratio))
                legal_test_size = max(10, int(len(legal_test_files) * test_sample_ratio))

                # Dosyaları örnekle
                phishing_test_files = random.sample(phishing_test_files, phishing_test_size)
                legal_test_files = random.sample(legal_test_files, legal_test_size)

                self.logger.info(
                    f"After test sampling: {len(phishing_test_files)} phishing, {len(legal_test_files)} legal")

            # Eğitim verilerini dengele
            min_train_count = min(len(phishing_train_files), len(legal_train_files))
            self.logger.info(f"Balancing train files to {min_train_count} per class")

            # Eğer bir sınıf daha büyükse, random sampling ile küçült
            if len(phishing_train_files) > min_train_count:
                phishing_train_files = random.sample(phishing_train_files, min_train_count)
            if len(legal_train_files) > min_train_count:
                legal_train_files = random.sample(legal_train_files, min_train_count)

            # Test verilerini dengele
            min_test_count = min(len(phishing_test_files), len(legal_test_files))
            self.logger.info(f"Balancing test files to {min_test_count} per class")

            # Eğer bir sınıf daha büyükse, random sampling ile küçült
            if len(phishing_test_files) > min_test_count:
                phishing_test_files = random.sample(phishing_test_files, min_test_count)
            if len(legal_test_files) > min_test_count:
                legal_test_files = random.sample(legal_test_files, min_test_count)

            # Dataset metrics kaydet
            self.training_metrics['dataset_info'] = {
                'original_train_phishing': original_phishing_train_count,
                'original_train_legal': original_legal_train_count,
                'original_test_phishing': original_phishing_test_count,
                'original_test_legal': original_legal_test_count,
                'final_train_phishing': len(phishing_train_files),
                'final_train_legal': len(legal_train_files),
                'final_test_phishing': len(phishing_test_files),
                'final_test_legal': len(legal_test_files),
                'train_sample_ratio': train_sample_ratio,
                'test_sample_ratio': test_sample_ratio
            }

            # Dengeleme sonuçlarını logla
            self.logger.info("Data processing results:")
            self.logger.info(
                f"Training data: {original_phishing_train_count} phishing files → sampled to {len(phishing_train_files)}")
            self.logger.info(
                f"Training data: {original_legal_train_count} legal files → sampled to {len(legal_train_files)}")
            self.logger.info(
                f"Test data: {original_phishing_test_count} phishing files → sampled to {len(phishing_test_files)}")
            self.logger.info(f"Test data: {original_legal_test_count} legal files → sampled to {len(legal_test_files)}")

            # Etiketleri oluştur
            phishing_train_labels = [1.0] * len(phishing_train_files)
            legal_train_labels = [0.0] * len(legal_train_files)
            phishing_test_labels = [1.0] * len(phishing_test_files)
            legal_test_labels = [0.0] * len(legal_test_files)

            # Eğitim ve test dosyalarını birleştir
            train_files = phishing_train_files + legal_train_files
            train_labels = phishing_train_labels + legal_train_labels
            test_files = phishing_test_files + legal_test_files
            test_labels = phishing_test_labels + legal_test_labels

            # Aynı indeksleri kullanan listeleri karıştır
            train_combined = list(zip(train_files, train_labels))
            random.shuffle(train_combined)
            train_files, train_labels = zip(*train_combined)

            test_combined = list(zip(test_files, test_labels))
            random.shuffle(test_combined)
            test_files, test_labels = zip(*test_combined)

            # Tokenizer oluştur - ŞİMDİ ÖRNEKLENMİŞ VERİ ÜZERİNDE
            all_files = train_files + test_files
            self.logger.info(f"Creating tokenizer from {len(all_files)} sampled and balanced files")

            # Henüz tokenizer oluşturulmamışsa önce onu oluştur
            if self.tokenizer is None:
                self.logger.info("Creating tokenizer")
                self._create_tokenizer_from_files(all_files)

            # TF Datasets oluştur
            self.logger.info(
                f"Creating datasets from {len(train_files)} training files and {len(test_files)} test files")
            train_dataset = data_loader.create_tf_dataset(train_files, train_labels, self.tokenizer)
            test_dataset = data_loader.create_tf_dataset(test_files, test_labels, self.tokenizer)

            self.logger.info("Memory-efficient balanced data loading completed")
            return train_dataset, test_dataset
        except Exception as e:
            self.logger.error(f"Error in memory-efficient data loading: {str(e)}", exc_info=True)
            raise

    def _create_tokenizer_from_files(self, file_paths, chunk_size=None, min_freq=None):
        """Dosya yollarından bellek verimli tokenizer oluşturur ve DOĞRU vocabulary size kaydeder."""

        # Yeni params yapısından değerleri al
        if chunk_size is None:
            if hasattr(self, 'full_params'):
                chunk_size = self.full_params["data"]["chunk_size"]
            else:
                chunk_size = 20000

        if min_freq is None:
            if hasattr(self, 'full_params'):
                min_freq = self.full_params["tokenizer"]["min_freq"]
            else:
                min_freq = 60

        self.logger.info(f"Building vocabulary from {len(file_paths)} files with chunk size {chunk_size}")

        # İteratif kelime sayımı için sözlük
        word_counts = {}
        total_words_processed = 0

        # Dosya yollarını chunk'lara böl
        for i in range(0, len(file_paths), chunk_size):
            chunk = file_paths[i:i + chunk_size]
            chunk_end = min(i + chunk_size, len(file_paths))
            self.logger.info(
                f"Processing vocabulary chunk {i // chunk_size + 1}/{(len(file_paths) + chunk_size - 1) // chunk_size} (files {i}-{chunk_end})")

            # Bu chunk'taki dosyaları yükle
            texts = []
            for filepath in tqdm(chunk, desc=f"Loading files for vocabulary chunk {i // chunk_size + 1}"):
                try:
                    with open(filepath, 'r', encoding='utf-8') as file:
                        content = file.read()
                except UnicodeDecodeError:
                    try:
                        with open(filepath, 'r', encoding='latin1') as file:
                            content = file.read()
                    except Exception as e:
                        self.logger.error(f"Error reading file {filepath}: {str(e)}")
                        content = ""

                # Temel ön işleme (küçük harfe çevirme)
                content = content.lower()
                texts.append(content)

            # Geçici tokenizer oluştur ve kelime sayımlarını yap
            temp_tokenizer = Tokenizer(oov_token="<OOV>",
                                       filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
            temp_tokenizer.fit_on_texts(texts)

            # Bu chunk'ın kelime istatistikleri (sadece log için)
            chunk_total_words = sum(temp_tokenizer.word_counts.values())
            total_words_processed += chunk_total_words

            # Kelime sayımlarını birleştir
            for word, count in temp_tokenizer.word_counts.items():
                if word in word_counts:
                    word_counts[word] += count
                else:
                    word_counts[word] = count

            # Belleği temizle
            del texts, temp_tokenizer
            gc.collect()

        # İşlem sonrası toplam kelime istatistikleri
        unique_words_count = len(word_counts)
        self.logger.info(f"\n=== Vocabulary Statistics ===")
        self.logger.info(f"Total unique words: {unique_words_count}")
        self.logger.info(f"Total words processed: {total_words_processed}")

        # Kelime uzunluğu istatistikleri
        word_lengths = [len(word) for word in word_counts.keys()]
        avg_word_length = sum(word_lengths) / len(word_lengths) if word_lengths else 0
        self.logger.info(f"Word length statistics:")
        self.logger.info(f"  - Average word length: {avg_word_length:.2f} characters")
        self.logger.info(f"  - Shortest word: {min(word_lengths)} characters")
        self.logger.info(f"  - Longest word: {max(word_lengths)} characters")

        # Frekans dağılımı istatistikleri
        freq_bins = {
            "1-5": 0,
            "6-10": 0,
            "11-25": 0,
            "26-50": 0,
            "51-100": 0,
            "101-500": 0,
            "501-1000": 0,
            "1001+": 0
        }

        for count in word_counts.values():
            if count <= 5:
                freq_bins["1-5"] += 1
            elif count <= 10:
                freq_bins["6-10"] += 1
            elif count <= 25:
                freq_bins["11-25"] += 1
            elif count <= 50:
                freq_bins["26-50"] += 1
            elif count <= 100:
                freq_bins["51-100"] += 1
            elif count <= 500:
                freq_bins["101-500"] += 1
            elif count <= 1000:
                freq_bins["501-1000"] += 1
            else:
                freq_bins["1001+"] += 1

        self.logger.info("Word frequency distribution:")
        for bin_name, count in freq_bins.items():
            percentage = (count / unique_words_count) * 100
            self.logger.info(f"  - {bin_name}: {count} words ({percentage:.2f}%)")

        # CRITICAL FIX: Kelime filtreleme uygulanıyor
        self.logger.info(f"\n=== Applying Vocabulary Filtering ===")
        self.logger.info(f"Filter criteria: word length 3-25 chars, min frequency {min_freq}")

        # IMPROVED FILTERING: Rakam-only kelimeleri filtrele
        self.logger.info(f"\n=== Applying Enhanced Vocabulary Filtering ===")
        self.logger.info(f"Filter criteria: word length 3-25 chars, min frequency {min_freq}, no digit-only words")

        before_filtering = len(word_counts)

        # Filtering statistics için
        digit_only_count = 0
        repetitive_count = 0
        short_words_count = 0
        low_freq_count = 0

        filtered_words = {}

        for word, count in word_counts.items():
            # Temel kontroller
            if word is None or not isinstance(word, str) or len(word.strip()) == 0:
                continue

            # Kelime uzunluğu kontrolü
            if not (3 <= len(word) <= 25):
                short_words_count += 1
                continue

            # Frekans kontrolü
            if count < min_freq:
                low_freq_count += 1
                continue

            # YENİ: Sadece rakamlardan oluşan kelimeleri filtrele
            if word.isdigit():
                digit_only_count += 1
                continue

            # YENİ: Tek karakter tekrarı filtrele (örn: "aaaa", "bbbb", "----")
            if len(set(word)) == 1 and len(word) > 3:
                repetitive_count += 1
                continue

            # Geçerli kelime
            filtered_words[word] = count

        after_filtering = len(filtered_words)
        removed_words = before_filtering - after_filtering

        # Detaylı filtreleme istatistikleri
        self.logger.info(f"Words before filtering: {before_filtering}")
        self.logger.info(f"Words after filtering: {after_filtering}")
        self.logger.info(f"Words removed: {removed_words} ({(removed_words / before_filtering) * 100:.2f}%)")
        self.logger.info(f"  - Removed due to digit-only: {digit_only_count}")
        self.logger.info(f"  - Removed due to repetitive chars: {repetitive_count}")
        self.logger.info(f"  - Removed due to length: {short_words_count}")
        self.logger.info(f"  - Removed due to low frequency: {low_freq_count}")

        # CRITICAL: Yeni tokenizer oluştur - SADECE FİLTRELENMİŞ KELİMELERLE
        self.tokenizer = Tokenizer(oov_token="<OOV>",
                                   filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')

        # Filtered kelimelerden word_index oluştur
        sorted_filtered_words = sorted(filtered_words.items(), key=lambda x: x[1], reverse=True)

        # Word index oluştur: 1'den başlayarak
        new_word_index = {}
        for idx, (word, _) in enumerate(sorted_filtered_words):
            new_word_index[word] = idx + 1

        # OOV token'ı en sona ekle
        oov_index = len(new_word_index) + 1
        new_word_index["<OOV>"] = oov_index

        # Tokenizer'ı ayarla
        self.tokenizer.word_index = new_word_index
        self.tokenizer.index_word = {v: k for k, v in new_word_index.items()}
        self.word_freq = filtered_words

        # CRITICAL FIX: DOĞRU vocabulary size hesapla
        # Model'in göreceği vocabulary size = filtered words + OOV + padding(0)
        actual_vocab_size = len(new_word_index) + 1  # +1 for padding token (index 0)
        self.params["vocab_size"] = actual_vocab_size

        # Vocabulary metrics kaydet
        self.training_metrics['vocabulary_info'] = {
            'original_unique_words': unique_words_count,
            'total_words_processed': total_words_processed,
            'words_before_filtering': before_filtering,
            'words_after_filtering': after_filtering,
            'words_removed': removed_words,
            'avg_word_length': avg_word_length,
            'min_word_frequency': min_freq,
            'frequency_distribution': freq_bins,
            'actual_vocabulary_size': actual_vocab_size,  # DOĞRU DEĞER
            'max_token_index': max(new_word_index.values()),  # En büyük token ID
            'oov_token_index': oov_index,
            'includes_padding_token': True
        }

        self.logger.info(f"\n=== FINAL TOKENIZER CONFIGURATION ===")
        self.logger.info(f"Filtered vocabulary words: {after_filtering}")
        self.logger.info(f"OOV token index: {oov_index}")
        self.logger.info(f"Max token index in vocabulary: {max(new_word_index.values())}")
        self.logger.info(f"ACTUAL MODEL VOCABULARY SIZE: {actual_vocab_size}")
        self.logger.info(f"Token range: 0 (padding) to {max(new_word_index.values())} (OOV)")

        # Filtered kelimeler için frekans istatistikleri
        if filtered_words:
            freq_values = list(filtered_words.values())
            avg_freq = sum(freq_values) / len(freq_values)
            median_freq = sorted(freq_values)[len(freq_values) // 2]
            self.logger.info(f"\n=== Filtered Words Statistics ===")
            self.logger.info(f"Average frequency: {avg_freq:.2f}")
            self.logger.info(f"Median frequency: {median_freq}")
            self.logger.info(f"Min frequency: {min(freq_values)}")
            self.logger.info(f"Max frequency: {max(freq_values)}")

        # En sık karşılaşılan kelimelerden örnekler
        if filtered_words:
            top_filtered = sorted(filtered_words.items(), key=lambda x: x[1], reverse=True)[:20]
            self.logger.info("\n=== Top 20 Most Frequent Filtered Words ===")
            for i, (word, count) in enumerate(top_filtered):
                token_id = new_word_index.get(word, "UNKNOWN")
                self.logger.info(f"{i + 1}. '{word}' (ID: {token_id}): {count} occurrences")

        return self.tokenizer

    def create_model(self):
        self.logger.info("Creating model")
        try:
            model = tf.keras.Sequential([
                tf.keras.layers.Embedding(
                    input_dim=self.params["vocab_size"],
                    output_dim=self.params["embedding_output_dim"]
                ),
                tf.keras.layers.Conv1D(
                    filters=self.params["filters_0"],
                    kernel_size=self.params["kernel_size_0"],
                    activation='relu'
                ),
                tf.keras.layers.MaxPooling1D(pool_size=2),
                tf.keras.layers.GlobalMaxPooling1D(),
                tf.keras.layers.Dense(
                    units=self.params["dense_units_0"],
                    activation=self.params["dense_activation_0"]
                ),
                tf.keras.layers.Dropout(self.params["dropout_rate_0"]),
                tf.keras.layers.Dense(1, activation=None)
            ])

            model.compile(
                optimizer=Adam(learning_rate=self.params["learning_rate"]),
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=['accuracy']
            )

            # Model architecture kaydet
            self.training_metrics['model_info'] = {
                'vocab_size': self.params["vocab_size"],
                'embedding_dim': self.params["embedding_output_dim"],
                'conv_filters': self.params["filters_0"],
                'conv_kernel_size': self.params["kernel_size_0"],
                'dense_units': self.params["dense_units_0"],
                'dropout_rate': self.params["dropout_rate_0"],
                'learning_rate': self.params["learning_rate"],
                'total_parameters': model.count_params(),
                'trainable_parameters': sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
            }

            self.model = model
            self.logger.info("Model created successfully")
            self.logger.info(f"Total parameters: {model.count_params()}")
            return model
        except Exception as e:
            self.logger.error("Error creating model: %s", str(e), exc_info=True)
            raise

    def train_model_with_datasets(self, train_dataset, test_dataset):
        """TF Datasets ile model eğitimi."""
        self.logger.info("Starting model training with TF Datasets and performance tracking")
        try:
            callbacks = [
                EarlyStopping(
                    monitor='val_accuracy',
                    patience=5,
                    restore_best_weights=True,
                    mode='max',
                    min_delta=0.001,
                    verbose=1
                ),
            ]

            history = self.model.fit(
                train_dataset,
                epochs=self.params["epoch"],
                validation_data=test_dataset,
                callbacks=callbacks,
                verbose=1
            )

            # Training history kaydet
            self.training_metrics['training_history'] = {
                'epochs_completed': len(history.history['loss']),
                'final_train_loss': float(history.history['loss'][-1]),
                'final_val_loss': float(history.history['val_loss'][-1]),
                'final_train_accuracy': float(history.history['accuracy'][-1]),
                'final_val_accuracy': float(history.history['val_accuracy'][-1]),
                'best_val_accuracy': float(max(history.history['val_accuracy'])),
                'best_epoch': int(np.argmax(history.history['val_accuracy'])) + 1,
                'early_stopping_delta': 0.001
            }

            self.logger.info("Model training with TF Datasets completed")
            self.logger.info(
                f"Best validation accuracy: {self.training_metrics['training_history']['best_val_accuracy']:.4f} at epoch {self.training_metrics['training_history']['best_epoch']}")
            self.logger.info(f"Early stopping triggered only if improvement < 0.1% (min_delta=0.001)")

            return history
        except Exception as e:
            self.logger.error(f"Error training model with datasets: {str(e)}", exc_info=True)
            raise

    def export_federated_model_and_tokenizer(self, export_dir):
        logger = logging.getLogger(__name__)
        logger.info("Starting federated model export with CORRECTED vocabulary size")

        try:
            if not os.path.exists(export_dir):
                os.makedirs(export_dir)

            SAVED_MODEL_DIR = os.path.join(export_dir, "saved_model")

            # FL Model oluştur (training operasyonları ile)
            fl_model = FLModel(self.model, self.params["learning_rate"])

            # SavedModel olarak kaydet
            tf.saved_model.save(
                fl_model,
                SAVED_MODEL_DIR,
                signatures={
                    "train": fl_model.train.get_concrete_function(),
                    "infer": fl_model.infer.get_concrete_function(),
                    "save": fl_model.save.get_concrete_function(),
                }
            )

            # TFLite dönüşümü
            converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_DIR)

            # Training operasyonları için SELECT_TF_OPS gerekli
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,
                tf.lite.OpsSet.SELECT_TF_OPS  # Training ops için gerekli
            ]

            # Float16 quantization
            use_float16 = True
            if hasattr(self, 'full_params'):
                use_float16 = self.full_params["export"]["use_float16"]

            if use_float16:
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.target_spec.supported_types = [tf.float16]
                logger.info("Using float16 quantization for model export")
            else:
                logger.info("No quantization applied for model export")

            # On-device training için gerekli
            converter.experimental_enable_resource_variables = True

            # TFLite modeli oluştur
            tflite_model = converter.convert()

            # federated_model.tflite dosyasını kaydet
            tflite_path = os.path.join(export_dir, "federated_model.tflite")
            with open(tflite_path, 'wb') as f:
                f.write(tflite_model)

            logger.info("federated_model.tflite saved to %s", tflite_path)
            logger.info("Model contains training operations - requires Flex delegate on Android")

            # Model boyutunu logla
            model_size_mb = os.path.getsize(tflite_path) / (1024 * 1024)
            logger.info(f"Model size: {model_size_mb:.2f} MB")

            # CRITICAL FIX: DOĞRU vocabulary bilgilerini kaydet
            # Model'in gerçekte desteklediği vocabulary size'ı kullan
            actual_vocab_size = self.params["vocab_size"]
            max_token_id = max(self.tokenizer.word_index.values())

            tokenizer_config = {
                'word_index': self.tokenizer.word_index,
                'word_counts': self.word_freq,
                'max_sequence_length': self.params["max_sequence_length"],
                'oov_token': "<OOV>",
                'oov_token_index': self.tokenizer.word_index["<OOV>"],

                # CORRECTED VALUES
                'vocabulary_size': actual_vocab_size,  # Model'in gerçek vocabulary size'ı
                'max_token_index': max_token_id,  # En büyük geçerli token ID
                'filtered_words_count': len(self.word_freq),  # Filtrelenmiş kelime sayısı
                'total_tokens_including_special': len(self.tokenizer.word_index),  # OOV dahil

                # Model compatibility info
                'token_range': {
                    'min_valid_token': 0,  # Padding token
                    'max_valid_token': max_token_id,  # En büyük geçerli token
                    'padding_token': 0,
                    'oov_token': self.tokenizer.word_index["<OOV>"]
                },

                # Debug info
                'filtering_applied': True,
                'min_frequency_threshold': self.full_params["tokenizer"]["min_freq"] if hasattr(self,
                                                                                                'full_params') else 60,
                'creation_timestamp': datetime.now().isoformat()
            }

            tokenizer_json_path = os.path.join(export_dir, 'tokenizer_config.json')
            try:
                with open(tokenizer_json_path, 'w', encoding='utf-8') as f:
                    json.dump(tokenizer_config, f, ensure_ascii=False, indent=2)
                logger.info("tokenizer_config.json saved to %s", tokenizer_json_path)

                # Tokenizer boyutunu logla
                tokenizer_size_mb = os.path.getsize(tokenizer_json_path) / (1024 * 1024)
                logger.info(f"Tokenizer size: {tokenizer_size_mb:.2f} MB")
                logger.info(f"Total size (model + tokenizer): {model_size_mb + tokenizer_size_mb:.2f} MB")

                # CRITICAL: Doğru vocabulary bilgilerini logla
                logger.info(f"\n=== EXPORTED MODEL VOCABULARY INFO ===")
                logger.info(f"Model vocabulary size: {actual_vocab_size}")
                logger.info(f"Valid token range: 0 to {max_token_id}")
                logger.info(f"OOV token index: {tokenizer_config['oov_token_index']}")
                logger.info(f"Filtered words count: {len(self.word_freq)}")
                logger.info(f"Total tokens (including special): {len(self.tokenizer.word_index)}")

                # Export metrics kaydet
                self.training_metrics['export_info'] = {
                    'model_size_mb': model_size_mb,
                    'tokenizer_size_mb': tokenizer_size_mb,
                    'total_size_mb': model_size_mb + tokenizer_size_mb,
                    'quantization_used': use_float16,
                    'export_timestamp': datetime.now().isoformat(),
                    'requires_flex_delegate': True,
                    'training_operations_included': True,
                    'vocabulary_size': actual_vocab_size,
                    'max_token_index': max_token_id,
                    'token_range_valid': f"0-{max_token_id}"
                }

            except Exception as e:
                logger.error(f"Error saving tokenizer JSON: {str(e)}")
                raise

            logger.info("Federated model and tokenizer config exported successfully with CORRECT vocabulary size")

            # savedModel klasörünü temizle
            import shutil
            shutil.rmtree(SAVED_MODEL_DIR, ignore_errors=True)
            logger.info("Temporary SavedModel directory cleaned up")

        except Exception as e:
            logger.error(f"Error in federated model export: {str(e)}")
            raise

    def save_training_metrics(self, export_dir):
        """Training metrics'leri JSON olarak kaydet."""

        def convert_numpy_types(obj):
            """Numpy tiplerini JSON serializable tiplere çevir."""
            if isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj

        # Numpy tiplerini çevir
        serializable_metrics = convert_numpy_types(self.training_metrics)

        metrics_path = os.path.join(export_dir, 'training_metrics.json')
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_metrics, f, indent=2, ensure_ascii=False)
        self.logger.info(f"Training metrics saved to {metrics_path}")


def main():
    logger = logging.getLogger(__name__)
    logger.info("Starting main execution with comprehensive performance analysis")

    try:
        # Geliştirilmiş model parametreleri
        params = {
            # Veri işleme parametreleri
            "data": {
                "max_sequence_length": 500,
                "chunk_size": 20000,  # Veri işleme chunk boyutu
                "train_sample_ratio": 0.8,  # Eğitim verisi örnekleme oranı
                "test_sample_ratio": 0.4,  # Test verisi örnekleme oranı
                "balance_classes": True,
                "train_dir": "./",
                "test_dir": "./",
                "export_dir": "./",
                "checkpoint_dir": "./"
            },

            # Tokenizer parametreleri
            "tokenizer": {
                "min_freq": 60,  # Minimum kelime frekansı
                "min_word_length": 3,
                "max_word_length": 20,
                "oov_token": "<OOV>"
            },

            # Model mimarisi
            "model": {
                # Embedding katmanı
                "embedding_output_dim": 64,

                # Konvolüsyon katmanı
                "filters_0": 128,
                "kernel_size_0": 5,
                "pool_size_0": 2,

                # Dense katmanı
                "dense_units_0": 64,
                "dense_activation_0": "elu",
                "dropout_rate_0": 0.2
            },

            # Eğitim parametreleri
            "training": {
                "batch_size": 32,
                "epochs": 50,
                "learning_rate": 0.0001,
                "early_stopping_patience": 4
            },

            # Model dönüşüm/export parametreleri
            "export": {
                "use_float16": True,
                "include_metadata": True
            }
        }

        logger.debug("Model parameters: %s", params)

        # Model eğitimi için params'ı uyumlu hale getir
        # Geriye uyumluluk için flat params oluştur
        flat_params = {
            "max_sequence_length": params["data"]["max_sequence_length"],
            "embedding_output_dim": params["model"]["embedding_output_dim"],
            "filters_0": params["model"]["filters_0"],
            "kernel_size_0": params["model"]["kernel_size_0"],
            "dense_units_0": params["model"]["dense_units_0"],
            "dense_activation_0": params["model"]["dense_activation_0"],
            "dropout_rate_0": params["model"]["dropout_rate_0"],
            "learning_rate": params["training"]["learning_rate"],
            "batch_size": params["training"]["batch_size"],
            "epoch": params["training"]["epochs"]
        }

        # Dizin yolları params'tan al
        train_dir = params["data"]["train_dir"]
        test_dir = params["data"]["test_dir"]
        export_dir = params["data"]["export_dir"]
        checkpoint_dir = params["data"]["checkpoint_dir"]

        # Dizinleri oluştur
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(export_dir, exist_ok=True)
        logger.info("Directories created")

        # Model eğitimi
        trainer = InitialModelTrainer(flat_params)

        # Orijinal params yapısını da ekle (ileri kullanım için)
        trainer.full_params = params

        # Performance analyzer oluştur
        performance_analyzer = PerformanceAnalyzer(export_dir)

        # Veri yükleme zamanlaması
        logger.info("Loading data with memory-efficient approach")
        # Bellek dostu veri yükleme (sample_ratio params'tan al)
        train_dataset, test_dataset = trainer.load_data_efficiently(
            train_dir,
            test_dir,
            train_sample_ratio=params["data"]["train_sample_ratio"],
            test_sample_ratio=params["data"]["test_sample_ratio"]
        )

        # Model oluşturma zamanlaması
        logger.info("Creating model")
        trainer.create_model()

        # Model eğitimi zamanlaması
        logger.info("Training model with TF Datasets")
        # TF Datasets ile eğitim
        history = trainer.train_model_with_datasets(
            train_dataset,
            test_dataset
        )

        # Model değerlendirme zamanlaması
        logger.info("Evaluating model performance")
        evaluation_metrics = performance_analyzer.evaluate_model(trainer.model, test_dataset, history)
        # Model export zamanlaması
        logger.info("Exporting federated model and tokenizer config")
        trainer.export_federated_model_and_tokenizer(export_dir)

        # Training metrics'leri kaydet
        trainer.save_training_metrics(export_dir)

        # Final performans raporu
        logger.info("=" * 60)
        logger.info("FINAL PERFORMANCE SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Model Accuracy: {evaluation_metrics['accuracy']:.4f}")
        logger.info(f"Model Precision: {evaluation_metrics['precision']:.4f}")
        logger.info(f"Model Recall: {evaluation_metrics['recall']:.4f}")
        logger.info(f"Model F1-Score: {evaluation_metrics['f1_score']:.4f}")
        logger.info(f"Model ROC AUC: {evaluation_metrics['roc_auc']:.4f}")
        logger.info("=" * 60)

        logger.info("Main execution completed successfully with comprehensive performance analysis")

    except Exception:
        logger.error("An error occurred during execution", exc_info=True)
        raise


if __name__ == "__main__":
    main()
