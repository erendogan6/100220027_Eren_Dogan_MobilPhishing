import gc
import tensorflow as tf
import os
import json
import logging
import sys
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
import glob
import random
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
import time
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve

# Logging konfigürasyonu
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(name)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)


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

    def evaluate_tflite_model(self, tflite_model_path, test_dataset, tokenizer_config=None):
        """TFLite modeli için kapsamlı performans analizi."""
        self.logger.info(f"Starting TFLite model evaluation: {tflite_model_path}")

        # TFLite interpreter oluştur
        interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
        interpreter.allocate_tensors()

        # Model input/output detayları
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        self.logger.info(f"Model input shape: {input_details[0]['shape']}")
        self.logger.info(f"Model output shape: {output_details[0]['shape']}")
        self.logger.info(f"Model input dtype: {input_details[0]['dtype']}")
        self.logger.info(f"Model output dtype: {output_details[0]['dtype']}")
        self.logger.info(f"Model input name: {input_details[0]['name']}")
        self.logger.info(f"Model output name: {output_details[0]['name']}")

        # Test verilerini topla
        y_true = []
        y_pred_probs = []

        total_batches = 0
        for batch_x, batch_y in test_dataset:
            total_batches += 1

        self.logger.info(f"Processing {total_batches} batches for evaluation")

        batch_count = 0
        for batch_x, batch_y in tqdm(test_dataset, desc="Evaluating TFLite model"):
            batch_count += 1

            # Her bir örneği ayrı ayrı işle (TFLite batch processing için)
            for i in range(batch_x.shape[0]):
                # Tek örneği al ve doğru tipte (INT32) ayarla
                single_input = np.expand_dims(batch_x[i].numpy(), axis=0).astype(np.int32)

                # Input detaylarını kontrol et ve uygun tipe çevir
                expected_dtype = input_details[0]['dtype']
                if expected_dtype == np.int32:
                    single_input = single_input.astype(np.int32)
                elif expected_dtype == np.float32:
                    single_input = single_input.astype(np.float32)

                # TFLite modeline input ver
                interpreter.set_tensor(input_details[0]['index'], single_input)
                interpreter.invoke()

                # Output al
                output_data = interpreter.get_tensor(output_details[0]['index'])

                # Output değerini kontrol et ve uygun şekilde işle
                raw_output = output_data[0][0]

                # Eğer output zaten 0-1 arasındaysa sigmoid uygulanmış demektir
                if 0 <= raw_output <= 1:
                    pred_prob = float(raw_output)
                else:
                    # Logit output ise sigmoid uygula
                    # Overflow kontrolü ile
                    if raw_output > 500:  # Very large logit
                        pred_prob = 1.0
                    elif raw_output < -500:  # Very small logit
                        pred_prob = 0.0
                    else:
                        pred_prob = 1.0 / (1.0 + np.exp(-raw_output))  # Sigmoid

                y_pred_probs.append(pred_prob)
                y_true.append(float(batch_y[i].numpy()))

        y_true = np.array(y_true)
        y_pred_probs = np.array(y_pred_probs)
        y_pred = (y_pred_probs > 0.5).astype(int)

        self.logger.info(f"Evaluation completed. Processed {len(y_true)} samples")

        # Temel metrikler
        metrics = self._calculate_metrics(y_true, y_pred, y_pred_probs)

        # TFLite özel bilgileri ekle
        metrics['model_info'] = {
            'model_type': 'TFLite',
            'model_path': tflite_model_path,
            'model_size_mb': os.path.getsize(tflite_model_path) / (1024 * 1024),
            'input_shape': input_details[0]['shape'].tolist(),
            'output_shape': output_details[0]['shape'].tolist(),
            'input_dtype': str(input_details[0]['dtype']),
            'output_dtype': str(output_details[0]['dtype'])
        }

        if tokenizer_config:
            metrics['tokenizer_info'] = {
                'vocabulary_size': tokenizer_config.get('vocabulary_size', 'unknown'),
                'max_sequence_length': tokenizer_config.get('max_sequence_length', 'unknown'),
                'oov_token': tokenizer_config.get('oov_token', 'unknown')
            }

        # Görselleştirmeler
        self._plot_confusion_matrix(y_true, y_pred)
        self._plot_roc_curve(y_true, y_pred_probs)
        self._plot_precision_recall_curve(y_true, y_pred_probs)
        self._plot_prediction_distribution(y_pred_probs, y_true)

        # Rapor kaydet
        self._save_evaluation_report(metrics, y_true, y_pred, model_type="TFLite")

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
        plt.boxplot(data_to_plot, tick_labels=['Legal', 'Phishing'])
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

    def _save_evaluation_report(self, metrics, y_true, y_pred, model_type="Standard"):
        """Detaylı değerlendirme raporu kaydet."""
        report_path = os.path.join(self.export_dir, 'evaluation_report.txt')

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write(f"{model_type.upper()} MODEL PERFORMANCE EVALUATION REPORT\n")
            f.write("=" * 60 + "\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Model bilgileri (eğer varsa)
            if 'model_info' in metrics:
                f.write("MODEL INFORMATION:\n")
                for key, value in metrics['model_info'].items():
                    f.write(f"{key}: {value}\n")
                f.write("\n")

            # Tokenizer bilgileri (eğer varsa)
            if 'tokenizer_info' in metrics:
                f.write("TOKENIZER INFORMATION:\n")
                for key, value in metrics['tokenizer_info'].items():
                    f.write(f"{key}: {value}\n")
                f.write("\n")

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


class MemoryEfficientDataLoader:
    """Büyük veri setleri için bellek verimli veri yükleme sınıfı."""

    def __init__(self, logger, max_sequence_length=500, batch_size=32):
        self.logger = logger
        self.max_sequence_length = max_sequence_length
        self.batch_size = batch_size

    def find_files(self, directory, file_extension):
        """Belirtilen dizindeki dosya yollarını bulur."""
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

    def create_tf_dataset_from_tokenizer(self, file_paths, labels, tokenizer_config):
        """Tokenizer config'i kullanarak TensorFlow dataset oluştur."""
        self.logger.info(f"Creating dataset from {len(file_paths)} files using tokenizer config")

        # Tokenizer'ı reconstruct et
        tokenizer = Tokenizer(oov_token=tokenizer_config.get('oov_token', '<OOV>'))
        tokenizer.word_index = tokenizer_config['word_index']
        tokenizer.index_word = {v: k for k, v in tokenizer_config['word_index'].items()}

        # Process in chunks
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
                # Clean special characters
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
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        self.logger.info("Dataset creation complete")
        return dataset


class TFLiteModelEvaluator:
    """TFLite modellerinin performansını değerlendiren ana sınıf."""

    def __init__(self, model_path, tokenizer_config_path, test_data_dir, export_dir):
        self.model_path = model_path
        self.tokenizer_config_path = tokenizer_config_path
        self.test_data_dir = test_data_dir
        self.export_dir = export_dir
        self.logger = logging.getLogger(__name__)

        # Export dizinini oluştur
        os.makedirs(export_dir, exist_ok=True)

    def load_tokenizer_config(self):
        """Tokenizer config dosyasını yükle."""
        self.logger.info(f"Loading tokenizer config from {self.tokenizer_config_path}")
        try:
            with open(self.tokenizer_config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)

            self.logger.info(f"Tokenizer config loaded successfully")
            self.logger.info(f"Vocabulary size: {config.get('vocabulary_size', 'unknown')}")
            self.logger.info(f"Max sequence length: {config.get('max_sequence_length', 'unknown')}")

            return config
        except Exception as e:
            self.logger.error(f"Error loading tokenizer config: {str(e)}")
            raise

    def load_test_data(self, tokenizer_config, test_sample_ratio=1.0):
        """Test verilerini yükle ve dataset oluştur."""
        self.logger.info("Loading test data for TFLite model evaluation")

        # Data loader oluştur
        max_seq_len = tokenizer_config.get('max_sequence_length', 500)
        data_loader = MemoryEfficientDataLoader(
            logger=self.logger,
            max_sequence_length=max_seq_len,
            batch_size=32
        )

        # Test dosyalarını bul
        phishing_test_files = data_loader.find_files(os.path.join(self.test_data_dir, 'phishingTest'), '.html')
        legal_test_files = data_loader.find_files(os.path.join(self.test_data_dir, 'legalTest'), '.html')

        # Alternatif klasörleri kontrol et
        if not legal_test_files:
            legal_test_files = data_loader.find_files(os.path.join(self.test_data_dir, 'legal'), '.txt')
        if not phishing_test_files:
            phishing_test_files = data_loader.find_files(os.path.join(self.test_data_dir, 'phishing'), '.html')

        self.logger.info(f"Found {len(phishing_test_files)} phishing test files")
        self.logger.info(f"Found {len(legal_test_files)} legal test files")

        # Sample ratio uygula
        if test_sample_ratio < 1.0:
            phishing_size = max(10, int(len(phishing_test_files) * test_sample_ratio))
            legal_size = max(10, int(len(legal_test_files) * test_sample_ratio))

            phishing_test_files = random.sample(phishing_test_files, phishing_size)
            legal_test_files = random.sample(legal_test_files, legal_size)

            self.logger.info(f"After sampling: {len(phishing_test_files)} phishing, {len(legal_test_files)} legal")

        # Verileri dengele
        min_count = min(len(phishing_test_files), len(legal_test_files))
        if len(phishing_test_files) > min_count:
            phishing_test_files = random.sample(phishing_test_files, min_count)
        if len(legal_test_files) > min_count:
            legal_test_files = random.sample(legal_test_files, min_count)

        # Etiketleri oluştur
        phishing_labels = [1.0] * len(phishing_test_files)
        legal_labels = [0.0] * len(legal_test_files)

        # Birleştir ve karıştır
        test_files = phishing_test_files + legal_test_files
        test_labels = phishing_labels + legal_labels

        combined = list(zip(test_files, test_labels))
        random.shuffle(combined)
        test_files, test_labels = zip(*combined)

        self.logger.info(
            f"Final test dataset: {len(test_files)} files ({len(phishing_test_files)} phishing, {len(legal_test_files)} legal)")

        # TF Dataset oluştur
        test_dataset = data_loader.create_tf_dataset_from_tokenizer(test_files, test_labels, tokenizer_config)

        return test_dataset

    def evaluate_model(self, test_sample_ratio=1.0):
        """Ana değerlendirme fonksiyonu."""
        self.logger.info("=" * 60)
        self.logger.info("STARTING TFLITE MODEL EVALUATION")
        self.logger.info("=" * 60)

        try:
            # Tokenizer config yükle
            tokenizer_config = self.load_tokenizer_config()

            # Test verilerini yükle
            test_dataset = self.load_test_data(tokenizer_config, test_sample_ratio)

            # Performance analyzer oluştur
            analyzer = PerformanceAnalyzer(self.export_dir)

            # Model değerlendirmesi yap
            self.logger.info("Starting TFLite model performance evaluation")
            metrics = analyzer.evaluate_tflite_model(
                self.model_path,
                test_dataset,
                tokenizer_config
            )

            # Sonuçları kaydet
            metrics_path = os.path.join(self.export_dir, 'tflite_evaluation_metrics.json')
            with open(metrics_path, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False, default=str)

            self.logger.info(f"Evaluation metrics saved to {metrics_path}")

            # Final sonuçları göster
            self.logger.info("=" * 60)
            self.logger.info("TFLITE MODEL EVALUATION RESULTS")
            self.logger.info("=" * 60)
            self.logger.info(f"Model: {os.path.basename(self.model_path)}")
            self.logger.info(f"Model Size: {metrics['model_info']['model_size_mb']:.2f} MB")
            self.logger.info(f"Test Samples: {metrics['total_samples']}")
            self.logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
            self.logger.info(f"Precision: {metrics['precision']:.4f}")
            self.logger.info(f"Recall: {metrics['recall']:.4f}")
            self.logger.info(f"F1-Score: {metrics['f1_score']:.4f}")
            self.logger.info(f"ROC AUC: {metrics['roc_auc']:.4f}")
            self.logger.info("=" * 60)

            return metrics

        except Exception as e:
            self.logger.error(f"Error during model evaluation: {str(e)}", exc_info=True)
            raise


def main():
    """Ana fonksiyon - TFLite model değerlendirmesi için örnek kullanım."""
    logger.info("TFLite Model Performance Evaluator")

    # KULLANICI AYARLARI - Bu değerleri kendi dosya yollarınıza göre değiştirin
    model_path = "model_v1.5.tflite"  # TFLite model dosyanızın yolu
    tokenizer_config_path = "tokenizer_config.json"  # Tokenizer config dosyanızın yolu
    test_data_dir = "./"  # Test verilerinin bulunduğu ana dizin
    export_dir = "./evaluation_results"  # Sonuçların kaydedileceği dizin
    test_sample_ratio = 1.0  # Test verilerinin ne kadarını kullanacağınız (0.2 = %20, daha hızlı evaluation için)

    try:
        # Model dosyalarının varlığını kontrol et
        if not os.path.exists(model_path):
            logger.error(f"TFLite model file not found: {model_path}")
            logger.info("Please set the correct path to your .tflite model file")
            return

        if not os.path.exists(tokenizer_config_path):
            logger.error(f"Tokenizer config file not found: {tokenizer_config_path}")
            logger.info("Please set the correct path to your tokenizer_config.json file")
            return

        if not os.path.exists(test_data_dir):
            logger.error(f"Test data directory not found: {test_data_dir}")
            logger.info("Please set the correct path to your test data directory")
            return

        # Evaluator oluştur ve çalıştır
        evaluator = TFLiteModelEvaluator(
            model_path=model_path,
            tokenizer_config_path=tokenizer_config_path,
            test_data_dir=test_data_dir,
            export_dir=export_dir
        )

        # Değerlendirmeyi başlat
        metrics = evaluator.evaluate_model(test_sample_ratio=test_sample_ratio)

        logger.info("TFLite model evaluation completed successfully!")
        logger.info(f"Results saved to: {export_dir}")

    except Exception as e:
        logger.error(f"An error occurred during evaluation: {str(e)}", exc_info=True)


if __name__ == "__main__":
    main()