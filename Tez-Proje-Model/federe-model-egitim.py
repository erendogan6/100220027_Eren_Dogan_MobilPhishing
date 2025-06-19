import glob
import os
import tempfile
import numpy as np
import tensorflow as tf
import firebase_admin
from firebase_admin import credentials, firestore, storage
from google.cloud.firestore_v1.query import Query
from datetime import datetime
import logging
import json
import re
from urllib.parse import urlparse
from model_egitim import FLModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


class FederatedLearningServer:
    def __init__(self, service_account_key_path, bucket_name):
        """
        Initialize the server with Firebase credentials

        Args:
            service_account_key_path: Path to Firebase service account key JSON file
            bucket_name: Name of the Firebase Storage bucket
        """
        # Initialize Firebase
        cred = credentials.Certificate(service_account_key_path)
        firebase_admin.initialize_app(cred, {
            'storageBucket': bucket_name  # Use the actual bucket name
        })

        self.db = firestore.client()
        self.bucket = storage.bucket()
        self.temp_dir = tempfile.mkdtemp()
        self.source_model_versions = []
        logger.info(f"Temporary directory created at {self.temp_dir}")

    def get_model_metadata(self, min_version=None, max_models=10):
        """
        Get metadata for models from Firestore

        Args:
            min_version: Minimum version to consider
            max_models: Maximum number of models to retrieve

        Returns:
            List of model metadata documents
        """
        # Use Query.DESCENDING from google.cloud.firestore_v1.query
        query = self.db.collection('models').order_by('version', direction=Query.DESCENDING).limit(max_models)

        if min_version:
            query = query.where('version', '>=', min_version)

        model_docs = query.get()

        metadata_list = []
        for doc in model_docs:
            metadata = doc.to_dict()
            metadata['id'] = doc.id
            metadata_list.append(metadata)

        logger.info(f"Retrieved metadata for {len(metadata_list)} models")
        return metadata_list

    def download_models(self, metadata_list):
        """
        Download model files from Firebase Storage
        """
        model_paths = {}

        for metadata in metadata_list:
            version = metadata.get('version', 'unknown')

            # Get all URLs for this model
            all_urls = metadata.get('allUrls', {})
            if not all_urls:
                model_url = metadata.get('modelUrl')
                if model_url:
                    all_urls = {"checkpoint": model_url}
                else:
                    logger.warning(f"Model v{version} için URL bulunamadı")
                    continue

            # Create directory for this model
            model_dir = os.path.join(self.temp_dir, f"model_v{version}")
            os.makedirs(model_dir, exist_ok=True)

            # Sadece .ckpt dosyalarını bul ve indir
            checkpoint_file = None

            for file_name, file_url in all_urls.items():
                if '.ckpt' in file_name:
                    try:
                        # Parse URL
                        parsed_url = urlparse(file_url)
                        path = parsed_url.path

                        # Extract blob path
                        if '/o/' in path:
                            blob_path = path.split('/o/')[1]
                            blob_path = blob_path.replace('%2F', '/')
                        else:
                            match = re.search(r'models/[^?]+', path)
                            if match:
                                blob_path = match.group(0)
                            else:
                                logger.error(f"URL'den blob yolu çıkarılamadı: {file_url}")
                                continue

                        # Remove query parameters
                        if '?' in blob_path:
                            blob_path = blob_path.split('?')[0]

                        # Dosya adını koru
                        local_file_name = os.path.basename(file_name)
                        local_path = os.path.join(model_dir, local_file_name)

                        # Download file
                        blob = self.bucket.blob(blob_path)
                        blob.download_to_filename(local_path)
                        logger.info(f"İndirildi: {local_path}")

                        # İndirilen dosyanın boyutunu kontrol et
                        if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
                            checkpoint_file = local_path
                            logger.info(f"Geçerli checkpoint dosyası indirildi: {local_path}")
                        else:
                            logger.warning(f"İndirilen dosya geçersiz veya boş: {local_path}")

                    except Exception as e:
                        logger.error(f"Dosya indirme hatası {file_name}: {str(e)}")

            if checkpoint_file:
                model_paths[version] = {'path': checkpoint_file, 'type': 'raw_checkpoint'}
                logger.info(f"Model v{version} için checkpoint dosyası: {checkpoint_file}")

        return model_paths

    def _extract_weights_from_raw_checkpoint(self, checkpoint_path):
        """
        Extract weights directly from raw checkpoint file
        """
        try:
            logger.info(f"Ham checkpoint dosyasından ağırlıklar çıkarılıyor: {checkpoint_path}")

            if not os.path.exists(checkpoint_path):
                logger.error(f"Checkpoint dosyası bulunamadı: {checkpoint_path}")
                return None

            # Dosya boyutunu kontrol et
            file_size = os.path.getsize(checkpoint_path)
            logger.info(f"Checkpoint dosya boyutu: {file_size / (1024 * 1024):.2f} MB")

            # Bilinen ağırlık isimleri ve şekilleri
            weight_specs = [
                {"name": "embedding/embeddings:0", "shape": (415155, 64), "dtype": np.float32},
                {"name": "conv1d/kernel:0", "shape": (5, 64, 128), "dtype": np.float32},
                {"name": "conv1d/bias:0", "shape": (128,), "dtype": np.float32},
                {"name": "dense/kernel:0", "shape": (128, 64), "dtype": np.float32},
                {"name": "dense/bias:0", "shape": (64,), "dtype": np.float32},
                {"name": "dense_1/kernel:0", "shape": (64, 1), "dtype": np.float32},
                {"name": "dense_1/bias:0", "shape": (1,), "dtype": np.float32}
            ]

            # TensorFlow'un düşük seviyeli API'sini kullanarak ağırlıkları çıkarmaya çalış
            weights = {}

            # Ağırlıkları çıkarmak için düşük seviyeli TensorFlow raw ops kullan
            with tf.Graph().as_default():
                with tf.compat.v1.Session() as sess:
                    try:
                        # Her değişken için restore op oluştur ve çalıştır
                        for spec in weight_specs:
                            try:
                                restore_op = tf.raw_ops.Restore(
                                    file_pattern=checkpoint_path,
                                    tensor_name=spec["name"],
                                    dt=tf.as_dtype(spec["dtype"])
                                )

                                # Restore op'u çalıştır
                                tensor_value = sess.run(restore_op)

                                # Beklenen şekil ile uyuştuğundan emin ol
                                if tensor_value.shape != spec["shape"]:
                                    logger.warning(
                                        f"Şekil uyuşmazlığı: {spec['name']} - beklenen: {spec['shape']}, alınan: {tensor_value.shape}")

                                weights[spec["name"]] = tensor_value
                                logger.info(f"Değişken çıkarıldı: {spec['name']} şekli={tensor_value.shape}")

                            except tf.errors.NotFoundError:
                                logger.warning(f"Değişken bulunamadı: {spec['name']}")
                            except Exception as variable_error:
                                logger.error(f"Değişken çıkarma hatası {spec['name']}: {variable_error}")

                        if weights:
                            logger.info(f"Toplam {len(weights)} değişken çıkarıldı")
                            return weights
                        else:
                            logger.warning("Hiçbir değişken çıkarılamadı")

                    except Exception as e:
                        logger.error(f"TensorFlow checkpoint okuma hatası: {e}")

            # Doğrudan binary içeriğini analiz etmeyi dene
            try:
                # Dosyayı binary olarak oku
                with open(checkpoint_path, 'rb') as f:
                    data = f.read()

                # Doğrudan raw binary data'dan ağırlıkları çıkarmak
                # çok karmaşık ve TensorFlow'un iç formatlarına bağlı
                # Bu nedenle hata veriyoruz
                logger.error("Checkpoint formatı desteklenmiyor")
                return None

            except Exception as e:
                logger.error(f"Binary içerik analizi hatası: {e}")
                return None

        except Exception as e:
            logger.error(f"Checkpoint işleme hatası: {e}")
            return None

    def _parse_tf_checkpoint(self, data, checkpoint_path):
        """
        Parse standard TensorFlow checkpoint
        """
        try:
            # Dosyayı farklı bir isimle kopyala (.ckpt, .ckpt.index, .ckpt.data-*)
            dirname = os.path.dirname(checkpoint_path)
            base_name = os.path.basename(checkpoint_path).split('.')[0]

            # Try using Python binary reading approach
            try:
                # Doğrudan checkpoint içeriğinden değişkenleri çıkar
                import struct
                import io

                # Binary stream'i analiz et
                f = io.BytesIO(data)

                # Header'ı oku
                header_size = struct.unpack('<I', f.read(4))[0]
                header = f.read(header_size).decode('utf-8')

                if 'variables' in header:
                    logger.info(f"Checkpoint header: {header[:100]}...")

                    # Header'daki değişken bilgilerini çıkar
                    import json
                    try:
                        header_data = json.loads(header)
                        if 'variables' in header_data:
                            variables = header_data['variables']

                            weights = {}
                            for var_name, var_info in variables.items():
                                # Değişken bilgilerini al
                                dtype = var_info['dtype']
                                shape = var_info['shape']

                                # Değişken datasını oku
                                # Bu kısım checkpoint formatına bağlı olarak değişir
                                # ve karmaşık olabilir

                            return weights
                    except:
                        logger.exception("Header parsing failed")

            except Exception as e:
                logger.error(f"Binary parsing failed: {e}")

            # Son çare: Dosyayı manuel olarak numpy tipine dönüştür
            try:
                from tensorflow.python.framework import tensor_util
                from tensorflow.core.protobuf import tensor_pb2
                import numpy as np

                # Binary dosyayı tensor proto gibi çözmeye çalış
                tensor_proto = tensor_pb2.TensorProto()
                tensor_proto.ParseFromString(data)
                weights = {}

                # Tensör adını ve değerini çıkar
                tensor_name = tensor_proto.name if hasattr(tensor_proto, 'name') else "unknown"
                tensor_value = tensor_util.MakeNdarray(tensor_proto)

                weights[tensor_name] = tensor_value
                logger.info(f"Çıkarılan tensor: {tensor_name} şekli={tensor_value.shape}")

                return weights

            except Exception as e:
                logger.error(f"Tensor proto çözme hatası: {e}")

            raise ValueError("TensorFlow checkpoint çözülemedi")

        except Exception as e:
            logger.error(f"Checkpoint çözme hatası: {e}")
            raise ValueError(f"Checkpoint çözme hatası: {e}")

    def _parse_custom_binary(self, data):
        """
        Parse custom binary format (assuming Android saves in a custom format)
        """
        try:
            import struct
            import numpy as np

            # Dosya formatının bir header ile başladığını varsayalım
            offset = 0
            weights = {}

            while offset < len(data):
                # Format: [name_length:int] [name:string] [shape_length:int] [shape:int[]] [data_length:int] [data:bytes]
                try:
                    # İsim uzunluğunu oku
                    if offset + 4 > len(data):
                        break

                    name_length = struct.unpack('<I', data[offset:offset + 4])[0]
                    offset += 4

                    # İsmi oku
                    if offset + name_length > len(data):
                        break

                    name = data[offset:offset + name_length].decode('utf-8')
                    offset += name_length

                    # Şekil uzunluğunu oku
                    if offset + 4 > len(data):
                        break

                    shape_length = struct.unpack('<I', data[offset:offset + 4])[0]
                    offset += 4

                    # Şekli oku
                    shape = []
                    for i in range(shape_length):
                        if offset + 4 > len(data):
                            break

                        dim = struct.unpack('<I', data[offset:offset + 4])[0]
                        shape.append(dim)
                        offset += 4

                    # Veri uzunluğunu oku
                    if offset + 8 > len(data):
                        break

                    data_length = struct.unpack('<Q', data[offset:offset + 8])[0]
                    offset += 8

                    # Veriyi oku
                    if offset + data_length > len(data):
                        break

                    tensor_data = data[offset:offset + data_length]
                    offset += data_length

                    # NumPy array'e dönüştür
                    tensor = np.frombuffer(tensor_data, dtype=np.float32).reshape(shape)
                    weights[name] = tensor

                    logger.info(f"Çıkarılan değişken: {name} şekli={tensor.shape}")

                except Exception as e:
                    logger.error(f"Değişken çıkarma hatası offset={offset}: {e}")
                    # Bir sonraki değişkene geç
                    offset += 1

            if weights:
                logger.info(f"Toplam {len(weights)} değişken çıkarıldı")
                return weights
            else:
                raise ValueError("Hiç değişken çıkarılamadı")

        except Exception as e:
            logger.error(f"Özel binary format çözme hatası: {e}")
            raise ValueError(f"Özel binary format çözme hatası: {e}")

    def load_model_weights(self, model_paths):
        """
        Load weights from model files
        """
        model_weights = {}

        for version, info in model_paths.items():
            path = info['path']
            file_type = info['type']

            try:
                logger.info(f"Model v{version} için ağırlıklar yükleniyor: {path} (tip: {file_type})")

                if file_type == 'tflite':
                    weights = self._extract_weights_from_tflite(path)
                elif file_type == 'raw_checkpoint':
                    weights = self._extract_weights_from_raw_checkpoint(path)
                else:
                    weights = self._extract_weights_from_checkpoint(path)

                if weights:
                    model_weights[version] = weights
                    logger.info(f"Model v{version} ağırlıkları başarıyla yüklendi - {len(weights)} değişken")
                else:
                    logger.warning(f"Model v{version} için ağırlık yüklenemedi")

            except Exception as e:
                logger.error(f"Model v{version} için ağırlık yükleme hatası: {e}")

        return model_weights

    def _extract_weights_from_tflite(self, tflite_path):
        """
        Improved TFLite weight extraction that handles on-device trained models
        """
        try:
            logger.info(f"Extracting weights from TFLite model: {tflite_path}")

            # Load the TFLite model
            interpreter = tf.lite.Interpreter(model_path=tflite_path)
            interpreter.allocate_tensors()

            # Get tensor details
            tensor_details = interpreter.get_tensor_details()

            # Extract weights
            weights = {}

            # Map to track unique layer names
            layer_name_map = {}

            for tensor in tensor_details:
                # Filter for weight tensors
                if tensor['name'].endswith('/weights') or tensor['name'].endswith('/kernel') or \
                        tensor['name'].endswith('/bias') or 'embedding' in tensor['name'] or \
                        'dense' in tensor['name'] or 'conv' in tensor['name']:

                    tensor_index = tensor['index']
                    tensor_value = interpreter.tensor(tensor_index)()

                    # Create standardized weight name
                    if 'embedding' in tensor['name']:
                        weight_name = 'embedding/embeddings:0'
                    elif '/kernel' in tensor['name'] or '/weights' in tensor['name']:
                        # Extract the layer name
                        if '/kernel' in tensor['name']:
                            layer_name = tensor['name'].split('/kernel')[0]
                        else:
                            layer_name = tensor['name'].split('/weights')[0]

                        # Check for numbered layers (dense_1, conv1d_1, etc)
                        base_name = layer_name.split('_')[0]
                        if base_name in layer_name_map:
                            # Use consistent naming for TF compatibility
                            weight_name = f"{base_name}_{layer_name_map[base_name]}/kernel:0"
                            layer_name_map[base_name] += 1
                        else:
                            weight_name = f"{base_name}/kernel:0"
                            layer_name_map[base_name] = 1
                    elif '/bias' in tensor['name']:
                        layer_name = tensor['name'].split('/bias')[0]
                        base_name = layer_name.split('_')[0]

                        if base_name in layer_name_map:
                            weight_name = f"{base_name}_{layer_name_map[base_name] - 1}/bias:0"
                        else:
                            weight_name = f"{base_name}/bias:0"
                            layer_name_map[base_name] = 1
                    else:
                        # For unknown formats, use the original name
                        weight_name = tensor['name'] + ":0"

                    weights[weight_name] = tensor_value
                    logger.info(f"Extracted weight {weight_name} with shape {tensor_value.shape}")

            # Validate model structure
            required_layers = ['embedding/embeddings:0', 'conv1d/kernel:0', 'dense/kernel:0', 'dense_1/kernel:0']
            missing_layers = [layer for layer in required_layers if not any(layer in key for key in weights.keys())]

            if missing_layers:
                logger.warning(f"Missing expected layers in TFLite model: {missing_layers}")
                # Try to infer missing layers if possible
                # This would depend on your specific model structure

            logger.info(f"Successfully extracted {len(weights)} weights from TFLite model")
            return weights

        except Exception as e:
            logger.error(f"Error extracting weights from TFLite: {e}", exc_info=True)
            return None

    def federated_averaging(self, model_weights_dict, weights=None):
        """
        Robust federated averaging with variable name handling
        """
        if not model_weights_dict:
            raise ValueError("Ortalaması alınacak model ağırlıkları yok")

        # Create equal weights if not specified
        if weights is None:
            weights = {version: 1.0 for version in model_weights_dict.keys()}

        # Normalize weights
        total_weight = sum(weights.values())
        normalized_weights = {k: v / total_weight for k, v in weights.items()}

        # Log the weighting
        for version, weight in normalized_weights.items():
            logger.info(f"Model v{version} ağırlığı: {weight:.4f}")

        # Collect all variable names and standardize them
        all_var_names = set()
        for model_weights in model_weights_dict.values():
            all_var_names.update(model_weights.keys())

        logger.info(f"Tüm modellerde {len(all_var_names)} benzersiz değişken var")

        # Count how many models have each variable
        var_presence = {name: 0 for name in all_var_names}
        for model_weights in model_weights_dict.values():
            for name in model_weights:
                var_presence[name] += 1

        # Sort variables by presence (most common first)
        sorted_vars = sorted(var_presence.items(), key=lambda x: x[1], reverse=True)

        # Get shapes for each variable from the first model that has it
        var_shapes = {}
        for name in all_var_names:
            for model_weights in model_weights_dict.values():
                if name in model_weights:
                    var_shapes[name] = model_weights[name].shape
                    break

        # Initialize aggregated weights - use variables that are present in at least one model
        aggregated_weights = {}

        # Process each variable
        for var_name, presence_count in sorted_vars:
            # Only include variables that appear in at least one model
            if presence_count > 0:
                # Get the shape from our saved shapes
                shape = var_shapes[var_name]

                # Initialize with zeros of the right shape
                aggregated_weights[var_name] = np.zeros(shape, dtype=np.float32)

                # Keep track of total weight used for this variable
                total_var_weight = 0.0

                # Weighted sum for models that have this variable
                for version, model_weights in model_weights_dict.items():
                    if var_name in model_weights:
                        weight_factor = normalized_weights[version]
                        aggregated_weights[var_name] += weight_factor * model_weights[var_name]
                        total_var_weight += weight_factor

                # Renormalize if not all models contributed to this variable
                if total_var_weight > 0 and total_var_weight < 1.0:
                    aggregated_weights[var_name] /= total_var_weight

                logger.info(f"Değişken {var_name}: {presence_count}/{len(model_weights_dict)} model katkısı")

        if not aggregated_weights:
            raise ValueError("Birleştirme için ortak değişken bulunamadı")

        logger.info(f"{len(aggregated_weights)} değişken için birleştirilmiş ağırlıklar oluşturuldu")
        return aggregated_weights

    def save_aggregated_model(self, aggregated_weights, base_model_path):
        """
        Save the aggregated model to a simple .ckpt file compatible with the Android app
        """
        # Find highest source version
        highest_source_version = 0.0
        for version_str in self.source_model_versions:
            try:
                version = float(version_str)
                highest_source_version = max(highest_source_version, version)
            except (ValueError, TypeError):
                pass

        # Increment for new version
        new_version = highest_source_version + 0.1
        formatted_version = f"{new_version:.1f}"

        # Create directory for saving
        save_dir = os.path.join(self.temp_dir, f"aggregated_model_v{formatted_version}")
        os.makedirs(save_dir, exist_ok=True)

        # Save as .ckpt using the same format as Android
        checkpoint_path = os.path.join(save_dir, f"model_v{formatted_version}.ckpt")

        # Save using TensorFlow ops - aynı Android'in kullandığı format
        with tf.Graph().as_default():
            with tf.compat.v1.Session() as sess:
                # Create variables from aggregated weights
                tf_vars = []
                for name, value in aggregated_weights.items():
                    # Strip :0 suffix if present
                    clean_name = name.split(':')[0] if ':' in name else name
                    var = tf.Variable(value, name=clean_name)
                    tf_vars.append(var)

                # Initialize variables
                sess.run(tf.compat.v1.global_variables_initializer())

                # Extract names and values for tf.raw_ops.Save
                tensor_names = [var.name for var in tf_vars]
                tensor_values = [var.read_value() for var in tf_vars]

                # Save using raw op - exactly like Android
                save_op = tf.raw_ops.Save(
                    filename=checkpoint_path,
                    tensor_names=tensor_names,
                    data=tensor_values
                )

                # Run the save op
                sess.run(save_op)

                logger.info(f"Birleştirilmiş model .ckpt formatında kaydedildi: {checkpoint_path}")

        # Verify the file was created
        if os.path.exists(checkpoint_path) and os.path.getsize(checkpoint_path) > 0:
            logger.info(
                f"Checkpoint dosyası başarıyla oluşturuldu: {os.path.getsize(checkpoint_path) / (1024 * 1024):.2f} MB")
        else:
            logger.error("Checkpoint dosyası oluşturulamadı veya boş")
            raise ValueError("Checkpoint dosyası oluşturulamadı")

        return save_dir, formatted_version

    def upload_aggregated_model(self, model_dir, version, aggregated_weights):
        """
        Upload the aggregated model to Firebase Storage and update Firestore
        """
        # Find the checkpoint file
        checkpoint_file = os.path.join(model_dir, f"model_v{version}.ckpt")

        if not os.path.exists(checkpoint_file):
            logger.error(f"Yüklenecek checkpoint dosyası bulunamadı: {checkpoint_file}")
            return None

        # Initialize URLs dictionary
        urls = {}

        # First, create and upload TFLite model
        tflite_output_dir = os.path.join(model_dir, "android_output")
        os.makedirs(tflite_output_dir, exist_ok=True)

        try:
            # Create TFLite model
            tflite_path = self.prepare_model_for_android(model_dir, tflite_output_dir)

            if tflite_path and os.path.exists(tflite_path):
                # Upload TFLite model to Firebase
                tflite_storage_path = f"models/aggregated_model_v{version}/model_v{version}.tflite"
                tflite_blob = self.bucket.blob(tflite_storage_path)
                tflite_blob.upload_from_filename(tflite_path)
                tflite_blob.make_public()
                urls[f"model_v{version}.tflite"] = tflite_blob.public_url

                logger.info(f"Yüklendi: model_v{version}.tflite -> {tflite_blob.public_url}")

                # TFLite will be the primary model URL
                primary_model_url = tflite_blob.public_url
                model_format = 'tflite'
            else:
                logger.warning(f"TFLite model oluşturulamadı veya bulunamadı")
                primary_model_url = None
                model_format = 'checkpoint'
        except Exception as e:
            logger.error(f"TFLite model oluşturma ve yükleme hatası: {str(e)}")
            primary_model_url = None
            model_format = 'checkpoint'

        # Toplam örnek sayısını hesapla - source_model_versions kullanarak
        # Her model için toplam örnek sayısını topla
        total_scanned_examples = 0
        for source_version in self.source_model_versions:
            try:
                # Her model için metadata'yı al
                model_docs = self.db.collection('models').where('version', '==', source_version).limit(1).get()
                for doc in model_docs:
                    model_data = doc.to_dict()
                    total_scanned_examples += model_data.get('totalScannedExamples', 0)
            except Exception as e:
                logger.warning(f"Örnek sayısı alınamadı - model v{source_version}: {e}")

        # Update Firestore metadata
        metadata = {
            'version': version,
            'uploadDate': firestore.firestore.SERVER_TIMESTAMP,
            'modelUrl': primary_model_url,
            'allUrls': urls,
            'type': 'aggregated',
            'sourceModels': self.source_model_versions,
            'aggregationMethod': 'FedAvg',
            'aggregationTimestamp': datetime.now().isoformat(),
            'modelFormat': model_format,
            'totalScannedExamples': total_scanned_examples,
            'modelArchitecture': {
                'embedding_dim': 64,
                'filters': 192,
                'kernel_size': 5,
                'dense_units': 64,
                'vocab_size': 415155
            },
            'learning_rate': 0.0001
        }

        # Add to Firestore
        doc_ref = self.db.collection('models').document(f"{version}")
        doc_ref.set(metadata)

        logger.info(f"Birleştirilmiş model v{version} Firebase'e yüklendi")
        return doc_ref.id

    def run_federated_aggregation(self, min_version=None, max_models=10, cleanup=False):
        """
        Run the full federated aggregation process with only .ckpt files
        """
        try:
            # Step 1: Get model metadata
            metadata_list = self.get_model_metadata(min_version, max_models)
            if not metadata_list:
                logger.warning("Birleştirme için model bulunamadı")
                return None, None

            self.source_model_versions = [meta.get('version') for meta in metadata_list]
            logger.info(f"Bulunan model sürümleri: {self.source_model_versions}")

            # Step 2: Download models
            model_paths = self.download_models(metadata_list)
            if not model_paths:
                logger.warning("Birleştirme için model indirilemedi")
                return None, None

            # Step 3: Load model weights
            model_weights = self.load_model_weights(model_paths)
            logger.info(f"Toplam {len(model_paths)} modelden {len(model_weights)} tanesi için ağırlık yüklendi")

            # Yeterli model var mı kontrol et
            if len(model_weights) < 2:
                logger.warning(f"Birleştirme için en az 2 model gerekli, ancak {len(model_weights)} bulundu")

                # Hiç model yoksa işlemi bitir
                if len(model_weights) == 0:
                    logger.error("Birleştirmek için model ağırlığı bulunamadı. İşlem sonlandırılıyor.")
                    return None, None

                # Sadece bir model varsa, işlemi bitir (opsiyonel)
                if len(model_weights) == 1:
                    logger.warning("Sadece bir model var, birleştirme yapılamıyor. İşlem sonlandırılıyor.")
                    return None, None

            # Normal federated averaging process
            training_weights = {}
            for meta in metadata_list:
                version = meta.get('version')
                if version in model_weights:
                    examples = meta.get('totalScannedExamples', 0)
                    training_weights[version] = max(1, examples)
                    logger.info(f"Sürüm {version}: {examples} örnek, ağırlık={training_weights[version]}")

            # Perform federated averaging
            aggregated_weights = self.federated_averaging(model_weights, training_weights)

            # Base model path for saving
            base_model_path = list(model_paths.values())[0]['path'] if model_paths else None

            # Save the aggregated model
            save_path, new_version = self.save_aggregated_model(aggregated_weights, base_model_path)

            # Upload the aggregated model
            self.upload_aggregated_model(save_path, new_version, aggregated_weights)

            logger.info(f"Birleştirilmiş model v{new_version} başarıyla oluşturuldu")
            return new_version, save_path

        except Exception as e:
            logger.error(f"Federated birleştirme hatası: {e}", exc_info=True)
            return None, None
        finally:
            if cleanup:
                self.cleanup_temp_files()

    def prepare_tflite_model(self, aggregated_weights, save_dir):
        """
        Improved TFLite model creation that ensures compatibility with Android
        """
        try:
            logger.info(f"Creating TFLite model from aggregated weights")

            # 1. Ensure we have all required layers
            required_weights = ['embedding/embeddings:0', 'conv1d/kernel:0', 'conv1d/bias:0',
                                'dense/kernel:0', 'dense/bias:0', 'dense_1/kernel:0', 'dense_1/bias:0']

            for weight_name in required_weights:
                if weight_name not in aggregated_weights:
                    logger.warning(f"Missing required weight: {weight_name}")
                    # Initialize with zeros if missing
                    if 'kernel' in weight_name or 'embeddings' in weight_name:
                        # For kernels, infer shape from other layers
                        if 'embedding' in weight_name:
                            aggregated_weights[weight_name] = np.zeros((415155, 64), dtype=np.float32)
                        elif 'conv1d/kernel' in weight_name:
                            aggregated_weights[weight_name] = np.zeros((5, 64, 128), dtype=np.float32)
                        elif 'dense/kernel' in weight_name:
                            aggregated_weights[weight_name] = np.zeros((256, 64), dtype=np.float32)
                        elif 'dense_1/kernel' in weight_name:
                            aggregated_weights[weight_name] = np.zeros((64, 1), dtype=np.float32)
                    elif 'bias' in weight_name:
                        # For biases, use appropriate sizes
                        if 'conv1d/bias' in weight_name:
                            aggregated_weights[weight_name] = np.zeros((128,), dtype=np.float32)
                        elif 'dense/bias' in weight_name:
                            aggregated_weights[weight_name] = np.zeros((64,), dtype=np.float32)
                        elif 'dense_1/bias' in weight_name:
                            aggregated_weights[weight_name] = np.zeros((1,), dtype=np.float32)

            # 2. Build Keras model with the exact structure expected on Android
            embedding_layer = tf.keras.layers.Embedding(
                input_dim=aggregated_weights['embedding/embeddings:0'].shape[0],
                output_dim=aggregated_weights['embedding/embeddings:0'].shape[1],
                input_length=500,
                name='embedding'
            )

            conv_layer = tf.keras.layers.Conv1D(
                filters=aggregated_weights['conv1d/kernel:0'].shape[2],
                kernel_size=aggregated_weights['conv1d/kernel:0'].shape[0],
                activation='relu',
                name='conv1d'
            )

            pooling_layer = tf.keras.layers.MaxPooling1D(pool_size=2)
            global_pooling = tf.keras.layers.GlobalMaxPooling1D()

            dense_layer = tf.keras.layers.Dense(
                units=aggregated_weights['dense/kernel:0'].shape[1],
                activation='elu',
                name='dense'
            )

            dropout_layer = tf.keras.layers.Dropout(0.2)

            output_layer = tf.keras.layers.Dense(1, activation=None)

            # Create sequential model
            model = tf.keras.Sequential([
                embedding_layer,
                conv_layer,
                pooling_layer,
                global_pooling,
                dense_layer,
                dropout_layer,
                output_layer
            ])

            # Set weights directly
            # Embedding layer
            embedding_layer.set_weights([aggregated_weights['embedding/embeddings:0']])

            # Conv1D layer
            conv_layer.set_weights([
                aggregated_weights['conv1d/kernel:0'],
                aggregated_weights['conv1d/bias:0']
            ])

            # Dense layer
            dense_layer.set_weights([
                aggregated_weights['dense/kernel:0'],
                aggregated_weights['dense/bias:0']
            ])

            # Output layer
            output_layer.set_weights([
                aggregated_weights['dense_1/kernel:0'],
                aggregated_weights['dense_1/bias:0']
            ])

            # Create the on-device training model
            def train_step(inputs, labels):
                with tf.GradientTape() as tape:
                    predictions = model(inputs, training=True)
                    loss = tf.keras.losses.BinaryCrossentropy()(labels, predictions)

                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                return loss, predictions

            # Define the signature functions for the TFLite model
            @tf.function(input_signature=[
                tf.TensorSpec(shape=[1, 500], dtype=tf.int32, name="train_input"),
                tf.TensorSpec(shape=[1], dtype=tf.float32, name="train_label")
            ])
            def train(inputs, labels):
                loss, predictions = train_step(inputs, labels)
                return {
                    'train_loss': loss,
                    'train_prediction': predictions
                }

            @tf.function(input_signature=[
                tf.TensorSpec(shape=[1, 500], dtype=tf.int32, name="infer_input")
            ])
            def infer(inputs):
                predictions = model(inputs, training=False)
                return {
                    'infer_output': predictions
                }

            # Create optimizer
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

            # Build concrete functions
            train_concrete = train.get_concrete_function()
            infer_concrete = infer.get_concrete_function()

            # Save model to SavedModel format first
            saved_model_dir = os.path.join(save_dir, "saved_model")

            signatures = {
                "train": train_concrete,
                "infer": infer_concrete,
            }

            tf.saved_model.save(
                model,
                saved_model_dir,
                signatures=signatures
            )

            # Convert to TFLite with training enabled
            converter = tf.lite.TFLiteConverter.from_saved_model(
                saved_model_dir,
                signature_keys=["train", "infer"]
            )

            # Enable TF ops and enable training
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,
                tf.lite.OpsSet.SELECT_TF_OPS
            ]
            converter.experimental_enable_resource_variables = True
            converter.allow_custom_ops = True

            # Convert model
            tflite_model = converter.convert()

            # Save TFLite model
            tflite_path = os.path.join(save_dir, "federated_model.tflite")
            with open(tflite_path, 'wb') as f:
                f.write(tflite_model)

            logger.info(f"Created TFLite model with training enabled at {tflite_path}")
            return tflite_path

        except Exception as e:
            logger.error(f"Error creating TFLite model: {e}", exc_info=True)
            return None

    def prepare_model_for_android(self, model_dir, output_dir=None):
        """
        Birleştirilmiş modeli Android için hazırla
        """
        logger.info(f"Preparing model from directory: {model_dir}")

        # Eğer output_dir belirtilmemişse, model_dir içinde oluştur
        if output_dir is None:
            output_dir = os.path.join(model_dir, "android_output")

        # Output dir'i oluştur
        os.makedirs(output_dir, exist_ok=True)

        try:
            # Birleştirilmiş ağırlıkları doğrudan kullan
            # İlk olarak ckpt dosyasını bul
            ckpt_files = glob.glob(os.path.join(model_dir, "*.ckpt"))
            if not ckpt_files:
                raise ValueError(f"Dizinde .ckpt dosyası bulunamadı: {model_dir}")

            # İlk ckpt dosyasını kullan
            ckpt_file = ckpt_files[0]
            logger.info(f"Birleştirilmiş model dosyası: {ckpt_file}")

            # ckpt dosyasından ağırlıkları çıkar - önceki _extract_weights_from_raw_checkpoint fonksiyonunu kullan
            weights = self._extract_weights_from_raw_checkpoint(ckpt_file)

            if not weights or len(weights) == 0:
                raise ValueError("Ağırlıklar yüklenemedi!")

            # Ağırlıkların yüklendiğini doğrula
            logger.info(f"Ağırlıklar başarıyla yüklendi ({len(weights)} değişken)")

            # Model mimarisini ağırlıklardan oluştur
            # Embedding boyutunu belirle
            vocab_size = weights["embedding/embeddings:0"].shape[0]
            embedding_dim = weights["embedding/embeddings:0"].shape[1]

            # Diğer model parametrelerini belirle
            filters = weights["conv1d/bias:0"].shape[0]
            kernel_size = weights["conv1d/kernel:0"].shape[0]
            dense_units = weights["dense/bias:0"].shape[0]

            logger.info(
                f"Model mimarisi: vocab={vocab_size}, embed_dim={embedding_dim}, filters={filters}, dense={dense_units}")

            # Keras modelini oluştur
            keras_model = tf.keras.Sequential([
                tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=500),
                tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, activation='relu'),
                tf.keras.layers.MaxPooling1D(pool_size=2),
                tf.keras.layers.GlobalMaxPooling1D(),
                tf.keras.layers.Dense(units=dense_units, activation='elu'),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(1, activation=None)
            ])

            # Modeli build et
            dummy_input = tf.zeros((1, 500), dtype=tf.int32)
            keras_model(dummy_input)

            # Ağırlıkları direkt olarak set et
            # Embedding layer
            keras_model.layers[0].set_weights([weights["embedding/embeddings:0"]])

            # Conv1D layer
            keras_model.layers[1].set_weights([
                weights["conv1d/kernel:0"],
                weights["conv1d/bias:0"]
            ])

            # Dense layer
            keras_model.layers[4].set_weights([
                weights["dense/kernel:0"],
                weights["dense/bias:0"]
            ])

            # Output layer
            keras_model.layers[6].set_weights([
                weights["dense_1/kernel:0"],
                weights["dense_1/bias:0"]
            ])

            # Import FLModel from simule module
            from model_egitim import FLModel

            # Create FLModel with our Keras model
            fl_model = FLModel(keras_model, learning_rate=0.0001)

            # Save to SavedModel format with signatures
            signatures = {
                "train": fl_model.train,
                "infer": fl_model.infer,
                "save": fl_model.save
            }

            saved_model_dir = os.path.join(output_dir, "saved_model")
            tf.saved_model.save(fl_model, saved_model_dir, signatures=signatures)

            # Convert to TFLite with signatures
            converter = tf.lite.TFLiteConverter.from_saved_model(
                saved_model_dir,
                signature_keys=["train", "infer", "save"]
            )

            # Enable TF ops for training
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,
                tf.lite.OpsSet.SELECT_TF_OPS
            ]
            converter.experimental_enable_resource_variables = True
            converter.allow_custom_ops = True

            # Convert to TFLite
            tflite_model = converter.convert()

            # Save TFLite model
            tflite_path = os.path.join(output_dir, "federated_model.tflite")
            with open(tflite_path, 'wb') as f:
                f.write(tflite_model)

            logger.info(f"Prepared model for Android at {tflite_path}")

            # Add metadata JSON file
            metadata = {
                "model_version": os.path.basename(model_dir).replace("aggregated_model_v", ""),
                "creation_date": datetime.now().isoformat(),
                "model_architecture": {
                    "vocab_size": int(vocab_size),
                    "embedding_dim": int(embedding_dim),
                    "filters": int(filters),
                    "kernel_size": int(kernel_size),
                    "dense_units": int(dense_units)
                },
                "source_models": self.source_model_versions,
                "has_signatures": True  # Add flag to indicate this model has signatures
            }

            with open(os.path.join(output_dir, "model_metadata.json"), 'w') as f:
                json.dump(metadata, f, indent=2)

            return tflite_path

        except Exception as e:
            logger.error(f"Error preparing model for Android: {e}")
            raise

    def _apply_weights_to_model(self, model, aggregated_weights):
        """
        Helper to apply weights to the model

        Args:
            model: Keras model to update
            aggregated_weights: Dictionary of weights to apply
        """
        # Define a mapping from weight names to layers
        layer_mapping = {
            'embedding': model.layers[0],
            'conv1d': model.layers[1],
            'dense': model.layers[4],
            'dense_1': model.layers[6]
        }

        # Apply weights
        for var_name, weight_value in aggregated_weights.items():
            # Strip :0 suffix if present
            clean_name = var_name.split(':')[0] if ':' in var_name else var_name

            # Find which layer this weight belongs to
            for prefix, layer in layer_mapping.items():
                if clean_name.startswith(prefix):
                    # Determine which weights to set (kernel or bias)
                    if "kernel" in clean_name or "embeddings" in clean_name:
                        idx = 0  # Weight index for kernel/embeddings
                    elif "bias" in clean_name:
                        idx = 1  # Weight index for bias
                    else:
                        continue  # Skip other variables

                    # Check if shapes match
                    if layer.weights[idx].shape == weight_value.shape:
                        layer.weights[idx].assign(weight_value)
                        logger.info(f"Applied weight {clean_name} to {layer.name}")
                    else:
                        logger.warning(
                            f"Shape mismatch for {clean_name}: expected {layer.weights[idx].shape}, got {weight_value.shape}")
                    break

    def cleanup_temp_files(self):
        """
        Safely clean up temporary files
        """
        if not os.path.exists(self.temp_dir):
            return

        try:
            # First list all files
            for root, dirs, files in os.walk(self.temp_dir, topdown=False):
                # Remove files first
                for name in files:
                    try:
                        os.remove(os.path.join(root, name))
                    except (OSError, PermissionError) as e:
                        logger.warning(f"Could not remove file {name}: {e}")

                # Then try to remove directories
                for name in dirs:
                    try:
                        dir_path = os.path.join(root, name)
                        if os.path.exists(dir_path):
                            os.rmdir(dir_path)
                    except (OSError, PermissionError) as e:
                        logger.warning(f"Could not remove directory {name}: {e}")

            # Finally try to remove the temp directory itself
            try:
                if os.path.exists(self.temp_dir):
                    os.rmdir(self.temp_dir)
            except (OSError, PermissionError) as e:
                logger.warning(f"Could not remove temp directory: {e}")

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


# Example usage
if __name__ == "__main__":
    # Initialize server variable to None
    server = None

    try:
        # Path to your Firebase service account key
        service_account_path = "firebase.json"

        # Your actual Firebase Storage bucket name
        bucket_name = "havadismedya-fce44.appspot.com"

        # Initialize the server with proper bucket name
        server = FederatedLearningServer(service_account_path, bucket_name)

        # Run federated aggregation without automatic cleanup
        result = server.run_federated_aggregation(max_models=10, cleanup=True)

        # Check if the result is a tuple as expected
        if result and isinstance(result, tuple) and len(result) == 2:
            new_version, model_dir = result

            print(f"Successfully created new aggregated model version {new_version}")
        else:
            print("Failed to create aggregated model.")

    except Exception as e:
        print(f"Error in federated learning process: {e}")
    finally:
        # Clean up temporary files after everything is done
        print("Cleaning up temporary files...")
        try:
            # Now we can safely check if 'server' exists and has 'temp_dir'
            if server is not None and hasattr(server, 'temp_dir'):
                server.cleanup_temp_files()
        except Exception as e:
            print(f"Error during cleanup: {e}")