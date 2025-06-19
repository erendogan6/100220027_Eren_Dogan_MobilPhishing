package acr.browser.lightning.phishing

import android.net.Uri
import android.util.Log
import com.google.android.gms.tasks.Tasks
import com.google.firebase.firestore.FieldValue
import com.google.firebase.firestore.FirebaseFirestore
import com.google.firebase.storage.FirebaseStorage
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.util.UUID
import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class FirebaseService @Inject constructor(
    private val phishingDetector: PhishingDetector,
    private val scanCounter: ScanCounter,
    private val firestore: FirebaseFirestore,
    private val firebaseStorage: FirebaseStorage
) {
    private val TAG = "FirebaseService"
    private val modelCollection = firestore.collection("models")

    suspend fun uploadModel(modelVersion: Float): Boolean {
        return withContext(Dispatchers.IO) {
            try {
                // Checkpoint dosyasını kaydet
                val checkpointFile = phishingDetector.saveCheckpoint()
                val checkpointDir = checkpointFile.parentFile ?: return@withContext false

                val checkpointFiles = checkpointDir.listFiles { file ->
                    file.name.startsWith("model_checkpoint")
                } ?: emptyArray()

                if (checkpointFiles.isEmpty()) {
                    Log.e(TAG, "Checkpoint dosyası bulunamadı")
                    return@withContext false
                }

                val versionFormatted = String.format("%.1f", modelVersion)
                val uploadedUrls = mutableMapOf<String, String>()

                Log.d(TAG, "${checkpointFiles.size} dosya yüklenecek")

                // Dosyaları Firebase Storage'a yükle
                for (file in checkpointFiles) {
                    val fileName = "v${versionFormatted}_${file.name}"
                    val storageRef = firebaseStorage.reference
                        .child("models")
                        .child("model_v${versionFormatted}")
                        .child(fileName)

                    val uploadTask = storageRef.putFile(Uri.fromFile(file))
                    Tasks.await(uploadTask)
                    val downloadUrl = Tasks.await(storageRef.downloadUrl)

                    uploadedUrls[fileName] = downloadUrl.toString()
                    Log.d(TAG, "Yüklendi: $fileName")
                }

                if (uploadedUrls.isEmpty()) {
                    Log.e(TAG, "Hiçbir dosya yüklenemedi")
                    return@withContext false
                }

                val mainModelUrl = uploadedUrls.values.first()

                // Firestore'a metadata kaydet
                val modelData = hashMapOf(
                    "version" to versionFormatted + UUID.randomUUID().toString().take(8),
                    "uploadDate" to FieldValue.serverTimestamp(),
                    "totalScannedExamples" to scanCounter.getTotalScanCount(),
                    "type" to "device",
                    "modelFormat" to "checkpoint",
                    "modelUrl" to mainModelUrl,
                    "allUrls" to uploadedUrls,
                    "deviceInfo" to mapOf(
                        "model" to android.os.Build.MODEL,
                        "sdk" to android.os.Build.VERSION.SDK_INT
                    )
                )

                Tasks.await(modelCollection.document(versionFormatted).set(modelData))
                Log.d(TAG, "Model metadata Firestore'a kaydedildi")

                true
            } catch (e: Exception) {
                Log.e(TAG, "Model yükleme hatası", e)
                false
            }
        }
    }
}