package acr.browser.lightning.phishing

import android.content.Context
import android.util.Log
import com.google.firebase.firestore.FirebaseFirestore
import com.google.firebase.firestore.Query
import com.google.firebase.storage.FirebaseStorage
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.tasks.await
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.net.HttpURLConnection
import java.net.URL
import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class ModelManager @Inject constructor(
    private val context: Context,
    private val firestore: FirebaseFirestore,
    private val firebaseStorage: FirebaseStorage
) {
    private val prefs = context.getSharedPreferences("PhishingDetectorPrefs", Context.MODE_PRIVATE)
    private val TAG = "ModelManager"

    fun getCurrentModelVersion(): Float {
        return prefs.getFloat("currentModelVersion", 1.0f)
    }

    private fun updateCurrentModelVersion(version: Float) {
        prefs.edit().putFloat("currentModelVersion", version).apply()
    }

    fun getCurrentModelPath(): String {
        return prefs.getString("currentModelPath", "") ?: ""
    }

    private fun updateCurrentModelPath(path: String) {
        prefs.edit().putString("currentModelPath", path).apply()
    }

    suspend fun checkForNewModel(): Pair<Float, String>? {
        return withContext(Dispatchers.IO) {
            try {
                val snapshot = firestore.collection("models")
                    .whereEqualTo("type", "aggregated")
                    .orderBy("version", Query.Direction.DESCENDING)
                    .limit(1)
                    .get()
                    .await()

                if (snapshot.isEmpty) {
                    Log.d(TAG, "Firebase'de model bulunamadı")
                    return@withContext null
                }

                val modelDoc = snapshot.documents[0]
                val versionStr = modelDoc.getString("version") ?: return@withContext null
                val modelUrl = modelDoc.getString("modelUrl") ?: return@withContext null

                if (!modelUrl.endsWith(".tflite")) {
                    Log.e(TAG, "Model TFLite formatında değil: $modelUrl")
                    return@withContext null
                }

                val version = versionStr.toFloatOrNull() ?: 0.0f
                val currentVersion = getCurrentModelVersion()
                Log.d(TAG, "Mevcut: $currentVersion, Sunucu: $version")

                if (version > currentVersion) {
                    return@withContext Pair(version, modelUrl)
                }

                null
            } catch (e: Exception) {
                Log.e(TAG, "Model kontrol hatası", e)
                null
            }
        }
    }

    suspend fun downloadAndInstallNewModel(versionInfo: Pair<Float, String>): Boolean {
        return withContext(Dispatchers.IO) {
            try {
                val (version, modelUrl) = versionInfo
                val versionFormatted = "%.1f".format(version)

                val modelsDir = File(context.filesDir, "models")
                if (!modelsDir.exists()) {
                    modelsDir.mkdirs()
                }

                val modelFile = File(modelsDir, "model_v$versionFormatted.tflite")
                downloadFileFromUrl(modelUrl, modelFile)

                if (!modelFile.exists() || modelFile.length() == 0L) {
                    Log.e(TAG, "Model indirme başarısız")
                    return@withContext false
                }

                updateCurrentModelVersion(version)
                updateCurrentModelPath(modelFile.absolutePath)
                cleanupOldModels(modelFile)

                Log.d(TAG, "Model başarıyla yüklendi: v$versionFormatted")
                true
            } catch (e: Exception) {
                Log.e(TAG, "Model indirme hatası", e)
                false
            }
        }
    }

    private suspend fun downloadFileFromUrl(url: String, destinationFile: File) {
        withContext(Dispatchers.IO) {
            val connection = URL(url).openConnection() as HttpURLConnection
            connection.connectTimeout = 15000
            connection.readTimeout = 30000
            connection.connect()

            if (connection.responseCode != HttpURLConnection.HTTP_OK) {
                throw Exception("HTTP error ${connection.responseCode}")
            }

            val input = connection.inputStream
            val output = FileOutputStream(destinationFile)

            val buffer = ByteArray(4096)
            var bytesRead: Int
            while (input.read(buffer).also { bytesRead = it } != -1) {
                output.write(buffer, 0, bytesRead)
            }

            output.close()
            input.close()
            Log.d(TAG, "İndirildi: ${destinationFile.name}")
        }
    }

    private fun cleanupOldModels(currentModelFile: File) {
        try {
            val modelsDir = File(context.filesDir, "models")
            modelsDir.listFiles()?.forEach { file ->
                if (file.isFile && file.absolutePath != currentModelFile.absolutePath &&
                    file.name.matches(Regex("model_v\\d+\\.\\d+\\.tflite"))) {
                    if (file.delete()) {
                        Log.d(TAG, "Eski model silindi: ${file.name}")
                    }
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Model temizleme hatası", e)
        }
    }
}