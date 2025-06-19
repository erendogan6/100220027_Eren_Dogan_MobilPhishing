package acr.browser.lightning.phishing

import android.annotation.SuppressLint
import android.content.Context
import android.util.JsonReader
import android.util.Log
import androidx.core.content.edit
import kotlinx.coroutines.*
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.File
import java.io.FileInputStream
import java.io.InputStreamReader
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import javax.inject.Inject
import javax.inject.Singleton
import kotlin.random.Random

@Singleton
class PhishingDetector @Inject constructor(
    private val context: Context,
    private val modelManager: ModelManager,
    private val scanCounter: ScanCounter
) {
    private var interpreter: Interpreter? = null
    private var tokenizer: MutableMap<String, Int>? = null
    private val maxSequenceLength = 500
    private var checkpointCounter = 0
    private val CHECKPOINT_INTERVAL = 5

    // Model version tracking
    private var modelVersion = 1.0F

    // Memory and threading safety
    private val isTrainingInProgress = AtomicBoolean(false)
    private val trainingQueue = AtomicInteger(0)
    private val maxConcurrentOperations = 2

    // Coroutine scope for safe operations
    private val detectorScope = CoroutineScope(
        Dispatchers.Default + SupervisorJob() +
                CoroutineExceptionHandler { _, throwable ->
                    Log.e(TAG, "Unhandled exception in PhishingDetector", throwable)
                }
    )

    private var lastTrainingTime = 0L
    private val trainingCooldownMs = 1000L

    private var bufferPool = mutableListOf<ByteBuffer>()

    private companion object {
        const val TAG = "PhishingDetector"
        const val INFER_KEY = "infer"
        const val TRAIN_KEY = "train"
        const val SAVE_KEY = "save"
        const val INFER_INPUT_NAME = "infer_input"
        const val INFER_OUTPUT_NAME = "infer_output"
        const val TRAIN_INPUT_NAME = "train_input"
        const val TRAIN_LABEL_NAME = "train_label"
        const val TRAIN_LOSS_NAME = "train_loss"
        const val TRAIN_PREDICTION_NAME = "train_prediction"
    }

    fun initModel() {
        try {
            val modelPath = modelManager.getCurrentModelPath()

            if (modelPath.isNotEmpty() && File(modelPath).exists()) {
                loadTFLiteModel(modelPath)
            } else {
                loadModelFromAssets()
            }

            loadTokenizerInChunks()
            Log.d(TAG, "Model initialized successfully, version: $modelVersion")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize model", e)
            loadModelFromAssets()
        }
    }

    fun getModelVersion(): Float = modelVersion

    fun loadModelFromFile(modelPath: String): Boolean {
        return try {
            val modelFile = File(modelPath)
            if (!modelFile.exists()) {
                Log.e(TAG, "Model dosyası bulunamadı: $modelPath")
                return false
            }

            interpreter?.close()
            clearBufferPool() // Buffer'ları temizle

            val options = Interpreter.Options().apply {
                setUseNNAPI(false)
                setNumThreads(2) // Thread sayısını sınırla
            }

            val modelBuffer = FileInputStream(modelFile).channel.map(
                FileChannel.MapMode.READ_ONLY,
                0,
                modelFile.length()
            )

            interpreter = Interpreter(modelBuffer, options)
            modelVersion = modelManager.getCurrentModelVersion()

            Log.d(TAG, "Yeni model yüklendi: v$modelVersion")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Model yükleme hatası", e)
            initModel()
            false
        }
    }

    /**
     * SAFE analyze for phishing - Fixed version
     */
    fun analyzeForPhishing(htmlContent: String?): Pair<Boolean, Float> {
        return try {

            // Queue check - prevent overwhelming
            if (trainingQueue.get() > maxConcurrentOperations) {
                Log.w(TAG, "Too many operations in queue, skipping analysis")
                return false to 0.0f
            }

            val processedInput = preprocessHtml(htmlContent)
            val predictionScore = predict(processedInput)

            println("Prediction Score: $predictionScore")
            scanCounter.incrementScanCount()

            // SAFE training - only if conditions are met
            val isPhishing = predictionScore > 0.6f
            performSafeAutomaticTraining(processedInput, predictionScore, isPhishing)

            isPhishing to predictionScore
        } catch (e: OutOfMemoryError) {
            Log.e(TAG, "OutOfMemoryError in analysis", e)
            emergencyCleanup()
            false to 0.0f
        } catch (e: Exception) {
            Log.e(TAG, "Phishing tespiti sırasında hata oluştu", e)
            false to 0.0f
        }
    }

    /**
     * SAFE automatic training - Fixed version
     */
    private fun performSafeAutomaticTraining(
        input: IntArray,
        predictionScore: Float,
        isPhishing: Boolean
    ) {
        // Training throttling
        val currentTime = System.currentTimeMillis()
        if (currentTime - lastTrainingTime < trainingCooldownMs) {
            Log.d(TAG, "Training skipped due to cooldown")
            return
        }

        // Check if training is already in progress
        if (!isTrainingInProgress.compareAndSet(false, true)) {
            Log.d(TAG, "Training already in progress, skipping")
            return
        }

        // Launch training in background with proper error handling
        detectorScope.launch {
            try {
                trainingQueue.incrementAndGet()

                // Main thread'de training yapma - Dispatcher.Default kullan
                val label = if (isPhishing) 1f else 0f
                val (loss, trainingPrediction) = trainOnDeviceSafe(input, label)

                lastTrainingTime = System.currentTimeMillis()

                Log.d(TAG, """
                    Safe Automatic Training Completed:
                    - Original Prediction: $predictionScore
                    - Is Phishing: $isPhishing
                    - Training Label: $label
                    - Training Loss: $loss
                    - Training Prediction: $trainingPrediction
                    - Model Version: $modelVersion
                    - Total Scans: ${scanCounter.getTotalScanCount()}
                """.trimIndent())

            } catch (e: OutOfMemoryError) {
                Log.e(TAG, "OOM during training", e)
                emergencyCleanup()
            } catch (e: Exception) {
                Log.e(TAG, "Safe automatic training failed", e)
            } finally {
                trainingQueue.decrementAndGet()
                isTrainingInProgress.set(false)
            }
        }
    }

    private fun loadTFLiteModel(modelFilePath: String): Boolean {
        try {
            val options = Interpreter.Options().apply {
                setUseNNAPI(false)
                setNumThreads(2) // Limit threads
            }

            val tfliteFile = File(modelFilePath)

            if (!tfliteFile.exists() || tfliteFile.length() == 0L) {
                Log.e(TAG, "TFLite model not found or empty at $modelFilePath")
                return false
            }

            try {
                interpreter?.close()
                interpreter = Interpreter(tfliteFile, options)
                Log.d(TAG, "Loaded TFLite model from $modelFilePath (${tfliteFile.length()} bytes)")

                interpreter?.signatureKeys?.forEach { key ->
                    Log.d(TAG, "Signature key: $key")
                }

                modelVersion = modelManager.getCurrentModelVersion()
                return true
            } catch (e: Exception) {
                Log.e(TAG, "Failed to initialize TFLite interpreter", e)
                return false
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load TFLite model", e)
            return false
        }
    }

    @SuppressLint("LogConditional")
    private fun loadModelFromAssets() {
        try {
            val options = Interpreter.Options().apply {
                setUseNNAPI(false)
                setNumThreads(2)
            }

            context.assets.openFd("federated_model.tflite").use { fd ->
                interpreter = Interpreter(
                    FileInputStream(fd.fileDescriptor).channel.map(
                        FileChannel.MapMode.READ_ONLY,
                        fd.startOffset,
                        fd.declaredLength
                    ),
                    options
                )
            }

            modelVersion = 1.0f
            Log.d(TAG, "Model loaded from assets successfully")
        } catch (e: Exception) {
            Log.e(TAG, "Model loading from assets failed", e)
            throw RuntimeException("Model loading failed: ${e.message}")
        }
    }

    @SuppressLint("LogConditional")
    private fun loadTokenizerInChunks() {
        tokenizer = mutableMapOf()
        try {
            context.assets.open("tokenizer_config.json").use { inputStream ->
                val reader = JsonReader(InputStreamReader(inputStream, "UTF-8"))

                reader.beginObject()
                while (reader.hasNext()) {
                    val name = reader.nextName()
                    if (name == "word_index") {
                        reader.beginObject()

                        var count = 0
                        while (reader.hasNext()) {
                            val word = reader.nextName()
                            val index = reader.nextInt()
                            tokenizer!![word] = index
                            count++
                        }
                        reader.endObject()
                    } else {
                        reader.skipValue()
                    }
                }
                reader.endObject()
            }
            Log.d(TAG, "Tokenizer loaded: ${tokenizer?.size} tokens")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load tokenizer", e)
            tokenizer = mutableMapOf()
        }
    }

    @SuppressLint("LogConditional")
    fun saveCheckpoint(): File {
        try {
            val checkpointDir = File(context.filesDir, "checkpoints")
            if (!checkpointDir.exists()) checkpointDir.mkdirs()

            val checkpointFile = File(checkpointDir, "model_checkpoint.ckpt")
            val pathString = checkpointFile.absolutePath

            Log.d(TAG, "Saving checkpoint to: $pathString")

            val outputs = mutableMapOf<String, Any>()
            interpreter?.runSignature(
                mapOf("checkpoint_path" to pathString),
                outputs,
                SAVE_KEY
            )

            val createdFiles = checkpointDir.listFiles()?.joinToString { it.name } ?: "none"
            Log.d(TAG, "Files in checkpoint directory: $createdFiles")

            val version = when {
                outputs["version"] is FloatArray -> (outputs["version"] as FloatArray)[0]
                outputs["version"] is Float -> outputs["version"] as Float
                else -> modelVersion + 0.1f
            }
            modelVersion = version
            context.getSharedPreferences("PhishingDetectorPrefs", Context.MODE_PRIVATE).edit {
                putFloat("modelVersion", modelVersion)
            }

            Log.d(TAG, "Checkpoint saved successfully: $pathString, Version: $modelVersion")
            return checkpointFile

        } catch (e: Exception) {
            Log.e(TAG, "Failed to save checkpoint", e)
            throw e
        }
    }

    private fun shouldSaveCheckpoint(): Boolean {
        checkpointCounter++
        if (checkpointCounter >= CHECKPOINT_INTERVAL) {
            checkpointCounter = 0
            return true
        }
        return false
    }

    /**
     * SAFE training with proper buffer management
     */
    @SuppressLint("LogConditional")
    suspend fun trainOnDeviceSafe(input: IntArray, label: Float): Pair<Float, Float> = withContext(Dispatchers.Default) {
        var inputBuffer: ByteBuffer? = null
        var labelBuffer: TensorBuffer? = null
        var lossBuffer: TensorBuffer? = null
        var predictionBuffer: TensorBuffer? = null

        try {

            // 1. Create buffers with proper management
            inputBuffer = getOrCreateBuffer(4 * 500).apply {
                clear() // Reset position
                input.forEach { putInt(it) }
                rewind()
            }

            labelBuffer = TensorBuffer.createFixedSize(intArrayOf(1), DataType.FLOAT32).apply {
                loadArray(floatArrayOf(label))
            }

            // 2. Create output buffers
            lossBuffer = TensorBuffer.createFixedSize(intArrayOf(1), DataType.FLOAT32)
            predictionBuffer = TensorBuffer.createFixedSize(intArrayOf(1), DataType.FLOAT32)

            // 3. Run training with timeout protection
            val trainingJob = async {
                interpreter?.runSignature(
                    mapOf(
                        TRAIN_INPUT_NAME to inputBuffer,
                        TRAIN_LABEL_NAME to labelBuffer.buffer
                    ),
                    mapOf(
                        TRAIN_LOSS_NAME to lossBuffer.buffer,
                        TRAIN_PREDICTION_NAME to predictionBuffer.buffer
                    ),
                    TRAIN_KEY
                )
            }

            // Timeout after 10 seconds
            withTimeout(10000L) {
                trainingJob.await()
            }

            // 4. Process results
            val loss = lossBuffer.floatArray[0]
            val prediction = predictionBuffer.floatArray[0]

            Log.d(TAG, """
                Safe Training Results:
                - Loss: $loss
                - Prediction: $prediction
                - Label: $label
            """.trimIndent())

            if (shouldSaveCheckpoint()) {
                saveCheckpoint()
            }

            return@withContext Pair(loss, prediction)

        } catch (e: TimeoutCancellationException) {
            Log.e(TAG, "Training timeout", e)
            throw e
        } catch (e: OutOfMemoryError) {
            Log.e(TAG, "OOM during training", e)
            emergencyCleanup()
            throw e
        } catch (e: Exception) {
            Log.e(TAG, "Safe training error", e)
            throw e
        } finally {
            // Always return buffer to pool
            inputBuffer?.let { returnBufferToPool(it) }
        }
    }

    // Buffer pool management
    private fun getOrCreateBuffer(size: Int): ByteBuffer {
        synchronized(bufferPool) {
            val existingBuffer = bufferPool.find { it.capacity() >= size }
            if (existingBuffer != null) {
                bufferPool.remove(existingBuffer)
                return existingBuffer.order(ByteOrder.nativeOrder())
            }
        }
        return ByteBuffer.allocateDirect(size).order(ByteOrder.nativeOrder())
    }

    private fun returnBufferToPool(buffer: ByteBuffer) {
        synchronized(bufferPool) {
            if (bufferPool.size < 5) { // Limit pool size
                buffer.clear()
                bufferPool.add(buffer)
            }
        }
    }

    private fun clearBufferPool() {
        synchronized(bufferPool) {
            bufferPool.clear()
        }
    }

    private fun emergencyCleanup() {
        try {
            Log.w(TAG, "Performing emergency cleanup")

            // Clear buffer pool
            clearBufferPool()

            // Force garbage collection
            System.gc()
            System.runFinalization()
            System.gc()

            // Reset training state
            isTrainingInProgress.set(false)
            trainingQueue.set(0)

            Log.i(TAG, "Emergency cleanup completed")
        } catch (e: Exception) {
            Log.e(TAG, "Emergency cleanup failed", e)
        }
    }

    fun preprocessHtml(htmlContent: String?): IntArray {
        Log.d(TAG, "Original input: ${htmlContent?.take(100)}...")

        val text = htmlContent ?: ""
        val lowercaseText = text.lowercase()

        var filteredText = lowercaseText
        val specialChars = "!\"#$%&()*+,-./:;<=>?@[]\\^_`{|}~\t\n"
        for (char in specialChars) {
            filteredText = filteredText.replace(char.toString(), " ")
        }

        val cleanText = filteredText.split("\\s+".toRegex())
            .filter { it.isNotEmpty() }
            .joinToString(" ")

        Log.d(TAG, "Cleaned text: ${cleanText.take(100)}...")

        val tokens = tokenizeText(cleanText)
        Log.d(TAG, "Tokenization result - count: ${tokens.size}, first 20: ${tokens.take(20)}")

        return padSequence(tokens)
    }

    private fun tokenizeText(text: String): List<Int> {
        return text.split(" ").map { word ->
            val tokenId = tokenizer?.get(word)
            val oovTokenId = tokenizer?.get("<OOV>") ?: 1

            val finalTokenId = when {
                tokenId != null -> tokenId
                else -> oovTokenId
            }

            finalTokenId
        }
    }

    private fun padSequence(tokens: List<Int>): IntArray {
        val result = IntArray(maxSequenceLength) { index ->
            if (index < tokens.size) tokens[index] else 0
        }

        Log.d(TAG, "Padded sequence: ${result.take(10)}...${result.takeLast(10)}")
        return result
    }

    @SuppressLint("LogConditional")
    fun predict(input: IntArray): Float {
        try {
            require(input.size == 500) {
                "Geçersiz girdi boyutu: ${input.size}. Tam olarak 500 eleman olmalı."
            }

            val inputArray = Array(1) { input }
            val outputs = mutableMapOf<String, Any>()

            val outputArray = Array(1) { FloatArray(1) }
            outputs[INFER_OUTPUT_NAME] = outputArray

            interpreter?.runSignature(
                mapOf(INFER_INPUT_NAME to inputArray),
                outputs,
                INFER_KEY
            ) ?: throw Exception("Interpreter null")

            val output = outputs["infer_output"] as? Array<*>
                ?: throw Exception("Geçersiz çıktı formatı")

            return (output[0] as FloatArray)[0]

        } catch (e: Exception) {
            Log.e(TAG, "Tahmin başarısız oldu", e)
            throw e
        }
    }

    fun close() {
        try {
            // Cancel all ongoing operations
            detectorScope.cancel()

            interpreter?.close()
            interpreter = null
            tokenizer?.clear()
            tokenizer = null

            clearBufferPool()
            isTrainingInProgress.set(false)
            trainingQueue.set(0)

            System.gc()
        } catch (e: Exception) {
            Log.e(TAG, "Kapatma hatası", e)
        }
    }
}