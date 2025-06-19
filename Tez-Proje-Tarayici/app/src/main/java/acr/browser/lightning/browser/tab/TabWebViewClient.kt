package acr.browser.lightning.browser.tab

import acr.browser.lightning.R
import acr.browser.lightning.adblock.AdBlocker
import acr.browser.lightning.adblock.allowlist.AllowListModel
import acr.browser.lightning.databinding.DialogAuthRequestBinding
import acr.browser.lightning.databinding.DialogSslWarningBinding
import acr.browser.lightning.extensions.resizeAndShow
import acr.browser.lightning.js.TextReflow
import acr.browser.lightning.log.Logger
import acr.browser.lightning.phishing.FirebaseService
import acr.browser.lightning.phishing.PhishingDetector
import acr.browser.lightning.phishing.PhishingEventBus
import acr.browser.lightning.phishing.ScanCounter
import acr.browser.lightning.preference.UserPreferences
import acr.browser.lightning.ssl.SslState
import acr.browser.lightning.ssl.SslWarningPreferences
import android.annotation.SuppressLint
import android.app.Application
import android.graphics.Bitmap
import android.net.http.SslError
import android.os.Message
import android.util.Log
import android.view.LayoutInflater
import android.webkit.HttpAuthHandler
import android.webkit.SslErrorHandler
import android.webkit.URLUtil
import android.webkit.WebResourceRequest
import android.webkit.WebResourceResponse
import android.webkit.WebView
import android.webkit.WebViewClient
import androidx.appcompat.app.AlertDialog
import androidx.lifecycle.LifecycleOwner
import androidx.lifecycle.lifecycleScope
import androidx.webkit.WebViewAssetLoader.InternalStoragePathHandler
import dagger.assisted.Assisted
import dagger.assisted.AssistedFactory
import dagger.assisted.AssistedInject
import io.reactivex.rxjava3.subjects.PublishSubject
import kotlinx.coroutines.*
import java.io.ByteArrayInputStream
import java.io.File
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.coroutines.resume
import kotlin.math.abs

/**
 * A [WebViewClient] that supports the tab adaptation - CRASH FIXED VERSION
 */
class TabWebViewClient @AssistedInject constructor(
    private val application: Application,
    private val adBlocker: AdBlocker,
    private val allowListModel: AllowListModel,
    private val urlHandler: UrlHandler,
    @Assisted private val headers: Map<String, String>,
    private val userPreferences: UserPreferences,
    private val sslWarningPreferences: SslWarningPreferences,
    private val textReflow: TextReflow,
    private val logger: Logger,
    private val phishingDetector: PhishingDetector,
    private val scanCounter: ScanCounter,
    private val firebaseService: FirebaseService,
    @Assisted("cache") private val cacheStoragePathHandler: InternalStoragePathHandler,
    @Assisted("files") private val filesStoragePathHandler: InternalStoragePathHandler,
) : WebViewClient() {

    private val cache by lazy {
        File(application.cacheDir, "favicon-cache")
    }

    private val files by lazy {
        File(application.filesDir, "generated-html")
    }

    // FIXED: Proper coroutine scope with error handling
    private val webViewClientScope = CoroutineScope(
        Dispatchers.Main + SupervisorJob() +
                CoroutineExceptionHandler { _, throwable ->
                    Log.e(TAG, "Unhandled exception in TabWebViewClient", throwable)
                }
    )

    // FIXED: Phishing detection throttling
    private val phishingDetectionInProgress = AtomicBoolean(false)
    private var lastPhishingCheckTime = 0L
    private val phishingCooldownMs = 3000L // 3 seconds between checks

    /**
     * Emits changes to the current URL.
     */
    val urlObservable: PublishSubject<String> = PublishSubject.create()

    /**
     * Emits changes to the current SSL state.
     */
    val sslStateObservable: PublishSubject<SslState> = PublishSubject.create()

    /**
     * Emits changes to the can go back state of the browser.
     */
    val goBackObservable: PublishSubject<Boolean> = PublishSubject.create()

    /**
     * Emits changes to the can go forward state of the browser.
     */
    val goForwardObservable: PublishSubject<Boolean> = PublishSubject.create()

    /**
     * Emit when the tab has finished rendering its content.
     */
    val finishedObservable = PublishSubject.create<Unit>()

    /**
     * Phishing tespit edildiğinde bildiri yayınlar (URL, Güven Oranı)
     */
    val phishingDetectedObservable: PublishSubject<Pair<String, Float>> = PublishSubject.create()

    /**
     * The current SSL state of the page.
     */
    var sslState: SslState = SslState.None
        private set

    private var currentUrl: String = ""
    private var isReflowRunning: Boolean = false
    private var zoomScale: Float = 0.0F
    private var urlWithSslError: String? = null

    private fun shouldBlockRequest(pageUrl: String, requestUrl: String) =
        !allowListModel.isUrlAllowedAds(pageUrl) &&
                adBlocker.isAd(requestUrl)

    override fun onPageStarted(view: WebView, url: String, favicon: Bitmap?) {
        super.onPageStarted(view, url, favicon)
        currentUrl = url
        urlObservable.onNext(url)
        if (urlWithSslError != url) {
            urlWithSslError = null
            sslState = if (URLUtil.isHttpsUrl(url)) {
                SslState.Valid
            } else {
                SslState.None
            }
        }
        sslStateObservable.onNext(sslState)
    }

    /**
     * FIXED: Safe phishing detection with proper threading and error handling
     */
    @SuppressLint("CheckResult")
    override fun onPageFinished(view: WebView, url: String) {
        super.onPageFinished(view, url)
        urlObservable.onNext(url)
        goBackObservable.onNext(view.canGoBack())
        goForwardObservable.onNext(view.canGoForward())

        // FIXED: Safe phishing detection
        if (shouldPerformPhishingCheck(url)) {
            performSafePhishingDetection(view, url)
        }

        view.postVisualStateCallback(1, object : WebView.VisualStateCallback() {
            override fun onComplete(requestId: Long) {
                finishedObservable.onNext(Unit)
            }
        })
    }

    /**
     * FIXED: Check if phishing detection should be performed
     */
    private fun shouldPerformPhishingCheck(url: String): Boolean {
        // Skip Google searches and internal pages
        if (url.startsWith("https://www.google.com/") ||
            url.startsWith("chrome://") ||
            url.startsWith("file://") ||
            url.startsWith("data:")) {
            return false
        }

        // Throttling check
        val currentTime = System.currentTimeMillis()
        if (currentTime - lastPhishingCheckTime < phishingCooldownMs) {
            Log.d(TAG, "Phishing check skipped due to cooldown")
            return false
        }

        // Check if detection already in progress
        if (phishingDetectionInProgress.get()) {
            Log.d(TAG, "Phishing detection already in progress")
            return false
        }

        return true
    }

    /**
     * FIXED: Safe phishing detection with proper error handling and threading
     */
    private fun performSafePhishingDetection(view: WebView, url: String) {
        if (!phishingDetectionInProgress.compareAndSet(false, true)) {
            return
        }

        webViewClientScope.launch {
            try {
                // Small delay to ensure page is fully loaded
                delay(100)

                // Get HTML content safely
                val htmlContent = getHtmlContentSafely(view)

                if (htmlContent.isNullOrBlank()) {
                    Log.d(TAG, "No HTML content available for phishing detection")
                    return@launch
                }

                // FIXED: Run phishing detection in background thread
                val result = withContext(Dispatchers.Default) {
                    try {
                        phishingDetector.analyzeForPhishing(htmlContent)
                    } catch (e: OutOfMemoryError) {
                        Log.e(TAG, "OOM in phishing detection", e)
                        false to 0.0f
                    } catch (e: Exception) {
                        Log.e(TAG, "Phishing detection error", e)
                        false to 0.0f
                    }
                }

                // FIXED: Handle results on main thread
                withContext(Dispatchers.Main) {
                    handlePhishingResult(url, result)
                }

                lastPhishingCheckTime = System.currentTimeMillis()

            } catch (e: CancellationException) {
                Log.d(TAG, "Phishing detection cancelled")
            } catch (e: Exception) {
                Log.e(TAG, "Error in safe phishing detection", e)
            } finally {
                phishingDetectionInProgress.set(false)
            }
        }
    }

    /**
     * FIXED: Safe HTML content extraction with timeout
     */
    private suspend fun getHtmlContentSafely(view: WebView): String? =
        withContext(Dispatchers.Main) {
            suspendCancellableCoroutine<String?> { continuation ->
                try {
                    view.evaluateJavascript("(function() { return document.documentElement.outerHTML; })();") { htmlContent ->
                        try {
                            if (continuation.isActive) {
                                val cleanHtml = htmlContent?.replace("^\"|\"$".toRegex(), "")
                                    ?.replace("\\\"", "\"")
                                    ?.replace("\\n", "\n")
                                    ?.replace("\\\\", "\\")
                                continuation.resume(cleanHtml)
                            }
                        } catch (e: Exception) {
                            if (continuation.isActive) {
                                continuation.resume(null)
                            }
                        }
                    }

                    // Timeout after 5 seconds
                    webViewClientScope.launch {
                        delay(5000)
                        if (continuation.isActive) {
                            continuation.resume(null)
                        }
                    }

                } catch (e: Exception) {
                    if (continuation.isActive) {
                        continuation.resume(null)
                    }
                }
            }
        }

    /**
     * FIXED: Handle phishing detection results safely
     */
    private fun handlePhishingResult(url: String, result: Pair<Boolean, Float>) {
        try {
            val (isPhishing, confidence) = result

            if (isPhishing) {
                Log.i(TAG, "Phishing detected: $url with confidence: $confidence")
                PhishingEventBus.reportPhishing(url, confidence)
                phishingDetectedObservable.onNext(url to confidence)
            }

            // Check if model upload is needed
            if (scanCounter.shouldUploadModel()) {
                Log.d(TAG, "Model upload gerekli, tarama sayısı: ${scanCounter.getScansSinceLastUpload()}")
                handleModelUploadSafely()
            }

        } catch (e: Exception) {
            Log.e(TAG, "Error handling phishing result", e)
        }
    }

    /**
     * FIXED: Safe model upload with proper error handling
     */
    private fun handleModelUploadSafely() {
        webViewClientScope.launch {
            try {
                val modelVersion = phishingDetector.getModelVersion()

                // Background thread'de model upload işlemini yap
                val success = withContext(Dispatchers.IO) {
                    try {
                        firebaseService.uploadModel(modelVersion)
                    } catch (e: Exception) {
                        Log.e(TAG, "Model upload failed", e)
                        false
                    }
                }

                // Main thread'de sonucu işle
                if (success) {
                    scanCounter.resetUploadCounter()
                    Log.d(TAG, "Model başarıyla yüklendi - version: $modelVersion")
                } else {
                    Log.w(TAG, "Model yüklenemedi")
                }
            } catch (e: Exception) {
                Log.e(TAG, "Model upload hatası", e)
            }
        }
    }

    override fun onScaleChanged(view: WebView, oldScale: Float, newScale: Float) {
        if (view.isShown && userPreferences.textReflowEnabled) {
            if (isReflowRunning)
                return
            val changeInPercent = abs(100 - 100 / zoomScale * newScale)
            if (changeInPercent > 2.5f && !isReflowRunning) {
                isReflowRunning = view.postDelayed({
                    zoomScale = newScale
                    view.evaluateJavascript(textReflow.provideJs()) { isReflowRunning = false }
                }, 100)
            }
        }
    }

    override fun onReceivedHttpAuthRequest(
        view: WebView,
        handler: HttpAuthHandler,
        host: String,
        realm: String
    ) {
        val context = view.context
        AlertDialog.Builder(context).apply {
            val dialogView = DialogAuthRequestBinding.inflate(LayoutInflater.from(context))

            val realmLabel = dialogView.authRequestRealmTextview
            val name = dialogView.authRequestUsernameEdittext
            val password = dialogView.authRequestPasswordEdittext

            realmLabel.text = context.getString(R.string.label_realm, realm)

            setView(dialogView.root)
            setTitle(R.string.title_sign_in)
            setCancelable(true)
            setPositiveButton(R.string.title_sign_in) { _, _ ->
                val user = name.text.toString()
                val pass = password.text.toString()
                handler.proceed(user.trim(), pass.trim())
                logger.log(TAG, "Attempting HTTP Authentication")
            }
            setNegativeButton(R.string.action_cancel) { _, _ ->
                handler.cancel()
            }
        }.resizeAndShow()
    }

    override fun onFormResubmission(view: WebView, dontResend: Message, resend: Message) {
        val context = view.context
        AlertDialog.Builder(context).apply {
            setTitle(context.getString(R.string.title_form_resubmission))
            setMessage(context.getString(R.string.message_form_resubmission))
            setCancelable(true)
            setPositiveButton(context.getString(R.string.action_yes)) { _, _ ->
                resend.sendToTarget()
            }
            setNegativeButton(context.getString(R.string.action_no)) { _, _ ->
                dontResend.sendToTarget()
            }
        }.resizeAndShow()
    }

    @SuppressLint("WebViewClientOnReceivedSslError")
    override fun onReceivedSslError(webView: WebView, handler: SslErrorHandler, error: SslError) {
        val context = webView.context
        urlWithSslError = webView.url

        sslState = SslState.Invalid(error)
        sslStateObservable.onNext(sslState)
        sslState = SslState.Invalid(error)

        when (sslWarningPreferences.recallBehaviorForDomain(webView.url)) {
            SslWarningPreferences.Behavior.PROCEED -> return handler.proceed()
            SslWarningPreferences.Behavior.CANCEL -> return handler.cancel()
            null -> Unit
        }

        val errorCodeMessageCodes = error.getAllSslErrorMessageCodes()

        val stringBuilder = StringBuilder()
        for (messageCode in errorCodeMessageCodes) {
            stringBuilder.append(" - ").append(context.getString(messageCode)).append('\n')
        }
        val alertMessage =
            context.getString(R.string.message_insecure_connection, stringBuilder.toString())

        AlertDialog.Builder(context).apply {
            val view = DialogSslWarningBinding.inflate(LayoutInflater.from(context))
            val dontAskAgain = view.checkBoxDontAskAgain
            setTitle(context.getString(R.string.title_warning))
            setMessage(alertMessage)
            setCancelable(true)
            setView(view.root)
            setOnCancelListener { handler.cancel() }
            setPositiveButton(context.getString(R.string.action_yes)) { _, _ ->
                if (dontAskAgain.isChecked) {
                    sslWarningPreferences.rememberBehaviorForDomain(
                        webView.url.orEmpty(),
                        SslWarningPreferences.Behavior.PROCEED
                    )
                }
                handler.proceed()
            }
            setNegativeButton(context.getString(R.string.action_no)) { _, _ ->
                if (dontAskAgain.isChecked) {
                    sslWarningPreferences.rememberBehaviorForDomain(
                        webView.url.orEmpty(),
                        SslWarningPreferences.Behavior.CANCEL
                    )
                }
                handler.cancel()
            }
        }.resizeAndShow()
    }

    @Deprecated("Deprecated in Java")
    override fun shouldOverrideUrlLoading(view: WebView, url: String): Boolean {
        return urlHandler.shouldOverrideLoading(view, url, headers) ||
                super.shouldOverrideUrlLoading(view, url)
    }

    override fun shouldOverrideUrlLoading(view: WebView, request: WebResourceRequest): Boolean {
        return urlHandler.shouldOverrideLoading(view, request.url.toString(), headers) ||
                super.shouldOverrideUrlLoading(view, request)
    }

    override fun shouldInterceptRequest(
        view: WebView,
        request: WebResourceRequest
    ): WebResourceResponse? {
        if (shouldBlockRequest(currentUrl, request.url.toString())) {
            val empty = ByteArrayInputStream(emptyResponseByteArray)
            return WebResourceResponse(BLOCKED_RESPONSE_MIME_TYPE, BLOCKED_RESPONSE_ENCODING, empty)
        }
        return if (request.url.path?.startsWith(files.path) == true) {
            filesStoragePathHandler.handle(request.url.path!!.substring(files.path.length))
        } else if (request.url.path?.startsWith(cache.path) == true) {
            cacheStoragePathHandler.handle(request.url.path!!.substring(cache.path.length))
        } else {
            super.shouldInterceptRequest(view, request)
        }
    }

    private fun SslError.getAllSslErrorMessageCodes(): List<Int> {
        val errorCodeMessageCodes = ArrayList<Int>(1)

        if (hasError(SslError.SSL_DATE_INVALID)) {
            errorCodeMessageCodes.add(R.string.message_certificate_date_invalid)
        }
        if (hasError(SslError.SSL_EXPIRED)) {
            errorCodeMessageCodes.add(R.string.message_certificate_expired)
        }
        if (hasError(SslError.SSL_IDMISMATCH)) {
            errorCodeMessageCodes.add(R.string.message_certificate_domain_mismatch)
        }
        if (hasError(SslError.SSL_NOTYETVALID)) {
            errorCodeMessageCodes.add(R.string.message_certificate_not_yet_valid)
        }
        if (hasError(SslError.SSL_UNTRUSTED)) {
            errorCodeMessageCodes.add(R.string.message_certificate_untrusted)
        }
        if (hasError(SslError.SSL_INVALID)) {
            errorCodeMessageCodes.add(R.string.message_certificate_invalid)
        }

        return errorCodeMessageCodes
    }

    /**
     * FIXED: Proper cleanup when WebViewClient is destroyed
     */
    fun cleanup() {
        try {
            webViewClientScope.cancel()
            phishingDetectionInProgress.set(false)
        } catch (e: Exception) {
            Log.e(TAG, "Error during cleanup", e)
        }
    }

    /**
     * The factory for constructing the client.
     */
    @AssistedFactory
    interface Factory {

        /**
         * Create the client.
         */
        fun create(
            headers: Map<String, String>,
            @Assisted("cache") cacheStoragePathHandler: InternalStoragePathHandler,
            @Assisted("files") filesStoragePathHandler: InternalStoragePathHandler,
        ): TabWebViewClient
    }

    companion object {
        private const val TAG = "TabWebViewClient"

        private val emptyResponseByteArray: ByteArray = byteArrayOf()

        private const val BLOCKED_RESPONSE_MIME_TYPE = "text/plain"
        private const val BLOCKED_RESPONSE_ENCODING = "utf-8"
    }
}