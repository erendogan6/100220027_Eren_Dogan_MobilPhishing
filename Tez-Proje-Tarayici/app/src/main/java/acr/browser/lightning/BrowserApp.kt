package acr.browser.lightning

import acr.browser.lightning.browser.di.AppComponent
import acr.browser.lightning.browser.di.DaggerAppComponent
import acr.browser.lightning.browser.di.DatabaseScheduler
import acr.browser.lightning.browser.di.injector
import acr.browser.lightning.browser.proxy.ProxyAdapter
import acr.browser.lightning.database.bookmark.BookmarkExporter
import acr.browser.lightning.database.bookmark.BookmarkRepository
import acr.browser.lightning.device.BuildInfo
import acr.browser.lightning.device.BuildType
import acr.browser.lightning.log.Logger
import acr.browser.lightning.migration.Cleanup
import acr.browser.lightning.utils.FileUtils
import acr.browser.lightning.utils.LeakCanaryUtils
import android.app.Application
import android.os.Build
import android.util.Log
import android.webkit.WebView
import com.google.firebase.FirebaseApp
import com.google.firebase.storage.FirebaseStorage
import io.reactivex.rxjava3.core.Scheduler
import io.reactivex.rxjava3.core.Single
import io.reactivex.rxjava3.plugins.RxJavaPlugins
import kotlinx.coroutines.MainScope
import kotlinx.coroutines.launch
import java.io.File
import javax.inject.Inject
import kotlin.system.exitProcess

/**
 * The browser application.
 */
class BrowserApp : Application() {

    @Inject
    internal lateinit var leakCanaryUtils: LeakCanaryUtils

    @Inject
    internal lateinit var bookmarkModel: BookmarkRepository

    @Inject
    @DatabaseScheduler
    internal lateinit var databaseScheduler: Scheduler

    @Inject
    internal lateinit var logger: Logger

    @Inject
    internal lateinit var buildInfo: BuildInfo

    @Inject
    internal lateinit var proxyAdapter: ProxyAdapter

    @Inject
    internal lateinit var cleanup: Cleanup

    lateinit var applicationComponent: AppComponent

    override fun onCreate() {
        super.onCreate()

        initializeFirebase()

        MainScope().launch {
            cleanup.cleanup()
        }

        if (Build.VERSION.SDK_INT >= 28) {
            if (getProcessName() == "$packageName:incognito") {
                File(dataDir, "app_webview_incognito").deleteRecursively()
                WebView.setDataDirectorySuffix("incognito")
            }
        }

        val defaultHandler = Thread.getDefaultUncaughtExceptionHandler()

        Thread.setDefaultUncaughtExceptionHandler { thread, ex ->
            if (BuildConfig.DEBUG) {
                FileUtils.writeCrashToStorage(ex)
            }

            if (defaultHandler != null) {
                defaultHandler.uncaughtException(thread, ex)
            } else {
                exitProcess(2)
            }
        }

        RxJavaPlugins.setErrorHandler { throwable: Throwable? ->
            if (BuildConfig.DEBUG && throwable != null) {
                FileUtils.writeCrashToStorage(throwable)
                throw throwable
            }
        }

        applicationComponent = DaggerAppComponent.builder()
            .application(this)
            .buildInfo(createBuildInfo())
            .build()
        injector.inject(this)

        Single.fromCallable(bookmarkModel::count)
            .filter { it == 0L }
            .flatMapCompletable {
                val assetsBookmarks = BookmarkExporter.importBookmarksFromAssets(this@BrowserApp)
                bookmarkModel.addBookmarkList(assetsBookmarks)
            }
            .subscribeOn(databaseScheduler)
            .subscribe()

        if (buildInfo.buildType == BuildType.DEBUG) {
            leakCanaryUtils.setup()
        }

        if (buildInfo.buildType == BuildType.DEBUG) {
            WebView.setWebContentsDebuggingEnabled(true)
        }

        registerActivityLifecycleCallbacks(proxyAdapter)
    }

    /**
     * Firebase'i initialize et
     */
    private fun initializeFirebase() {
        try {
            Log.d(TAG, "Initializing Firebase...")

            // Firebase'i initialize et
            if (FirebaseApp.getApps(this).isEmpty()) {
                FirebaseApp.initializeApp(this)
                Log.d(TAG, "Firebase initialized successfully")
            } else {
                Log.d(TAG, "Firebase already initialized")
            }

        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize Firebase", e)
            // Firebase initialization başarısız olsa bile uygulama çalışmaya devam etmeli
        }
    }

    /**
     * Create the [BuildType] from the [BuildConfig].
     */
    private fun createBuildInfo() = BuildInfo(
        when {
            BuildConfig.DEBUG -> BuildType.DEBUG
            else -> BuildType.RELEASE
        }
    )

    /**
     * Firebase Storage reference'ı al (model indirme için)
     */
    fun getFirebaseStorage(): FirebaseStorage? {
        return try {
            FirebaseStorage.getInstance()
        } catch (e: Exception) {
            Log.e(TAG, "Failed to get Firebase Storage instance", e)
            null
        }
    }

    companion object {
        private const val TAG = "BrowserApp"
    }
}