package acr.browser.lightning.browser.di

import acr.browser.lightning.phishing.FirebaseService
import acr.browser.lightning.phishing.ModelManager
import acr.browser.lightning.phishing.PhishingDetector
import acr.browser.lightning.phishing.ScanCounter
import android.content.Context
import com.google.firebase.firestore.FirebaseFirestore
import com.google.firebase.storage.FirebaseStorage
import dagger.Module
import dagger.Provides
import javax.inject.Singleton

@Module
class PhishingModule {

    @Provides
    @Singleton
    fun provideFirebaseFirestore(): FirebaseFirestore {
        return FirebaseFirestore.getInstance()
    }

    @Provides
    @Singleton
    fun provideFirebaseStorage(): FirebaseStorage {
        return FirebaseStorage.getInstance()
    }

    @Provides
    @Singleton
    fun provideModelManager(
        context: Context,
        firestore: FirebaseFirestore,
        firebaseStorage: FirebaseStorage
    ): ModelManager {
        return ModelManager(context, firestore, firebaseStorage)
    }

    @Provides
    @Singleton
    fun provideScanCounter(context: Context): ScanCounter {
        return ScanCounter(context)
    }

    @Provides
    @Singleton
    fun providePhishingDetector(
        context: Context,
        modelManager: ModelManager,
        scanCounter: ScanCounter
    ): PhishingDetector {
        val detector = PhishingDetector(context, modelManager, scanCounter)
        detector.initModel()
        return detector
    }

    @Provides
    @Singleton
    fun provideFirebaseService(
        phishingDetector: PhishingDetector,
        scanCounter: ScanCounter,
        firestore: FirebaseFirestore,
        firebaseStorage: FirebaseStorage
    ): FirebaseService {
        return FirebaseService(phishingDetector, scanCounter, firestore, firebaseStorage)
    }
}