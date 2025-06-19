package acr.browser.lightning.phishing

import io.reactivex.rxjava3.core.Observable
import io.reactivex.rxjava3.subjects.PublishSubject

/**
 * Global bir event bus ile phishing tespitlerini bildiren sınıf
 */
object PhishingEventBus {
    private val phishingSubject = PublishSubject.create<Pair<String, Float>>()

    fun observePhishingEvents(): Observable<Pair<String, Float>> = phishingSubject.hide()

    fun reportPhishing(url: String, confidence: Float) {
        println("PhishingEventBus: Reporting phishing event for $url with confidence $confidence")
        phishingSubject.onNext(url to confidence)
    }
}