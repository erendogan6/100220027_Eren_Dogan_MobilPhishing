package acr.browser.lightning.phishing

import android.content.Context
import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class ScanCounter @Inject constructor(
    private val context: Context
) {
    private val prefs = context.getSharedPreferences("PhishingDetectorPrefs", Context.MODE_PRIVATE)

    // Toplam tarama sayısı
    private var totalScanCount: Int = prefs.getInt("totalScanCount", 0)
        set(value) {
            field = value
            prefs.edit().putInt("totalScanCount", value).apply()
        }

    // Son upload'dan sonraki tarama sayısı
    private var scansSinceLastUpload: Int = prefs.getInt("scansSinceLastUpload", 0)
        set(value) {
            field = value
            prefs.edit().putInt("scansSinceLastUpload", value).apply()
        }

    // Tarama sayacını artır
    fun incrementScanCount() {
        totalScanCount++
        scansSinceLastUpload++
    }

    // Upload gerekli mi kontrol et (her 5 taramada bir)
    fun shouldUploadModel(): Boolean {
        println("Scans since last upload: $scansSinceLastUpload")
        return scansSinceLastUpload % 5 == 0 && scansSinceLastUpload > 0
    }

    // Checkpoint gerekli mi kontrol et (her 5 taramada bir)
    fun shouldCreateCheckpoint(): Boolean {
        return scansSinceLastUpload % 5 == 0 && scansSinceLastUpload > 0
    }

    // Upload counter'ını sıfırla
    fun resetUploadCounter() {
        scansSinceLastUpload = 0
    }

    // Getter'lar
    fun getTotalScanCount(): Int = totalScanCount
    fun getScansSinceLastUpload(): Int = scansSinceLastUpload
    fun getScansUntilNextUpload(): Int {
        val remainder = scansSinceLastUpload % 5
        return if (remainder == 0) 5 else (5 - remainder)
    }
}