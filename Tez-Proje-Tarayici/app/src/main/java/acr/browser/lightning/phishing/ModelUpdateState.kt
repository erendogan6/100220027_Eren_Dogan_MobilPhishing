// ModelUpdateState.kt
package acr.browser.lightning.phishing

sealed class ModelUpdateState {
    object Idle : ModelUpdateState()
    object Checking : ModelUpdateState()
    data class UpToDate(val currentVersion: Float) : ModelUpdateState()
    data class Downloading(val newVersion: Float, val progress: Float = 0f) : ModelUpdateState()
    data class Success(val newVersion: Float) : ModelUpdateState()
    data class Error(val message: String) : ModelUpdateState()
}