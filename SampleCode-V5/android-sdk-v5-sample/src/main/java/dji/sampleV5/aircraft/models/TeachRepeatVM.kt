package dji.sampleV5.aircraft.models

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import android.os.Environment
import android.os.Handler
import android.os.Looper
import androidx.lifecycle.MutableLiveData
import dji.sdk.keyvalue.key.FlightControllerKey
import dji.sdk.keyvalue.key.RemoteControllerKey
import dji.sdk.keyvalue.value.common.ComponentIndexType
import dji.v5.et.action
import dji.sdk.keyvalue.value.flightcontroller.FlightCoordinateSystem
import dji.sdk.keyvalue.value.flightcontroller.RollPitchControlMode
import dji.sdk.keyvalue.value.flightcontroller.VerticalControlMode
import dji.sdk.keyvalue.value.flightcontroller.YawControlMode
import dji.v5.common.callback.CommonCallbacks
import dji.v5.common.error.IDJIError
import dji.v5.et.create
import dji.v5.et.listen
import dji.v5.manager.KeyManager
import dji.v5.manager.aircraft.virtualstick.VirtualStickManager
import dji.v5.manager.datacenter.MediaDataCenter
import dji.v5.manager.interfaces.ICameraStreamManager
import dji.v5.manager.interfaces.ICameraStreamManager.FrameFormat
import dji.v5.utils.common.DJIExecutor
import dji.v5.utils.common.LogUtils
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.FileOutputStream
import java.io.PrintWriter
import java.net.InetSocketAddress
import java.net.Socket
import java.nio.ByteBuffer
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.atomic.AtomicBoolean

class TeachRepeatVM : DJIViewModel() {

    private val TAG = "TeachRepeatVM"

    // RC Stick ステータス (-660 ~ 660)
    val stickValue = MutableLiveData(RCStickValue(0, 0, 0, 0))

    // Teach モード (録画) ステータス
    val isTeachRecording = MutableLiveData(false)
    val teachStatusText = MutableLiveData("Teach Recording: Stopped\nSaved frames: 0")
    private var teachFrameCount = 0
    private var currentFlightDir: File? = null
    private var csvWriter: PrintWriter? = null

    // Repeat モード (自動飛行) ステータス
    val isRepeatRunning = MutableLiveData(false)
    val repeatStatusText = MutableLiveData("Repeat Flight: Stopped\nServer: Not Connected")
    private var tcpSocket: Socket? = null
    private val isSocketConnected = AtomicBoolean(false)

    // 最新のカメラフレームキャッシュ (NV21 / RGBA)
    private var latestFrameData: ByteArray? = null
    private var latestFrameWidth = 0
    private var latestFrameHeight = 0
    private var latestFrameFormat = FrameFormat.NV21

    private val mainHandler = Handler(Looper.getMainLooper())
    private var timerRunnable: Runnable? = null
    private val INTERVAL_MS = 100L // 10Hz (100ms)

    private val cameraFrameListener = object : ICameraStreamManager.CameraFrameListener {
        override fun onFrame(frameData: ByteArray, offset: Int, length: Int, width: Int, height: Int, format: FrameFormat) {
            synchronized(this@TeachRepeatVM) {
                if (latestFrameData == null || latestFrameData!!.size != length) {
                    latestFrameData = ByteArray(length)
                }
                System.arraycopy(frameData, offset, latestFrameData!!, 0, length)
                latestFrameWidth = width
                latestFrameHeight = height
                latestFrameFormat = format
            }
        }
    }

    init {
        listenRCStick()
    }

    private fun listenRCStick() {
        // 右スティック Horizontal (Roll: 左右傾き)
        RemoteControllerKey.KeyStickRightHorizontal.create().listen(this) { val1 ->
            val1?.let { stickValue.value?.roll = it }
            stickValue.postValue(stickValue.value)
        }
        // 右スティック Vertical (Pitch: 前後)
        RemoteControllerKey.KeyStickRightVertical.create().listen(this) { val2 ->
            val2?.let { stickValue.value?.pitch = it }
            stickValue.postValue(stickValue.value)
        }
        // 左スティック Horizontal (Yaw: 旋回)
        RemoteControllerKey.KeyStickLeftHorizontal.create().listen(this) { val3 ->
            val3?.let { stickValue.value?.yaw = it }
            stickValue.postValue(stickValue.value)
        }
        // 左スティック Vertical (Throttle: 上下)
        RemoteControllerKey.KeyStickLeftVertical.create().listen(this) { val4 ->
            val4?.let { stickValue.value?.throttle = it }
            stickValue.postValue(stickValue.value)
        }
    }

    fun startTakeOff() {
        FlightControllerKey.KeyStartTakeoff.create().action({
            dji.v5.ux.util.ToastUtils.showToast("Takeoff Success")
        }, { error: IDJIError ->
            dji.v5.ux.util.ToastUtils.showToast("Takeoff Failed: ${error.description()}")
        })
    }

    fun startLanding() {
        FlightControllerKey.KeyStartAutoLanding.create().action({
            dji.v5.ux.util.ToastUtils.showToast("Landing Success")
        }, { error: IDJIError ->
            dji.v5.ux.util.ToastUtils.showToast("Landing Failed: ${error.description()}")
        })
    }

    fun startCameraStream(cameraIndex: ComponentIndexType) {
        MediaDataCenter.getInstance().cameraStreamManager.addFrameListener(
            cameraIndex,
            FrameFormat.NV21,
            cameraFrameListener
        )
    }

    fun stopCameraStream() {
        MediaDataCenter.getInstance().cameraStreamManager.removeFrameListener(cameraFrameListener)
    }

    // -----------------------------------------------------------------------------------------
    // 1. Teach モード (データ録画機能)
    // -----------------------------------------------------------------------------------------

    fun startTeachRecording(context: Context) {
        if (isTeachRecording.value == true) return

        val sdf = SimpleDateFormat("yyyyMMDD_HHmmss", Locale.getDefault())
        val flightName = "flight_${sdf.format(Date())}"

        val baseDir = context.getExternalFilesDir(Environment.DIRECTORY_DOCUMENTS) ?: context.filesDir
        currentFlightDir = File(baseDir, "TeachRepeatData/$flightName")
        if (!currentFlightDir!!.exists()) {
            currentFlightDir!!.mkdirs()
        }

        val csvFile = File(currentFlightDir, "commands.csv")
        csvWriter = PrintWriter(FileOutputStream(csvFile, true))
        csvWriter?.println("timestamp,roll,pitch,yaw,throttle")

        teachFrameCount = 0
        isTeachRecording.value = true
        teachStatusText.value = "Teach Recording: STARTED\nDir: ${currentFlightDir?.name}\nSaved: 0 frames"

        startTimerLoop()
    }

    fun stopTeachRecording() {
        if (isTeachRecording.value != true) return

        isTeachRecording.value = false
        stopTimerLoop()

        csvWriter?.flush()
        csvWriter?.close()
        csvWriter = null

        teachStatusText.value = "Teach Recording: STOPPED\nSaved Total: $teachFrameCount frames\nPath: ${currentFlightDir?.absolutePath}"
    }

    private fun processTeachFrame() {
        val bytes: ByteArray?
        val w: Int
        val h: Int
        val fmt: FrameFormat
        synchronized(this) {
            bytes = latestFrameData?.clone()
            w = latestFrameWidth
            h = latestFrameHeight
            fmt = latestFrameFormat
        }

        if (bytes == null || w <= 0 || h <= 0) return

        DJIExecutor.getExecutor().execute {
            try {
                teachFrameCount++
                val timeSec = teachFrameCount * 0.1

                // 画像保存 (000001.jpg, 000002.jpg...)
                val imgFile = File(currentFlightDir, String.format(Locale.US, "%06d.jpg", teachFrameCount))
                saveFrameToJpeg(bytes, w, h, fmt, imgFile)

                // コマンドの CSV 保存
                val sv = stickValue.value ?: RCStickValue(0, 0, 0, 0)
                csvWriter?.println(String.format(Locale.US, "%.2f,%d,%d,%d,%d", timeSec, sv.roll, sv.pitch, sv.yaw, sv.throttle))
                csvWriter?.flush()

                mainHandler.post {
                    teachStatusText.value = "Teach Recording: RUNNING\nFlight: ${currentFlightDir?.name}\nSaved: $teachFrameCount frames\nRoll:${sv.roll} Pitch:${sv.pitch} Yaw:${sv.yaw}"
                }
            } catch (e: Exception) {
                LogUtils.e(TAG, "Error saving teach frame: ${e.message}")
            }
        }
    }

    // -----------------------------------------------------------------------------------------
    // 2. Repeat モード (リアルタイム自動飛行機能)
    // -----------------------------------------------------------------------------------------

    fun startRepeatFlight(ip: String, port: Int) {
        if (isRepeatRunning.value == true) return

        // VirtualStick 有効化
        VirtualStickManager.getInstance().enableVirtualStick(object : CommonCallbacks.CompletionCallback {
            override fun onSuccess() {
                // 重要: アドバンストモードを OFF にして標準ジョイスティック制御 (-660 ~ 660) を完全有効化
                VirtualStickManager.getInstance().setVirtualStickAdvancedModeEnabled(false)
                VirtualStickManager.getInstance().speedLevel = 1.0

                // TCP サーバ接続
                DJIExecutor.getExecutor().execute {
                    try {
                        tcpSocket = Socket()
                        tcpSocket?.connect(InetSocketAddress(ip, port), 3000)
                        isSocketConnected.set(true)

                        mainHandler.post {
                            isRepeatRunning.value = true
                            repeatStatusText.value = "Repeat Flight: RUNNING\nConnected to $ip:$port"
                            startTimerLoop()
                        }
                    } catch (e: Exception) {
                        mainHandler.post {
                            repeatStatusText.value = "Connect Failed: ${e.message}"
                            disableVirtualStick()
                        }
                    }
                }
            }

            override fun onFailure(error: IDJIError) {
                repeatStatusText.value = "VirtualStick Enable Failed: ${error.description()}"
            }
        })
    }

    fun stopRepeatFlight() {
        if (isRepeatRunning.value != true) return

        isRepeatRunning.value = false
        stopTimerLoop()

        DJIExecutor.getExecutor().execute {
            try {
                isSocketConnected.set(false)
                tcpSocket?.close()
                tcpSocket = null
            } catch (e: Exception) {
                // ignore
            }
        }

        disableVirtualStick()
        repeatStatusText.value = "Repeat Flight: STOPPED"
    }

    private fun disableVirtualStick() {
        VirtualStickManager.getInstance().disableVirtualStick(null)
    }

    private fun processRepeatFrame() {
        if (!isSocketConnected.get() || tcpSocket == null) return

        val bytes: ByteArray?
        val w: Int
        val h: Int
        val fmt: FrameFormat
        synchronized(this) {
            bytes = latestFrameData?.clone()
            w = latestFrameWidth
            h = latestFrameHeight
            fmt = latestFrameFormat
        }

        if (bytes == null || w <= 0 || h <= 0) return

        DJIExecutor.getExecutor().execute {
            try {
                // 1. JPEG バイト列へ圧縮 (224x224 にリサイズ)
                val jpegBytes = frameToJpegBytes(bytes, w, h, fmt, 224, 224) ?: return@execute

                // 2. TCP 送信 (4バイト長 + JPEG)
                val outputStream = tcpSocket?.getOutputStream() ?: return@execute
                val lengthBuffer = ByteBuffer.allocate(4).putInt(jpegBytes.size).array()
                outputStream.write(lengthBuffer)
                outputStream.write(jpegBytes)
                outputStream.flush()

                // 3. 応答 JSON 受信
                val inputStream = tcpSocket?.getInputStream() ?: return@execute
                val reader = java.io.BufferedReader(java.io.InputStreamReader(inputStream))
                val jsonLine = reader.readLine() ?: return@execute

                val json = JSONObject(jsonLine)
                val roll = json.optInt("roll", 0)
                val pitch = json.optInt("pitch", 0)
                val yaw = json.optInt("yaw", 0)
                val throttle = json.optInt("throttle", 0)

                // 4. 標準スティック制御 (プロポスティックと同じ -660 ~ 660 値を代入)
                VirtualStickManager.getInstance().rightStick.horizontalPosition = roll
                VirtualStickManager.getInstance().rightStick.verticalPosition = pitch
                VirtualStickManager.getInstance().leftStick.horizontalPosition = yaw
                VirtualStickManager.getInstance().leftStick.verticalPosition = throttle

                mainHandler.post {
                    repeatStatusText.value = "Repeat Active\nRecv Cmd -> Roll:$roll Pitch:$pitch Yaw:$yaw"
                }
            } catch (e: Exception) {
                LogUtils.e(TAG, "Error in repeat loop: ${e.message}")
            }
        }
    }

    // -----------------------------------------------------------------------------------------
    // タイマー・画像処理ヘルパー
    // -----------------------------------------------------------------------------------------

    private fun startTimerLoop() {
        if (timerRunnable != null) return
        timerRunnable = object : Runnable {
            override fun run() {
                if (isTeachRecording.value == true) {
                    processTeachFrame()
                }
                if (isRepeatRunning.value == true) {
                    processRepeatFrame()
                }
                mainHandler.postDelayed(this, INTERVAL_MS)
            }
        }
        mainHandler.post(timerRunnable!!)
    }

    private fun stopTimerLoop() {
        if (isTeachRecording.value != true && isRepeatRunning.value != true) {
            timerRunnable?.let { mainHandler.removeCallbacks(it) }
            timerRunnable = null
        }
    }

    private fun saveFrameToJpeg(bytes: ByteArray, w: Int, h: Int, format: FrameFormat, outputFile: File) {
        val bos = ByteArrayOutputStream()
        val yuvImage = YuvImage(bytes, ImageFormat.NV21, w, h, null)
        yuvImage.compressToJpeg(Rect(0, 0, w, h), 90, bos)
        val jpegData = bos.toByteArray()

        FileOutputStream(outputFile).use { fos ->
            fos.write(jpegData)
            fos.flush()
        }
    }

    private fun frameToJpegBytes(bytes: ByteArray, w: Int, h: Int, format: FrameFormat, reqWidth: Int, reqHeight: Int): ByteArray? {
        val bos = ByteArrayOutputStream()
        val yuvImage = YuvImage(bytes, ImageFormat.NV21, w, h, null)
        yuvImage.compressToJpeg(Rect(0, 0, w, h), 85, bos)
        val rawJpeg = bos.toByteArray()

        val origBmp = BitmapFactory.decodeByteArray(rawJpeg, 0, rawJpeg.size) ?: return null
        val scaledBmp = Bitmap.createScaledBitmap(origBmp, reqWidth, reqHeight, true)

        val outBos = ByteArrayOutputStream()
        scaledBmp.compress(Bitmap.CompressFormat.JPEG, 90, outBos)
        origBmp.recycle()
        scaledBmp.recycle()
        return outBos.toByteArray()
    }

    override fun onCleared() {
        super.onCleared()
        stopCameraStream()
        stopTeachRecording()
        stopRepeatFlight()
        KeyManager.getInstance().cancelListen(this)
    }

    data class RCStickValue(
        var roll: Int = 0,
        var pitch: Int = 0,
        var yaw: Int = 0,
        var throttle: Int = 0
    )
}
