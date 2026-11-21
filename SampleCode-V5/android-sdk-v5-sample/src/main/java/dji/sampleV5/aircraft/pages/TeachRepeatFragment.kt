package dji.sampleV5.aircraft.pages

import android.os.Bundle
import android.view.LayoutInflater
import android.view.SurfaceHolder
import android.view.SurfaceView
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import androidx.fragment.app.viewModels
import dji.sampleV5.aircraft.R
import dji.sampleV5.aircraft.models.TeachRepeatVM
import dji.sdk.keyvalue.value.common.ComponentIndexType
import dji.v5.manager.datacenter.MediaDataCenter
import dji.v5.manager.interfaces.ICameraStreamManager

class TeachRepeatFragment : DJIFragment(), SurfaceHolder.Callback {

    private val viewModel: TeachRepeatVM by viewModels()

    private lateinit var cameraSurfaceView: SurfaceView
    private lateinit var btnTakeoff: Button
    private lateinit var btnLanding: Button
    private lateinit var tvStickStatus: TextView
    private lateinit var btnStartTeach: Button
    private lateinit var btnStopTeach: Button
    private lateinit var tvTeachStatus: TextView

    private lateinit var etServerIp: EditText
    private lateinit var etServerPort: EditText
    private lateinit var btnStartRepeat: Button
    private lateinit var btnStopRepeat: Button
    private lateinit var tvRepeatStatus: TextView

    private var cameraIndex = ComponentIndexType.LEFT_OR_MAIN

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        return inflater.inflate(R.layout.fragment_teach_repeat, container, false)
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        cameraSurfaceView = view.findViewById(R.id.sv_camera)
        btnTakeoff = view.findViewById(R.id.btn_takeoff)
        btnLanding = view.findViewById(R.id.btn_landing)
        tvStickStatus = view.findViewById(R.id.tv_stick_status)
        btnStartTeach = view.findViewById(R.id.btn_start_teach)
        btnStopTeach = view.findViewById(R.id.btn_stop_teach)
        tvTeachStatus = view.findViewById(R.id.tv_teach_status)

        etServerIp = view.findViewById(R.id.et_server_ip)
        etServerPort = view.findViewById(R.id.et_server_port)
        btnStartRepeat = view.findViewById(R.id.btn_start_repeat)
        btnStopRepeat = view.findViewById(R.id.btn_stop_repeat)
        tvRepeatStatus = view.findViewById(R.id.tv_repeat_status)

        cameraSurfaceView.holder.addCallback(this)

        initObservers()
        initListeners()
    }

    private fun initObservers() {
        // プロポ スティック値のリアルタイム監視
        viewModel.stickValue.observe(viewLifecycleOwner) { sv ->
            tvStickStatus.text = String.format(
                "Roll: %d | Pitch: %d | Yaw: %d | Throttle: %d",
                sv.roll, sv.pitch, sv.yaw, sv.throttle
            )
        }

        // Teach モードのステータス監視
        viewModel.isTeachRecording.observe(viewLifecycleOwner) { isRecording ->
            btnStartTeach.isEnabled = !isRecording
            btnStopTeach.isEnabled = isRecording
        }

        viewModel.teachStatusText.observe(viewLifecycleOwner) { status ->
            tvTeachStatus.text = status
        }

        // Repeat モードのステータス監視
        viewModel.isRepeatRunning.observe(viewLifecycleOwner) { isRunning ->
            btnStartRepeat.isEnabled = !isRunning
            btnStopRepeat.isEnabled = isRunning
            etServerIp.isEnabled = !isRunning
            etServerPort.isEnabled = !isRunning
        }

        viewModel.repeatStatusText.observe(viewLifecycleOwner) { status ->
            tvRepeatStatus.text = status
        }
    }

    private fun initListeners() {
        btnTakeoff.setOnClickListener {
            viewModel.startTakeOff()
        }

        btnLanding.setOnClickListener {
            viewModel.startLanding()
        }

        btnStartTeach.setOnClickListener {
            context?.let { ctx ->
                viewModel.startTeachRecording(ctx)
            }
        }

        btnStopTeach.setOnClickListener {
            viewModel.stopTeachRecording()
        }

        btnStartRepeat.setOnClickListener {
            val ip = etServerIp.text.toString().trim()
            val portStr = etServerPort.text.toString().trim()
            val port = portStr.toIntOrNull() ?: 8000
            if (ip.isNotEmpty()) {
                viewModel.startRepeatFlight(ip, port)
            }
        }

        btnStopRepeat.setOnClickListener {
            viewModel.stopRepeatFlight()
        }
    }

    override fun surfaceCreated(holder: SurfaceHolder) {
        MediaDataCenter.getInstance().cameraStreamManager.putCameraStreamSurface(
            cameraIndex,
            holder.surface,
            cameraSurfaceView.width,
            cameraSurfaceView.height,
            ICameraStreamManager.ScaleType.CENTER_INSIDE
        )
        viewModel.startCameraStream(cameraIndex)
    }

    override fun surfaceChanged(holder: SurfaceHolder, format: Int, width: Int, height: Int) {
        MediaDataCenter.getInstance().cameraStreamManager.putCameraStreamSurface(
            cameraIndex,
            holder.surface,
            width,
            height,
            ICameraStreamManager.ScaleType.CENTER_INSIDE
        )
    }

    override fun surfaceDestroyed(holder: SurfaceHolder) {
        MediaDataCenter.getInstance().cameraStreamManager.removeCameraStreamSurface(holder.surface)
        viewModel.stopCameraStream()
    }
}
