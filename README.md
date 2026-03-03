# Blendshapes LiveLink Tracker

Real-time face tracking pipeline that sends ARKit-style blendshapes and head pose over UDP to Unreal Live Link.

## What it does

- Captures webcam frames.
- Runs MediaPipe face landmark + blendshape inference.
- Optionally applies emotion-based and eye post-processing.
- Sends blendshapes/head pose over UDP to a Live Link target.
- Accepts OSC/UDP control commands to change sender behavior at runtime.

## Project layout

- `main.py`: entry point.
- `core/`: app config, camera capture, face landmarker, shared landmarker state.
- `network/`: Live Link sender, OSC UDP command server/handler, Live Link payload model.
- `utils/`: supporting processors (head pose, scheduler, input handling, overlays, emotion).

## Installation (Conda)

### 1. Create environment

Use the complete environment file:

```bash
conda env create --name blendshapes --file facetracking_environment_complete.yml
```

If the environment already exists:

```bash
conda env update --name blendshapes --file facetracking_environment_complete.yml --prune
```

### 2. Activate environment

```bash
conda activate blendshapes
```

### 3. Verify model files exist

Required model paths (defaults):

- `./models/face_landmarker.task`
- `./models/enet_b0_8_best_afew.onnx`

### 4. Run

```bash
python main.py
```

Or on macOS, double-click/run:

```bash
./run_blendshapes.command
```

## Configuration (`config.json`)

`main.py` loads `config.json` and fills any missing keys with defaults from `core/app_config.py`.

| Key | Type | Default | Description |
|---|---|---|---|
| `SOURCE` | int / str | `0` | Camera source (`0` for default webcam, or device/path supported by OpenCV). |
| `FLIP_IMAGE` | bool | `false` | Mirror input horizontally before processing. |
| `TARGET_SIZE` | int | `1024` | Square crop size used in processing/display. |
| `LIVE_LINK_IP` | str | `192.168.100.2` | Destination IP for outgoing Live Link UDP packets. |
| `LIVE_LINK_PORT` | int | `11111` | Destination port for outgoing Live Link UDP packets. |
| `LIVE_LINK_CLIENT_NAME` | str | `Python_LiveLinkFace` | Subject/client name encoded in outgoing Live Link payloads. |
| `FACE_MODEL_PATH` | str | `./models/face_landmarker.task` | MediaPipe face landmarker model path. |
| `TARGET_FPS` | int | `30` | Main loop / sender target frame rate. |
| `DISPLAY_VIDEO` | bool | `true` | Show debug video window with overlay. |
| `SHOW_FPS` | bool | `false` | Print per-frame processing duration. |
| `BLENDSHAPE_SWAP_LR` | bool | `false` | Swap left/right blendshape names before sending. |
| `PAIR_EYELIDS` | bool | `true` | Average left/right blink values unless right blink override is forced. |
| `HP_FILTER_WINDOW` | int | `6` | Smoothing window size for head pose processing. |
| `HP_MAX_YAW` | float | `45.0` | Max absolute yaw angle used for normalization/clamping. |
| `HP_MAX_PITCH` | float | `20.0` | Max absolute pitch angle used for normalization/clamping. |
| `HP_MAX_ROLL` | float | `45.0` | Max absolute roll angle used for normalization/clamping. |
| `HP_YAW_OFFSET` | float | `0.0` | Calibration offset (degrees) added to yaw before normalization. |
| `HP_PITCH_OFFSET` | float | `0.0` | Calibration offset (degrees) added to pitch before normalization. |
| `HP_EULER_ORDER` | str | `"yxz"` | Euler decomposition order for head pose conversion. |
| `EMOTION_MODEL_PATH` | str | `./models/enet_b0_8_best_afew.onnx` | Emotion model path for emotion worker. |
| `EVERY_FPS` | int | `6` | Emotion inference cadence (process every N frames). |
| `EMOTION_RECOGNITION_ENABLED` | bool | `true` | Enables/disables emotion worker thread. |
| `POST_PROCESS_BLENDSHAPES` | bool | `true` | Enables expression profile blendshape post-processing. |
| `EXPRESSION_CONFIG_PATH` | str | `expression_profiles.json` | Expression profile config file path. |
| `EYE_POST_PROCESSOR` | bool | `true` | Enables dedicated eye post-processing pass. |
| `UDP_COMMAND_IP` | str | `127.0.0.1` | IP address for inbound OSC command server bind. |
| `UDP_COMMAND_PORT` | int | `12000` | Port for inbound OSC command server bind. |
| `UDP_STATE_IP` | str | `127.0.0.1` | Destination IP for outbound OSC state messages. |
| `UDP_STATE_PORT` | int | `12001` | Destination port for outbound OSC state messages. |

## Keyboard controls

Current runtime keys:

- `0`: reloads expression config (`reload_config` action).
- `Shift + Esc`: cleanly stops the application.

## UDP / OSC functions

The app listens on `UDP_COMMAND_IP:UDP_COMMAND_PORT` and accepts OSC messages.

### Supported OSC addresses

| Address | Args | Effect |
|---|---|---|
| `/livelink/normal` | none | Set sender mode to normal (live tracking data). |
| `/livelink/start` | none | Alias for `/livelink/normal`. |
| `/livelink/neutral` | none | Set sender mode to neutral pose. |
| `/livelink/stop` | none | Alias for `/livelink/neutral`. |
| `/livelink/random` | optional `rate_hz` (float/int) | Set random mode, optional random refresh rate. |
| `/livelink/random_rate` | `rate_hz` (float/int) | Update random mode refresh rate. |
| `/livelink/random_slow` | none | Random mode with `1.0 Hz`. |
| `/livelink/random_fast` | none | Random mode with `TARGET_FPS` rate. |
| `/livelink/blink_right` | optional bool-like arg | With arg: set forced right blink on/off. Without arg: toggle. |
| `/livelink/tongue_out` | optional bool-like arg | With arg: set tongue-out on/off. Without arg: toggle. |
| `/livelink/tracking` | optional bool-like arg | With arg (`0/1`, `false/true`, etc.): set face-tracking send OFF/ON. Without arg: toggle face-tracking send. |
| `/get_state` or `get_state` | none | Immediately publishes current `/livelink/code` and `/livelink/tracking`. |

## Outbound OSC state

The app also sends OSC status messages to `UDP_STATE_IP:UDP_STATE_PORT`.

- `/livelink/state`:
  - Emitted only after the first successful LiveLink UDP send.
  - `1` when the sender becomes effectively reachable/sending.
  - `0` when that state drops or on shutdown.
- `/livelink/code`:
  - `1` while the app code is running.
  - `0` on shutdown.
- `/livelink/tracking`:
  - `1` when face-tracking send is enabled.
  - `0` when face-tracking send is disabled.
