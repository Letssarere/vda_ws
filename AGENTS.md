# VDA Streaming Workspace (ROS2 Humble)

This project implements a ROS2 node for **Video Depth Anything (Metric/Small)** in streaming mode. It runs on **ROS2 Humble** (Ubuntu 22.04) and is optimized for **Jetson Orin NX** with **Intel RealSense** camera input.

## 📂 Project Structure

The workspace is configured as a single Git repository. The core logic resides in `src/vda_streaming`.

```text
vda_ws/
├── .gitignore                  # Git ignore rules (build/, install/, log/, checkpoints/*.pth)
├── AGENTS.md                   # Project documentation (This file)
└── src/
    └── vda_streaming/          # Main ROS2 Package
        ├── package.xml         # ROS2 dependency definitions
        ├── setup.py            # Python package setup
        ├── setup.cfg           # Package configuration
        ├── launch/
        │   └── streaming.launch.py   # Launch file for VDA node & RealSense
        ├── checkpoints/        # Model weights directory
        │   ├── .gitkeep        # Git tracking for empty folder
        │   └── metric_video_depth_anything_vits.pth  # (Ignored by Git)
        └── vda_streaming/      # Python source code
            ├── __init__.py
            ├── node.py         # Main ROS2 Node (StreamingDepthNode)
            └── libs/           # External libraries (Video-Depth-Anything Source)
                ├── video_depth_anything/  # Core model definitions (Copied from upstream)
                └── utils/                 # Utility functions (Copied from upstream)
```

## 🛠️ Prerequisites & Setup

### 1. Environment

- **OS:** Ubuntu 22.04 LTS  
- **ROS2:** Humble Hawksbill  
- **Python:** 3.10+  
- **Hardware:** NVIDIA Jetson Orin NX (recommended), Intel RealSense D435i/D455  

### 2. Dependencies

Install required Python libraries for Video-Depth-Anything:

```bash
pip install torch torchvision opencv-python matplotlib
```

> Note: On Jetson, ensure you have torch and torchvision installed with CUDA support compatible with JetPack.

### 3. External Source Integration

This project embeds the Video-Depth-Anything source code to ensure stability.

1. Clone/Download Video-Depth-Anything.  
2. Copy the `video_depth_anything` folder and `utils` folder into `src/vda_streaming/vda_streaming/libs/`.

### 4. Model Weights

Download the Metric-Small model and place it in the checkpoints directory:

- **Model:** Metric-Video-Depth-Anything-Small  
- **Path:** `src/vda_streaming/checkpoints/metric_video_depth_anything_vits.pth`  
- **Download:** HuggingFace Link  

## 🚀 Build & Run

### Build

```bash
cd ~/vda_ws
colcon build --symlink-install
source install/setup.bash
```

### Run

Execute the launch file to start the RealSense camera and the Depth Estimation node.

```bash
ros2 launch vda_streaming streaming.launch.py
```

## 🧩 Node Details (`vda_streaming/node.py`)

- **Node Name:** `vda_streaming_node`

### Subscribed Topic

- `/camera/color/image_raw` (`sensor_msgs/Image`): RGB input from RealSense.

### Published Topic

- `/vda/depth_image` (`sensor_msgs/Image`): Estimated metric depth (`32FC1`).

### Parameters

- `encoder`: `vits` (default)
- `input_size`: `518` (default)
- `max_depth`: `20.0` (optional visualization limit)

## 📝 Developer Notes

- **Streaming Logic:** The node uses `infer_video_depth_one` which internally manages a sliding window cache of hidden states. Do not manually reset the cache unless the scene changes drastically.
- **Performance:** Inference runs in FP16 (autocast) by default for speed.
- **Git Policy:** Large files (model weights > 100MB) and build artifacts (`build/`, `install/`) are ignored via `.gitignore`.
