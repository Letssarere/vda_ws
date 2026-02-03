"""ROS2 streaming node for Video Depth Anything (metric/small)."""
from __future__ import annotations

import sys
import threading
from pathlib import Path

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
import torch


def _add_libs_to_path() -> None:
    libs_dir = Path(__file__).resolve().parent / 'libs'
    libs_path = str(libs_dir)
    if libs_dir.exists() and libs_path not in sys.path:
        sys.path.insert(0, libs_path)


_add_libs_to_path()

from video_depth_anything.video_depth_stream import (  # noqa: E402
    VideoDepthAnything,
)


class StreamingDepthNode(Node):
    """Streaming depth estimation node using Video Depth Anything."""

    def __init__(self) -> None:
        super().__init__('vda_streaming_node')
        self._bridge = CvBridge()
        self._declare_parameters()
        self._read_parameters()
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self._device == 'cpu' and not self._fp32:
            self.get_logger().warning(
                'CUDA not available; forcing fp32 inference on CPU.'
            )
            self._fp32 = True
        self._model = self._load_model()
        self._frame_height = None
        self._frame_width = None
        self._crop_box = (80, 60, 560, 420)
        self._crop_last = None
        self._warp_points = {
            'top_left': (120, 80),
            'top_right': (520, 80),
            'bottom_left': (120, 400),
            'bottom_right': (520, 400),
        }

        qos_profile = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
        )
        self._pub = self.create_publisher(
            Image,
            self._output_topic,
            qos_profile,
        )
        self._vis_pub = self.create_publisher(
            Image,
            '/vda/depth_vis',
            qos_profile,
        )
        self._warp_pub = self.create_publisher(
            Image,
            '/vda/depth_warp',
            qos_profile,
        )
        self._warp_bin_pub = self.create_publisher(
            Image,
            '/vda/depth_warp_bin',
            qos_profile,
        )
        self._sub = self.create_subscription(
            Image,
            self._input_topic,
            self._image_callback,
            qos_profile,
        )

        self._latest_msg = None
        self._running = True
        self._frame_lock = threading.Lock()
        self._frame_cond = threading.Condition(self._frame_lock)
        self._worker = threading.Thread(
            target=self._process_loop,
            name='vda_streaming_worker',
            daemon=True,
        )
        self._worker.start()
        self._log_startup()

    def _declare_parameters(self) -> None:
        self.declare_parameter('input_topic', '/camera/color/image_raw')
        self.declare_parameter('output_topic', '/vda/depth_image')
        self.declare_parameter('encoder', 'vits')
        self.declare_parameter('input_size', 518)
        self.declare_parameter('metric', True)
        self.declare_parameter('fp32', False)
        self.declare_parameter('max_depth', 20.0)
        self.declare_parameter('checkpoint_path', '')

    def _read_parameters(self) -> None:
        input_param = self.get_parameter('input_topic').get_parameter_value()
        output_param = self.get_parameter('output_topic').get_parameter_value()
        encoder_param = self.get_parameter('encoder').get_parameter_value()
        input_size_param = (
            self.get_parameter('input_size').get_parameter_value()
        )
        metric_param = self.get_parameter('metric').get_parameter_value()
        fp32_param = self.get_parameter('fp32').get_parameter_value()
        max_depth_param = self.get_parameter('max_depth').get_parameter_value()
        checkpoint_param = (
            self.get_parameter('checkpoint_path').get_parameter_value()
        )

        self._input_topic = input_param.string_value
        self._output_topic = output_param.string_value
        self._encoder = encoder_param.string_value
        self._input_size = int(input_size_param.integer_value)
        self._metric = metric_param.bool_value
        self._fp32 = fp32_param.bool_value
        self._max_depth = float(max_depth_param.double_value)
        self._checkpoint_path = checkpoint_param.string_value

    def _default_checkpoint_path(self) -> Path:
        checkpoints_dir = None
        try:
            from ament_index_python.packages import (  # noqa: WPS433
                get_package_share_directory,
            )

            share_dir = Path(
                get_package_share_directory('vda_streaming')
            )
            checkpoints_dir = share_dir / 'checkpoints'
        except Exception:  # noqa: BLE001
            checkpoints_dir = None

        if checkpoints_dir is None:
            package_dir = Path(__file__).resolve().parent
            checkpoints_dir = package_dir.parent / 'checkpoints'
        checkpoint_name = (
            'metric_video_depth_anything'
            if self._metric
            else 'video_depth_anything'
        )
        return checkpoints_dir / f'{checkpoint_name}_{self._encoder}.pth'

    def _load_model(self) -> VideoDepthAnything:
        model_configs = {
            'vits': {
                'encoder': 'vits',
                'features': 64,
                'out_channels': [48, 96, 192, 384],
            },
            'vitb': {
                'encoder': 'vitb',
                'features': 128,
                'out_channels': [96, 192, 384, 768],
            },
            'vitl': {
                'encoder': 'vitl',
                'features': 256,
                'out_channels': [256, 512, 1024, 1024],
            },
        }
        if self._encoder not in model_configs:
            raise ValueError(f'Unsupported encoder: {self._encoder}')

        checkpoint_path = (
            Path(self._checkpoint_path)
            if self._checkpoint_path
            else self._default_checkpoint_path()
        )
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f'Checkpoint not found: {checkpoint_path}'
            )

        model = VideoDepthAnything(**model_configs[self._encoder])
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(state_dict, strict=True)
        model = model.to(self._device).eval()
        self._reset_stream_state(model)
        return model

    def _reset_stream_state(self, model: VideoDepthAnything) -> None:
        for attr, value in (
            ('transform', None),
            ('frame_cache_list', []),
            ('frame_id_list', []),
            ('id', -1),
        ):
            if hasattr(model, attr):
                setattr(model, attr, value)
        if hasattr(model, 'frame_height'):
            model.frame_height = None
        if hasattr(model, 'frame_width'):
            model.frame_width = None

    def _log_startup(self) -> None:
        checkpoint_path = (
            self._checkpoint_path or str(self._default_checkpoint_path())
        )
        self.get_logger().info(
            'VDA streaming node started with '
            f'input_topic={self._input_topic}, '
            f'output_topic={self._output_topic}, '
            f'encoder={self._encoder}, '
            f'input_size={self._input_size}, '
            f'metric={self._metric}, '
            f'fp32={self._fp32}, '
            f'max_depth={self._max_depth}, '
            f'checkpoint_path={checkpoint_path}, '
            f'device={self._device}'
        )

    def _image_callback(self, msg: Image) -> None:
        with self._frame_cond:
            self._latest_msg = msg
            self._frame_cond.notify()

    def _process_loop(self) -> None:
        while self._running and rclpy.ok():
            with self._frame_cond:
                while (
                    self._latest_msg is None
                    and self._running
                    and rclpy.ok()
                ):
                    self._frame_cond.wait(timeout=0.1)
                if not self._running or not rclpy.ok():
                    return
                msg = self._latest_msg
                self._latest_msg = None
            if msg is not None:
                self._process_message(msg)

    def _process_message(self, msg: Image) -> None:
        try:
            frame = self._bridge.imgmsg_to_cv2(
                msg, desired_encoding='rgb8'
            )
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f'Failed to decode image: {exc}')
            return

        input_height, input_width = frame.shape[:2]
        crop = self._compute_crop(input_height, input_width)
        if crop is None:
            self.get_logger().error(
                'Invalid crop region; check crop coordinates.'
            )
            return
        x1, y1, x2, y2 = crop
        frame = frame[y1:y2, x1:x2]
        frame_height, frame_width = frame.shape[:2]

        if self._crop_last != crop:
            self.get_logger().info(
                f'Using crop box x1={x1}, y1={y1}, x2={x2}, y2={y2}.'
            )
            self._crop_last = crop

        if self._frame_height is None:
            self._frame_height = frame_height
            self._frame_width = frame_width
        elif (
            frame_height != self._frame_height
            or frame_width != self._frame_width
        ):
            self.get_logger().warning(
                'Crop size changed; resetting streaming cache.'
            )
            self._frame_height = frame_height
            self._frame_width = frame_width
            self._reset_stream_state(self._model)

        depth = self._model.infer_video_depth_one(
            frame,
            input_size=self._input_size,
            device=self._device,
            fp32=self._fp32,
        )
        if self._max_depth > 0.0:
            depth = np.clip(depth, 0.0, self._max_depth)
        depth = depth.astype(np.float32)

        try:
            depth_msg = self._bridge.cv2_to_imgmsg(
                depth, encoding='32FC1'
            )
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f'Failed to encode depth image: {exc}')
            return

        depth_msg.header = msg.header
        self._pub.publish(depth_msg)

        depth_vis = self._depth_to_colormap(depth)
        if depth_vis is None:
            return
        try:
            vis_msg = self._bridge.cv2_to_imgmsg(depth_vis, encoding='bgr8')
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f'Failed to encode depth vis: {exc}')
            return
        vis_msg.header = msg.header
        self._vis_pub.publish(vis_msg)

        restored_depth = self._restore_depth(
            depth, input_height, input_width, crop
        )
        warp_depth = self._warp_depth(restored_depth)
        if warp_depth is None:
            return
        try:
            warp_msg = self._bridge.cv2_to_imgmsg(
                warp_depth, encoding='32FC1'
            )
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f'Failed to encode depth warp: {exc}')
            return
        warp_msg.header = msg.header
        self._warp_pub.publish(warp_msg)

        warp_bin = self._adaptive_binarize(warp_depth)
        if warp_bin is None:
            return
        try:
            bin_msg = self._bridge.cv2_to_imgmsg(
                warp_bin, encoding='mono8'
            )
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f'Failed to encode depth warp bin: {exc}')
            return
        bin_msg.header = msg.header
        self._warp_bin_pub.publish(bin_msg)

    def _compute_crop(
        self, frame_height: int, frame_width: int
    ) -> tuple[int, int, int, int] | None:
        x1, y1, x2, y2 = self._crop_box
        x1 = max(0, min(x1, frame_width - 1))
        y1 = max(0, min(y1, frame_height - 1))
        x2 = max(x1 + 1, min(x2, frame_width))
        y2 = max(y1 + 1, min(y2, frame_height))
        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2, y2

    def _depth_to_colormap(self, depth: np.ndarray) -> np.ndarray | None:
        depth_min = float(np.nanmin(depth))
        depth_max = float(np.nanmax(depth))
        if not np.isfinite(depth_min) or not np.isfinite(depth_max):
            return None
        if depth_max <= depth_min:
            return np.zeros((*depth.shape, 3), dtype=np.uint8)
        depth_norm = (depth - depth_min) / (depth_max - depth_min)
        depth_u8 = np.clip(depth_norm * 255.0, 0.0, 255.0).astype(np.uint8)
        return cv2.applyColorMap(depth_u8, cv2.COLORMAP_INFERNO)

    def _restore_depth(
        self,
        depth: np.ndarray,
        input_height: int,
        input_width: int,
        crop: tuple[int, int, int, int],
    ) -> np.ndarray:
        restored = np.zeros((input_height, input_width), dtype=np.float32)
        x1, y1, x2, y2 = crop
        restored[y1:y2, x1:x2] = depth
        return restored

    def _warp_depth(self, depth: np.ndarray) -> np.ndarray | None:
        points = self._warp_points
        src = np.array(
            [
                points['top_left'],
                points['top_right'],
                points['bottom_right'],
                points['bottom_left'],
            ],
            dtype=np.float32,
        )
        width_top = np.linalg.norm(src[1] - src[0])
        width_bottom = np.linalg.norm(src[2] - src[3])
        height_left = np.linalg.norm(src[3] - src[0])
        height_right = np.linalg.norm(src[2] - src[1])
        width = max(int(round(width_top)), int(round(width_bottom)))
        height = max(int(round(height_left)), int(round(height_right)))
        if width <= 1 or height <= 1:
            self.get_logger().error('Warp size is invalid.')
            return None
        dst = np.array(
            [
                [0.0, 0.0],
                [width - 1.0, 0.0],
                [width - 1.0, height - 1.0],
                [0.0, height - 1.0],
            ],
            dtype=np.float32,
        )
        matrix = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(
            depth,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderValue=0,
        )

    def _adaptive_binarize(self, depth: np.ndarray) -> np.ndarray | None:
        valid_mask = depth > 0.0
        if not np.any(valid_mask):
            return np.zeros(depth.shape, dtype=np.uint8)
        valid_values = depth[valid_mask]
        depth_min = float(valid_values.min())
        depth_max = float(valid_values.max())
        if depth_max <= depth_min:
            return np.zeros(depth.shape, dtype=np.uint8)
        depth_norm = (depth - depth_min) / (depth_max - depth_min)
        depth_u8 = np.clip(depth_norm * 255.0, 0.0, 255.0).astype(np.uint8)
        depth_u8[~valid_mask] = 0
        return cv2.adaptiveThreshold(
            depth_u8,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2,
        )

    def shutdown(self) -> None:
        """Stop the worker thread gracefully."""
        self._running = False
        with self._frame_cond:
            self._frame_cond.notify_all()
        if self._worker.is_alive():
            self._worker.join(timeout=1.0)


def main() -> None:
    """Entry point for the VDA streaming node."""
    rclpy.init()
    node = StreamingDepthNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
