# saving images

from xarm.wrapper import XArmAPI
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.cameras.realsense.camera_realsense import RealSenseCamera
from lerobot.cameras.configs import ColorMode, Cv2Rotation

# Create a `RealSenseCameraConfig` specifying your camera’s serial number and enabling depth.
config1 = RealSenseCameraConfig(
    serial_number_or_name="233522074606",
    fps=15,
    width=640,
    height=480,
    color_mode=ColorMode.RGB,
    use_depth=True,
    rotation=Cv2Rotation.NO_ROTATION
)

config2 = RealSenseCameraConfig(
    serial_number_or_name="233522074606",
    fps=15,
    width=640,
    height=480,
    color_mode=ColorMode.RGB,
    use_depth=True,
    rotation=Cv2Rotation.NO_ROTATION
)

# Instantiate and connect a `RealSenseCamera` with warm-up read (default).
camera1 = RealSenseCamera(config1)
camera1.connect()
camera2 = RealSenseCamera(config2)
camera2.connect()

# Capture a color frame via `read()` and a depth map via `read_depth()`.
try:
    color_frame = camera.read()
    depth_map = camera.read_depth()
    print("Color frame shape:", color_frame.shape)
    print("Depth map shape:", depth_map.shape)
finally:
    camera.disconnect()


# connecting to robot 

import os
from dataclasses import dataclass, field
from lerobot.cameras import CameraConfig
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.robots import RobotConfig
from lerobot.cameras import make_cameras_from_configs
from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus
from lerobot.robots import Robot

@RobotConfig.register_subclass("lerobot_robot_xarm")
@dataclass
class XarmConfig(RobotConfig):
    ip: str = "192.168.1.219"
    cameras: dict[str, CameraConfig] = field(
        default_factory={
            "cam_1": OpenCVCameraConfig(
                index_or_path=2,
                fps=30,
                width=480,
                height=640,
            ),
        }
    )
    ...

class XarmConfig(RobotConfig):
    ip: str = "192.168.1.184"
    use_effort: bool = False
    use_velocity: bool = True
    use_acceleration: bool = True
    home_translation: list[float] = field(default_factory=lambda: [0.2, 0.0, 0.05])
    home_orientation_euler: list[float] = field(default_factory=lambda: [3.14, 0.0, 0.0])
    cameras: dict[str, CameraConfig] = field(default_factory=dict)


class Xarm(Robot):
    config_class = XarmConfig
    name = "xarm"

    def __init__(self, config: XarmConfig):
        super().__init__(config)
        self.cameras = make_cameras_from_configs(config.cameras)

        this_dir = os.path.dirname(os.path.abspath(__file__))

        self.config = config
        self._is_connected = False
        self._arm = None
        self._gripper = None
    
        self._initial_pose = None
        self._prev_observation = None

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {
            "joint_1.pos": float,
            "joint_2.pos": float,
            "joint_3.pos": float,
            "joint_4.pos": float,
            "joint_5.pos": float,
            "joint_6.pos": float,
            "joint_7.pos": float,
        }
    
    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3)
            for cam in self.cameras
        }
    
    @property
    def observation_features(self) -> dict:
        features = {**self._motors_ft, **self._cameras_ft}
        if self.config.use_effort:
            for i in range(1, 7):
                features[f"joint{i}.effort"] = float
        if self.config.use_velocity:
            for i in range(1, 7):
                features[f"joint{i}.vel"] = float
        if self.config.use_acceleration:
            for i in range(1, 7):
                features[f"joint{i}.acc"] = float
        return features
    
    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected and all(cam.is_connected for cam in self.cameras.values())

    