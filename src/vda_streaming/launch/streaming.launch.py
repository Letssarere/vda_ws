"""Launch VDA streaming node with RealSense camera."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    input_topic = LaunchConfiguration('input_topic')
    output_topic = LaunchConfiguration('output_topic')
    encoder = LaunchConfiguration('encoder')
    input_size = LaunchConfiguration('input_size')
    metric = LaunchConfiguration('metric')
    fp32 = LaunchConfiguration('fp32')
    max_depth = LaunchConfiguration('max_depth')
    checkpoint_path = LaunchConfiguration('checkpoint_path')

    vda_node = Node(
        package='vda_streaming',
        executable='vda_streaming_node',
        name='vda_streaming_node',
        output='screen',
        parameters=[
            {
                'input_topic': input_topic,
                'output_topic': output_topic,
                'encoder': encoder,
                'input_size': ParameterValue(input_size, value_type=int),
                'metric': ParameterValue(metric, value_type=bool),
                'fp32': ParameterValue(fp32, value_type=bool),
                'max_depth': ParameterValue(max_depth, value_type=float),
                'checkpoint_path': checkpoint_path,
            }
        ],
    )

    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare('realsense2_camera'),
                    'launch',
                    'rs_launch.py',
                ]
            )
        ),
        launch_arguments={
            'enable_color': 'true',
            'enable_depth': 'false',
            'enable_infra1': 'false',
            'enable_infra2': 'false',
            'enable_gyro': 'false',
            'enable_accel': 'false',
        }.items(),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                'input_topic',
                default_value='/camera/color/image_raw',
                description='Input RGB topic for VDA.',
            ),
            DeclareLaunchArgument(
                'output_topic',
                default_value='/vda/depth_image',
                description='Output depth topic.',
            ),
            DeclareLaunchArgument(
                'encoder',
                default_value='vits',
                description='Encoder backbone: vits, vitb, vitl.',
            ),
            DeclareLaunchArgument(
                'input_size',
                default_value='518',
                description='Input size for depth model.',
            ),
            DeclareLaunchArgument(
                'metric',
                default_value='true',
                description='Use metric depth model.',
            ),
            DeclareLaunchArgument(
                'fp32',
                default_value='false',
                description='Force FP32 inference.',
            ),
            DeclareLaunchArgument(
                'max_depth',
                default_value='20.0',
                description='Clamp max depth (<=0 disables).',
            ),
            DeclareLaunchArgument(
                'checkpoint_path',
                default_value='',
                description='Optional override for checkpoint path.',
            ),
            realsense_launch,
            vda_node,
        ]
    )
