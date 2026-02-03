import os
from glob import glob

from setuptools import setup

package_name = 'vda_streaming'

package_data_files = []
for path in glob(os.path.join(package_name, 'libs', '**', '*.py'), recursive=True):
    package_data_files.append(os.path.relpath(path, package_name))

checkpoint_files = glob(os.path.join('checkpoints', '*.pth'))

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    package_data={package_name: package_data_files},
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/streaming.launch.py']),
        (
            'share/' + package_name + '/checkpoints',
            checkpoint_files,
        ),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='junho',
    maintainer_email='snowpoet@naver.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'vda_streaming_node = vda_streaming.vda_streaming_node:main'
        ],
    },
)
