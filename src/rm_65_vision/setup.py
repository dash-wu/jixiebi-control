from setuptools import setup
import os
from glob import glob

package_name = 'rm_65_vision'

setup(
    name=package_name,
    version='2.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
        ('share/' + package_name + '/models', glob('models/*.task')),
    ],
    scripts=['scripts/debug_arm_recognition.py'],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='student',
    maintainer_email='student@todo.todo',
    description='Gesture recognition and arm-camera grasp alignment for RM65.',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'gesture_recognizer = rm_65_vision.gesture_recognizer:main',
            'grasp_aligner = rm_65_vision.grasp_aligner:main',
            'arm_teleop_tracker = rm_65_vision.arm_teleop_tracker:main',
        ],
    },
)
