from setuptools import setup
from glob import glob

package_name = 'rm_65_task'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
        ('share/' + package_name + '/scripts', glob('scripts/*.sh')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='student',
    maintainer_email='student@todo.todo',
    description='Gesture-driven pick and place task manager for RM65.',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'task_manager = rm_65_task.task_manager:main',
            'teleop_controller = rm_65_task.teleop_controller:main',
        ],
    },
)
