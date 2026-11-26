import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'controller_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        # Trained clue character classifier (train_clue_reader.py)
        (os.path.join('share', package_name, 'models'), glob('controller_pkg/models/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='fizzer',
    maintainer_email='lucatrinca37@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'time_trial_controller = controller_pkg.time_trial_controller:main',
            'drone_teleop = controller_pkg.drone_teleop:main',
            'drone_time_trial = controller_pkg.drone_time_trial:main',
            'drone_road_follower = controller_pkg.drone_road_follower:main',
            'drone_position_hold = controller_pkg.drone_position_hold:main',
            'drone_sign_tour = controller_pkg.drone_sign_tour:main',
            'drone_wall_ride = controller_pkg.drone_wall_ride:main',
            'drone_west_ride = controller_pkg.drone_west_ride:main',
            'clue_finder = controller_pkg.clue_finder:main',
            'train_clue_reader = controller_pkg.train_clue_reader:main',
        ],
    },
)
