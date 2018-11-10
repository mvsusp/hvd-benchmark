from setuptools import setup

setup(
    name='sm_openmpi',
    version='0.1',
    py_modules=['sm_openmpi'],
    install_requires=['Click', 'sagemaker_containers', 'retry'],
    entry_points='''
        [console_scripts]
        sm-openmpi=sm_openmpi:cli
    ''',
)
