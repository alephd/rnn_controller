try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(name='rnn_controller',
      version='0.0.1',
      description='Rnn controller module',
      author='at, np',
      packages=['rnn_controller'],
      install_requires=[
          'numpy<=1.14.5,>=1.13.3',
          'matplotlib',
          'pandas',
          'boto3',
          'tensorflow == 1.12',
          'tensorflow-probability',
          'imageio'],
      python_requires='>=3.6',
      )
