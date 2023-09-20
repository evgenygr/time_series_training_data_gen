from setuptools import find_packages, setup

install_requires = [
  'numpy',
  'pandas',
  'pyarrow',
  'tensorflow',
  'pytest'
]

setup(
    name='time_series_training_data_gen',
    version='0.0.1',
    packages=find_packages(),
    install_requires=install_requires,
)
