# Training Data Generator for fitting time-serires-related Keras models

One of possible ways to feed data to .fit method of Keras model is to use 
[generator](https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit) of batches. 
Generator might be handy in case of really huge amount of data - when loading them to memory
may cause python kernel death. In this case data is usually provided as a file. This generator
is tailored to work with parquet files. It reads data batch by batch, reformatting file batches
to data batches. This allows to store in memory only that part of data that is necessary to 
build next data batch.  
## Usage
The way to use training data generator can be checked out at `tests/test_generator.py`.
## Installation
To install it as a package: `python setup.py develop` ('develop' is a mode that allows to modify
code of generator such that it could be reflected in the functionality of installed package)
## Run tests
The simplest way to run tests is to install the package, `cd` to a root directory of a package 
(`time_series_training_data_gen`) and run `pytest`.
