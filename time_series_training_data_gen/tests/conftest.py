import pytest
import pyarrow.parquet as pq
import os
from .data_for_test import full_series_1, full_series_2

file_name = 'test_series.parquet'


@pytest.fixture
def parquet_ts_file_1():
    pq.write_table(full_series_1, file_name)
    yield file_name
    os.remove(file_name)


@pytest.fixture
def parquet_ts_file_2():
    pq.write_table(full_series_2, file_name)
    yield file_name
    os.remove(file_name)
