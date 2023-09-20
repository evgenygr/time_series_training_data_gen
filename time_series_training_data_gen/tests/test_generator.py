from time_series_training_data_gen import FileBasedDataSource
from .test_utils import run_combined_asserts
from .data_for_test import (
    rows_in_file_1, time_window_1, offset_1, data_batch_1,
    rows_in_file_2, time_window_2, offset_2, data_batch_2, file_batch_2
)


def test_generator_1(parquet_ts_file_1):
    data_gen = iter(FileBasedDataSource(
        filename=parquet_ts_file_1,
        time_window=time_window_1,
        offset=offset_1,
        prediction_cname='b',
        batch_size=data_batch_1,
    ))
    run_combined_asserts(data_gen, rows_in_file_1, data_batch_1, time_window_1, offset_1)


def test_generator_2(parquet_ts_file_2):
    data_gen = FileBasedDataSource(
        filename=parquet_ts_file_2,
        time_window=time_window_2,
        offset=offset_2,
        prediction_cname='b',
        batch_size=data_batch_2,
    )
    data_gen._file_batch_size = file_batch_2
    data_gen = iter(data_gen)
    run_combined_asserts(data_gen, rows_in_file_2, data_batch_2, time_window_2, offset_2)
