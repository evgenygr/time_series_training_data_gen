import pyarrow as pa

rows_in_file_1 = 50
full_series_1 = pa.table(
    [
        pa.array(range(rows_in_file_1), type=pa.int8()),
        pa.array(range(10, 10 + rows_in_file_1), type=pa.float32())
    ],
    names=['a', 'b']
)
file_batch_1 = None
data_batch_1 = 5
time_window_1 = 9
offset_1 = 7


rows_in_file_2 = 67
full_series_2 = pa.table(
    [
        pa.array(range(rows_in_file_2), type=pa.int8()),
        pa.array(range(10, 10 + rows_in_file_2), type=pa.float32())
    ],
    names=['a', 'b']
)
file_batch_2 = 15
data_batch_2 = 8
time_window_2 = 11
offset_2 = 9
