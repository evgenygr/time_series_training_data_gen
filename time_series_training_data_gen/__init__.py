from itertools import product
from typing import Optional
import tensorflow as tf
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc


def rolling_window_max(ser: pa.Array, window: int):
    return sliding_window_view(
        ser.to_pandas().values,
        window_shape=window
    ).max(axis=1)


class FileBasedDataSource:
    def __init__(
        self,
        filename: str,
        time_window: int,
        offset: int,
        prediction_cname: str,
        condition_table: callable = lambda x, t: x,
        batch_size: int = 1,
        nrows: Optional[int] = None,
        time_index_name: Optional[str] = None
    ):
        """A class that can be used to feed data to .fit method of tensorflow.keras model. The model is
        supposed to be a time series predictive model.
        For some reason keras doesn't re-initialize data generator after the end of epoch, so we cannot rely on
        automatic inference of epoch length at the even of rising StopIteration. So, this generator never rises
        StopIteration, and just "rewinds" file to the beginning at the end of epoch.

        Attributes:
            filename (str): full path to a parquet file storing training time series.

            time_window (int): number of time points in 1 training example

            offset (int): number of 'future' time points after features-related time window
                that are used to calculate training predictions

            prediction_cname (str): a column name (of conditioned table) that stores the value that model is trying
                to predict

            condition_table: a function that should condition a table with original format, making in easily
                interpretable by ML model.

            batch_size (int): learning examples are bundled into batches.
                This argument is the number of examples in a batch.

            nrows (int): number of rows to be processed. If None, then entire all rows of a file will be processed.

            time_index_name: (str): the name of a column that will store time index (if any). Usually, when one
                saves pandas dataframe with timestamp index then pyarrow reads this index into a column named
                '__index_level_0__'
        """
        self.filename = filename
        file = pq.ParquetFile(filename)
        self.full_nrows = file.scan_contents()
        self.nrows = self.full_nrows if not nrows else nrows
        self._file_batch_size = 65536  # This 'magic number' is the default batch_size of
        # pyarrow.parquet.ParquetFile.iter_batches. The value of this file doesn't influence
        # the data being generated (otherwise it's a bug), but very convenient for unit-testing.
        self.iterator = file.iter_batches(batch_size=self._file_batch_size)
        self.condition_table = condition_table
        self.prediction_cname = prediction_cname
        self.tw = time_window
        self.offset = offset
        self.batch_size = batch_size
        self.time_index_name = time_index_name

    def __iter__(self):
        file = pq.ParquetFile(self.filename)
        self.fetched_rows = 0
        self.iterator = file.iter_batches(batch_size=self._file_batch_size)
        self.fetched_rows = 0
        self.last_timestamp = None
        stored_batches = []
        while self.fetched_rows < self.full_nrows:
            latest_table = self._get_next_file_batch()
            stored_batches.append(latest_table)
            self.fetched_rows += latest_table.shape[0]
            if self.fetched_rows >= self.tw + self.offset + self.batch_size - 1:
                break
        else:
            raise StopIteration('file is too short')
        self.atb = pa.concat_tables(stored_batches)
        self.x = self.tw + self.offset - 2  # "Last processed index"
        self.glob_x = self.x
        self.feature_cols = self.atb.column_names[:-1]
        return self

    def _get_next_file_batch(self):
        atb = pa.Table.from_batches(batches=[next(self.iterator)])
        new_last_timestamp = atb[self.time_index_name][-1].as_py() if self.time_index_name else None
        atb = self.condition_table(atb, self.last_timestamp)
        self.last_timestamp = new_last_timestamp
        return atb

    def _move_2_next_file_batch(self):
        uncovered_head = self.atb.shape[0] - self.x - 1
        tail_len = self.tw + self.offset + uncovered_head - 1
        head_len = 0
        stored_batches = []
        while self.fetched_rows < self.nrows:
            latest_table = self._get_next_file_batch()
            stored_batches.append(latest_table)
            head_len += latest_table.shape[0]
            self.fetched_rows += latest_table.shape[0]
            if tail_len + head_len >= self.tw + self.offset + self.batch_size - 1:
                break
        else:
            if not stored_batches:
                raise StopIteration('file is too short')
        self.atb = pa.concat_tables(
            [self.atb.take(list(range(self.atb.shape[0] - tail_len, self.atb.shape[0])))] +
            stored_batches
        )
        self.x = self.tw + self.offset - 2  # The same formula as in __init__

    def build_batch(self, atb: pa.Table, top_index, batch_size, y_offset, features_window):
        return (
            tf.transpose(
                tf.convert_to_tensor(sliding_window_view(
                    atb.slice(
                        offset=top_index - batch_size + 1 - y_offset - features_window + 1,
                        length=features_window + batch_size - 1
                    ).to_pandas().values,
                    window_shape=features_window,
                    axis=0
                )),
                perm=[0, 2, 1]
            ),
            tf.convert_to_tensor(
                pc.subtract(
                    rolling_window_max(
                        atb[self.prediction_cname].slice(
                            offset=top_index - batch_size - y_offset + 2,
                            length=batch_size + y_offset - 1
                        ),
                        window=y_offset
                    ),
                    atb[self.prediction_cname].slice(
                        offset=top_index - batch_size + 1 - y_offset,
                        length=batch_size
                    )
                )
            )
        )

    def __next__(self):
        if self.glob_x >= self.nrows - 1:
            self.__iter__()  # Normally here we would rise StopIteration, but the way keras works
            # requires us to rewind the generator
        if self.x + self.batch_size < self.atb.shape[0]:
            this_batch_size = self.batch_size
        else:
            if self.fetched_rows < self.nrows:
                self._move_2_next_file_batch()
            if self.glob_x + self.batch_size < self.nrows:
                this_batch_size = self.batch_size
            else:
                this_batch_size = self.nrows - self.x - 1
        self.x += this_batch_size
        self.glob_x += this_batch_size
        return self.build_batch(
            atb=self.atb,
            top_index=self.x,
            batch_size=this_batch_size,
            y_offset=self.offset,
            features_window=self.tw,
        )
