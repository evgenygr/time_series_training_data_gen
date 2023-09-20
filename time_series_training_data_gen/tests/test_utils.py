import tensorflow as tf


def build_expected_results(rows_in_file, data_batch, time_window, offset):
    number_of_examples = rows_in_file - time_window - offset + 1
    number_of_full_batches = number_of_examples // data_batch

    expected_output_x = [
        tf.convert_to_tensor(
            [
                [
                    [j*data_batch + i + k, j*data_batch + i + k + 10]
                    for k in range(time_window)
                ] for i in range(data_batch)
            ],
            dtype=tf.float32
        ) for j in range(number_of_full_batches)
    ]
    expected_output_y = [tf.convert_to_tensor([offset] * data_batch, dtype=tf.float32)] * number_of_full_batches
    last_batch = number_of_examples % data_batch

    if last_batch:
        full_batch_offset = number_of_full_batches * data_batch
        expected_output_x.append(
            tf.convert_to_tensor(
                [
                    [
                        [full_batch_offset + i + k, full_batch_offset + i + k + 10]
                        for k in range(time_window)
                    ] for i in range(last_batch)
                ],
                dtype=tf.float32
            )
        )
        expected_output_y.append(tf.convert_to_tensor([offset] * last_batch, dtype=tf.float32))
    return expected_output_x, expected_output_y


def run_combined_asserts(data_gen, rows_in_file, data_batch, time_window, offset):
    expected_output_x, expected_output_y = build_expected_results(rows_in_file, data_batch, time_window, offset)
    for i, (expected_x, expected_y) in enumerate(zip(expected_output_x, expected_output_y)):
        rx, ry = next(data_gen)
        # try:
        tf.debugging.assert_equal(rx, expected_x)
        # except Exception:
        #     pass
        tf.debugging.assert_equal(ry, expected_y)
    rx, ry = next(data_gen)
    tf.debugging.assert_equal(rx, expected_output_x[0])
    tf.debugging.assert_equal(ry, expected_output_y[0])