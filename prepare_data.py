import tensorflow as tf


def read_data(reader,file_pattern,is_training,batch_size):
    """Prefetches string values from disk into an input queue.

      In training the capacity of the queue is important because a larger queue
      means better mixing of training examples between shards. The minimum number of
      values kept in the queue is values_per_shard * input_queue_capacity_factor,
      where input_queue_memory factor should be chosen to trade-off better mixing
      with memory usage.

      Args:
        reader: Instance of tf.ReaderBase.
        file_pattern: Comma-separated list of file patterns (e.g.
            /tmp/train_data-?????-of-00100).
        is_training: Boolean; whether prefetching for training or eval.
        batch_size: Model batch size used to determine queue capacity.
        values_per_shard=2300
        num_reader_threads=1
        shard_queue_name ="filename_queue"
        value_queue_name="input_queue"
        input_queue_capacity_factor = 2
    Returns:
        A Queue containing prefetched string values.
    """

    data_files = []
    for pattern in file_pattern.split(","):
        data_files.extend(tf.gfile.Glob(pattern))

    values_per_shard = 2300
    num_reader_threads = 1
    shard_queue_name = "filename_queue"
    value_queue_name = "input_queue"
    input_queue_capacity_factor = 2

    # create queues to hold file names and values --- why those num and why diff in train and test
    if is_training:
        filename_queue = tf.train.string_input_producer(
            data_files, shuffle=True, capacity=16, name=shard_queue_name)
        min_queue_examples = values_per_shard * input_queue_capacity_factor
        capacity = min_queue_examples + 100 * batch_size
        values_queue = tf.RandomShuffleQueue(
            capacity=capacity,
            min_after_dequeue=min_queue_examples,
            dtypes=[tf.string],
            name="random_" + value_queue_name)
    else:
        filename_queue = tf.train.string_input_producer(
            data_files, shuffle=False, capacity=1, name=shard_queue_name)
        capacity = values_per_shard + 3 * batch_size
        values_queue = tf.FIFOQueue(
            capacity=capacity, dtypes=[tf.string], name="fifo_" + value_queue_name)

    # read and add to queue
    enqueue_ops = []
    for _ in range(num_reader_threads):
        _, value = reader.read(filename_queue)
        enqueue_ops.append(values_queue.enqueue([value]))
    tf.train.queue_runner.add_queue_runner(tf.train.queue_runner.QueueRunner(
        values_queue, enqueue_ops))

    return values_queue
