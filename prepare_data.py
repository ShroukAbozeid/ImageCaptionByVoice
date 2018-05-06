import tensorflow as tf


def read_data(reader, file_pattern, is_training, batch_size,
              values_per_shard, input_queue_capacity_factor=16,
              num_reader_threads=1, shard_queue_name="filename_queue",
              value_queue_name="input_queue"):
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

    # create queues to hold file names and values --- why those num for capacity?
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


def parse_sequence_example(example):
    """Parses a tensorflow.SequenceExample into an image and caption.

      Args:
        example: A scalar string Tensor; a single serialized SequenceExample.

      Returns:
        encoded_image: A scalar string Tensor containing a JPEG encoded image.
        caption: A 1-D uint64 Tensor with dynamically specified length.
      """
    context, seq = tf.parse_single_sequence_example(
        example,
        context_features={
            "image/data": tf.FixedLenFeature([], dtype=tf.string)
        },
        sequence_features={
            "image/caption_ids": tf.FixedLenSequenceFeature([], dtype=tf.int64),
        })
    encoded_image = context["image/data"]
    caption = seq["image/caption_ids"]
    return encoded_image, caption


def prepare_batch(data, batch_size, queue_capacity):
    """Batches input images and captions.

      This function splits the caption into an input sequence and a target sequence,
      where the target sequence is the input sequence right-shifted by 1. Input and
      target sequences are batched and padded up to the maximum length of sequences
      in the batch. A mask is created to distinguish real words from padding words.

      Args:
        data: A list of pairs [image, caption], where image is a
          Tensor of shape [height, width, channels] and caption is a 1-D Tensor of
          any length. Each pair will be processed and added to the queue in a
          separate thread.
        batch_size: Batch size.
        queue_capacity: Queue capacity.

      Returns:
        images: A Tensor of shape [batch_size, height, width, channels].
        input_seqs: An int32 Tensor of shape [batch_size, padded_length].
        target_seqs: An int32 Tensor of shape [batch_size, padded_length].
        mask: An int32 0/1 Tensor of shape [batch_size, padded_length].
      """
    batch_list = []
    for image, caption in data:
        cap_len = tf.shape(caption)[0]
        input_len = tf.expand_dims(tf.subtract(cap_len, 1), 0)

        input_seq = tf.slice(caption, [0], input_len)
        target_seq = tf.slice(caption, [1], input_len)
        mask = tf.ones(input_len, dtype=tf.int32)
        batch_list.append([image, input_seq, target_seq, mask])

    images, input_seqs, target_seqs, masks = tf.train.batch_join(
        batch_list,
        batch_size=batch_size,
        capacity=queue_capacity,
        dynamic_pad=True,
        name="batch_and_pad"
    )

    #[Todo] summary

    return images, input_seqs, target_seqs, masks
