class ModelConfig(object):
    def __init__(self):
        # File pattern of sharded TFRecord file containing SequenceExample protos.

        self.input_file_pattern = None
        self.image_format = "jpeg"

        # Approximate number of values per input shard. Used to ensure sufficient
        # mixing between shards in training.
        self.values_per_input_shard = 2300
        # Minimum number of shards to keep in the input queue.
        self.input_queue_capacity_factor = 2
        # Number of threads for prefetching SequenceExample protos.
        self.num_input_reader_threads = 1
        # Number of threads for image preprocessing. Should be a multiple of 2.
        self.num_preprocess_threads = 4

        self.vocab_size = 12000
        self.batch_size = 32

        self.inception_checkpoint_file = None

        self.image_height = 299
        self.image_width = 299

        self.minval = -0.08
        self.maxval = 0.08

        self.embedding_size = 512
        self.num_rnn_units = 512
        self.rnn_dropout_keep_prob = 0.7


class TrainingConfig(object):
    def __init__(self):
        self.num_examples_per_epoch = 586363

        self.optimizer = "SGD"

        self.initial_learning_rate = 2.0
        self.learning_rate_decay_factor = 0.5
        self.num_epochs_per_decay = 8.0

        self.train_inception_learning_rate = 0.0005

        self.clip_gradients = 5.0

        self.max_checkpoints_to_keep = 5
