class ModelConfig(object):
    def __init__(self):
        """Sets the default model hyperparameters."""
        # File pattern of sharded TFRecord file containing SequenceExample protos.
        # Must be provided in training and evaluation modes.
        self.input_file_pattern = None
        self.image_format = "jpeg"

        self.vocab_size = 9954
        self.batch_size = 10

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
        self.dataset_size = 55190
        self.num_examples_per_epoch = self.dataset_size * 5

        self.optimizer = "SGD"

        self.initial_learning_rate = 2.0
        self.learning_rate_decay_factor = 0.5
        self.num_epochs_per_decay = 8.0

        self.train_inception_learning_rate = 0.0005

        self.clip_gradients = 5.0

        self.max_checkpoints_to_keep = 5
