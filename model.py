import tensorflow as tf
from tensorflow.contrib.slim.python.slim.nets.inception_v3 import inception_v3_base
slim = tf.contrib.slim

import prepare_data


class model(object):
    """
        inception - lstm/gru model
    """
    def __init__(self, config, mode, rnn_type, train_inception):
        """
        Basic setup
        :param config: configuration for model
        :param mode: train, eval, test
        :param rnn_type: lstm ,gru
        :param train_inception: true, false
        """
        self.config = config
        self.train_inception = train_inception
        self.mode = mode
        self.rnn_type = rnn_type

        ## Reader to read tf records
        # change if reading images direct
        self.reader = tf.TFRecordReader()


        # initalizer
        self.initializer = tf.random_uniform_initializer(
            minval=self.config.minval,
            maxval=self.config.maxval
        )

        # A float32 Tensor with shape [batch_size, height, width, channels].
        self.images = None

        # An int32 Tensor with shape [batch_size, padded_length].
        self.input_seqs = None

        # An int32 Tensor with shape [batch_size, padded_length].
        self.target_seqs = None

        # An int32 0/1 Tensor with shape [batch_size, padded_length].
        self.input_mask = None

        # A float32 Tensor with shape [batch_size, embedding_size].
        self.image_embeddings = None

        # A float32 Tensor with shape [batch_size, padded_length, embedding_size].
        self.seq_embeddings = None

        # A float32 scalar Tensor; the total loss for the trainer to optimize.
        self.total_loss = None

        # A float32 Tensor with shape [batch_size * padded_length].
        self.target_cross_entropy_losses = None

        # A float32 Tensor with shape [batch_size * padded_length].
        self.target_cross_entropy_loss_weights = None

        # Collection of variables from the inception submodel.
        self.inception_variables = []

        # Function to restore the inception submodel from checkpoint.
        self.init_fn = None

        # Global step Tensor.
        self.global_step = None

    def process_image(self, encoded_image):
        """
        Decode an image, resize and apply random distortions.
        Args:
        encoded_image: String Tensor containing the image.
        Returns:
        A float32 Tensor of shape [height, width, 3] with values in [-1, 1].
        """
        #Todo [summary]
        # threads??

        # Decode to [ height, width, 3]
        with tf.name_scope("decode", values=[encoded_image]):
            if self.config.image_format == "jpeg":
                image = tf.image.decode_jpeg(encoded_image, channels=3)
            elif self.config.image_format == 'png':
                image = tf.image.decode_png(encoded_image, channels=3)

        # float
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)

        # resize ---------- why ???
        resize_height = 346
        resize_width = 346
        image = tf.image.resize_images(image,size=[resize_height, resize_width],
                                      method=tf.image.ResizeMethod.BILINEAR)

        # Crop to final dimensions.--------- why diff methods?
        if self.mode == "train":
            image = tf.random_crop(image, [self.config.height, self.config.width, 3])
        else:
            # Central crop, assuming resize_height > height, resize_width > width.
            image = tf.image.resize_image_with_crop_or_pad(image, self.config.height, self.config.width)

        # distort
        if self.mode == "train":
            with tf.name_scope("distort_color", values=[image]):
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.032)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)

            image = tf.clip_by_value(image, 0.0, 1.0)

        # Rescale to [-1,1] instead of [0, 1] -------- why?
        image = tf.subtract(image, 0.5)
        image = tf.multiply(image, 2.0)
        return image

    def build_image_embeddings(self):
        trainable = self.mode == "train"

        is_inception_model_training = \
            self.train_inception and trainable

        # Default parameters for batch normalization.
        batch_norm_params = {
            "is_training": is_inception_model_training,
            "trainable": trainable,
            # Decay for the moving averages.
            "decay": 0.9997,
            # Epsilon to prevent 0s in variance.
            "epsilon": 0.001,
            # Collection containing the moving mean and moving variance.
            "variables_collections": {
                "beta": None,
                "gamma": None,
                "moving_mean": ["moving_vars"],
                "moving_variance": ["moving_vars"],
            }
        }

        if trainable:
            weights_regularizer = tf.contrib.layers.l2_regularizer(0.00004)
        else:
            weights_regularizer = None

        with tf.variable_scope("InceptionV3", [self.images]) as scope:
            with slim.arg_scope(
                    [slim.conv2d, slim.fully_connected],
                    weights_regularizer=weights_regularizer,
                    trainable=trainable
            ):
                with slim.arg_scope(
                        [slim.conv2d],
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                        activation_fn=tf.nn.relu,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params
                ):
                    net, end_points = inception_v3_base(self.images, scope=scope)
                    with tf.variable_scope("logits"):
                        shape = net.get_shape()
                        net = slim.avg_pool2d(net, shape[1:3], padding="VALID", scope="pool")
                        net = slim.dropout(
                            net,
                            keep_prob=0.8,
                            is_training=is_inception_model_training,
                            scope="dropout"
                        )
                        net = slim.flatten(net, scope="flatten")

        # Add summaries (????)
        for v in end_points.values():
            tf.contrib.layers.summaries.summarize_activation(v)

        # ???
        self.inception_variables = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope="InceptionV3")

        # map inception output into embedding space
        with tf.variable_scope("image_embedding") as scope:
            image_embeddings = tf.contrib.layers.fully_connected(
            inputs=net,
            num_outputs=self.config.embedding_size,
            activation_fn=None,
            weights_initializer=self.initializer,
            biases_initializer=None,
            scope=scope)

        # Save the embedding size in the graph.
        tf.constant(self.config.embedding_size, name="embedding_size")

        self.image_embeddings = image_embeddings

    def build_inputs(self):
        """Input prefetching, preprocessing and batching.

           Outputs:
             self.images
             self.input_seqs
             self.target_seqs (training and eval only)
             self.input_mask (training and eval only)
        """
        if self.mode == "inference":
            image_feed = tf.placeholder(dtype=tf.string, shape=[], name="image_feed")
            input_feed = tf.placeholder(dtype=tf.int64,
                                        shape=[None],  # batch_size
                                        name="input_feed")

            # Process image and insert batch dimensions.
            image = self.process_image(image_feed)
            images = tf.expand_dims(image, 0)

            # No target sequences or input mask in inference mode.
            input_seqs = tf.expand_dims(input_feed, 1)
            target_seqs = None
            input_mask = None
        else:
            queue = prepare_data.read_data(
                self.reader,
                self.config.input_file_pattern,
                True,
                self.config.batch_size)
            data = []
            seq_example = queue.dequeue()
            encoded_img, caption = prepare_data.parse_sequence_example(seq_example)
            image = self.process_image(encoded_img)
            data.append([image, caption])

            queue_capacity = 2 * self.config.batch_size

            images, input_seqs, target_seqs, input_mask = (
                prepare_data.prepare_batch(data=data,
                                           batch_size=self.config.batch_size,
                                           queue_capacity=queue_capacity))
        self.images = images
        self.input_seqs = input_seqs
        self.target_seqs = target_seqs
        self.input_mask = input_mask
