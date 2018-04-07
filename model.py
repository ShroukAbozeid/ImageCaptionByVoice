import tensorflow as tf


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

