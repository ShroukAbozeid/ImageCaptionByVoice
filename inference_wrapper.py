import img2txt
import os.path
import tensorflow as tf

class InferenceWrapper(object):
    def __init__(self, rnn_type):
        self.rnn_type = rnn_type

    def build_model(self, model_config):
        model = img2txt.Model(config=model_config,
                              mode= "inference",
                              rnn_type=self.rnn_type,
                              train_inception=False)
        model.build()
        return model

    def feed_image(self, sess, encoded_image):
        initial_state = sess.run(fetches="lstm/initial_state:0",
                                 feed_dict={"image_feed:0": encoded_image})
        return initial_state

    def inference_step(self, sess, input_feed, state_feed):
        softmax_output, state_output = sess.run(
            fetches=["softmax:0", "lstm/state:0"],
            feed_dict={
                "input_feed:0": input_feed,
                "lstm/state_feed:0": state_feed,
            })
        return softmax_output, state_output, None

    def _create_restore_fn(self, checkpoint_path, saver):
        """Creates a function that restores a model from checkpoint.

            Args:
              checkpoint_path: Checkpoint file or a directory containing a checkpoint
                file.
              saver: Saver for restoring variables from the checkpoint file.

            Returns:
              restore_fn: A function such that restore_fn(sess) loads model variables
                from the checkpoint file.

            Raises:
              ValueError: If checkpoint_path does not refer to a checkpoint file or a
                directory containing a checkpoint file.
        """
        if tf.gfile.IsDirectory(checkpoint_path):
            checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)
            if not checkpoint_path:
                raise ValueError("No checkpoint file found in: %s" % checkpoint_path)

        def _restore_fn(sess):
            tf.logging.info("Loading model from checkpoint: %s", checkpoint_path)
            saver.restore(sess, checkpoint_path)
            tf.logging.info("Successfully loaded checkpoint: %s",
                            os.path.basename(checkpoint_path))

        return _restore_fn

    def build_graph_from_config(self, model_config, checkpoint_path):
        """Builds the inference graph from a configuration object.

        Args:
          model_config: Object containing configuration for building the model.
          checkpoint_path: Checkpoint file or a directory containing a checkpoint
            file.

        Returns:
          restore_fn: A function such that restore_fn(sess) loads model variables
            from the checkpoint file.
        """
        tf.logging.info("Building model.")
        self.build_model(model_config)
        saver = tf.train.Saver()

        return self._create_restore_fn(checkpoint_path, saver)
