import math
import os


import tensorflow as tf

import configuration
import img2txt
from inference_utils import caption_generator
from inference_utils import vocabulary

FLAGS = tf.flags.FLAGS

import os

tf.flags.DEFINE_string("checkpoint_path", "./model/train/model2.ckpt-2000000",
                       "Model checkpoint file or directory containing a "
                       "model checkpoint file.")
tf.flags.DEFINE_string("vocab_file", "./data/mscoco/word_counts.txt", "Text file containing the vocabulary.")
tf.flags.DEFINE_string("input_files","./data/mscoco/raw-data/test2014/COCO_test2014_000000000001.jpg,\
./data/mscoco/raw-data/test2014/COCO_test2014_000000000275.jpg,\
./data/mscoco/raw-data/test2014/COCO_test2014_000000000457.jpg,\
./data/mscoco/raw-data/test2014/COCO_test2014_000000000463.jpg,\
./data/mscoco/raw-data/test2014/COCO_test2014_000000001127.jpg,\
./data/mscoco/raw-data/test2014/COCO_test2014_000000002358.jpg,\
./data/mscoco/raw-data/test2014/COCO_test2014_000000002787.jpg,\
./data/mscoco/raw-data/test2014/COCO_test2014_000000002547.jpg,\
./data/mscoco/raw-data/test2014/COCO_test2014_000000002490.jpg,\
./data/mscoco/raw-data/test2014/COCO_test2014_000000007245.jpg"
,
                       "File pattern or comma-serated list of file patterns "
                       "of image files.")
tf.flags.DEFINE_string(flag_name="rnn_type", default_value= "lstm",
                       docstring="RNN cell type lstm/gru .")

tf.logging.set_verbosity(tf.logging.INFO)

def build_graph_from_config(saver):
    model_config = configuration.ModelConfig()
    checkpoint_path = FLAGS.checkpoint_path
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

def main(_):
  #build the inference graph
  g = tf.Graph()
  with g.as_default():
      model = img2txt.Model(configuration.ModelConfig(), mode="inference",
                            rnn_type=FLAGS.rnn_type,
                            train_inception=False)
      tf.logging.info("Building model.")
      model.build()
      saver = tf.train.Saver()
      restore_fn = build_graph_from_config(saver)
  g.finalize()

  # Create the vocabulary.
  vocab = vocabulary.Vocabulary(FLAGS.vocab_file)

  filenames = []
  for file_pattern in FLAGS.input_files.split(","):
    filenames.extend(tf.gfile.Glob(file_pattern))
  tf.logging.info("Running caption generation on %d files matching %s",
                  len(filenames), FLAGS.input_files)

  with tf.Session(graph=g) as sess:
    # Load the model from checkpoint.
    restore_fn(sess)

    # Prepare the caption generator. Here we are implicitly using the default
    # beam search parameters. See caption_generator.py for a description of the
    # available beam search parameters.
    generator = caption_generator.CaptionGenerator(model, vocab)

    for filename in filenames:
      with tf.gfile.GFile(filename, "rb") as f:
        image = f.read()
      captions = generator.beam_search(sess, image)
      print("Captions for image %s:" % os.path.basename(filename))
      for i, caption in enumerate(captions):
        # Ignore begin and end words.
        sentence = [vocab.id_to_word(w) for w in caption.sentence[1:-1]]
        sentence = " ".join(sentence)
        print("  %d) %s (p=%f)" % (i, sentence, math.exp(caption.logprob)))


if __name__ == "__main__":
  tf.app.run()