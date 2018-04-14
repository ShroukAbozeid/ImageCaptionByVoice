import math
import os


import tensorflow as tf

import configuration
import inference_wrapper
import img2txt
from inference_utils import caption_generator
from inference_utils import vocabulary

FLAGS = tf.flags.FLAGS

import os

tf.flags.DEFINE_string("checkpoint_path", "./model/train/",
                       "Model checkpoint file or directory containing a "
                       "model checkpoint file.")
tf.flags.DEFINE_string("vocab_file", "/media/higazy/New Volume/GP/ImageCaptionByVoice/word_counts.txt", "Text file containing the vocabulary.")
tf.flags.DEFINE_string("input_files","/media/higazy/New Volume/im2txt/24f34s.jpg"
,
                       "File pattern or comma-serated list of file patterns "
                       "of image files.")
tf.flags.DEFINE_string(flag_name="rnn_type", default_value= "lstm",
                       docstring="RNN cell type lstm/gru .")

tf.logging.set_verbosity(tf.logging.INFO)

def main(_):
  #build the inference graph
  g = tf.Graph()
  with g.as_default():
      model = inference_wrapper.InferenceWrapper(FLAGS.rnn_type)
      restore_fn = model.build_graph_from_config(configuration.ModelConfig(),
                                                 FLAGS.checkpoint_path)
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