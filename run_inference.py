import math
import os

import  glob
import tensorflow as tf
import json

import configuration
import inference_wrapper
import img2txt
from inference_utils import caption_generator
from inference_utils import vocabulary

FLAGS = tf.flags.FLAGS

import os

tf.flags.DEFINE_string("test_captions_file", "captions_val2014.json",
                       "testing captions JSON file.")

tf.flags.DEFINE_string("checkpoint_path", "./model/train/",
                       "Model checkpoint file or directory containing a "
                       "model checkpoint file.")

tf.flags.DEFINE_string("vocab_file", "word_counts101.txt", "Text file containing the vocabulary.")
tf.flags.DEFINE_string("input_files", "./img",
                       "File pattern or comma-serated list of file patterns "
                       "of image files.")
tf.flags.DEFINE_string(flag_name="rnn_type", default_value="lstm",
                       docstring="RNN cell type lstm/gru .")
tf.flags.DEFINE_boolean("mode", True, "true for directory and false for file patterns")
tf.logging.set_verbosity(tf.logging.INFO)


def main(_):
  #build the inference graph
  g = tf.Graph()
  with g.as_default():
      model = inference_wrapper.InferenceWrapper(FLAGS.rnn_type)
      restore_fn = model.build_graph_from_config(configuration.ModelConfig(), FLAGS.checkpoint_path)
  g.finalize()

  # Create the vocabulary.
  vocab = vocabulary.Vocabulary(FLAGS.vocab_file)


  filenames = []

  path = FLAGS.input_files + "/*.jpg"
  for image in glob.glob(path):
    filenames.extend(tf.gfile.Glob(image))
  tf.logging.info("Running caption generation on %d files matching %s",
    len(filenames), FLAGS.input_files)

  with tf.Session(graph=g) as sess:
    # Load the model from checkpoint.
    restore_fn(sess)

    # Prepare the caption generator. Here we are implicitly using the default
    # beam search parameters. See caption_generator.py for a description of the
    # available beam search parameters.
    generator = caption_generator.CaptionGenerator(model, vocab)

    with tf.gfile.FastGFile(FLAGS.test_captions_file, "r") as f:
        caption_data = json.load(f)

    # Extract the filenames.
    id_to_filename = {x["file_name"]: x["id"] for x in caption_data["images"]}

    data = []

    with open('val_images.json') as f:
        val_data = json.load(f)
    val_image_names = val_data['images_name']

    for filename in filenames:
      with tf.gfile.GFile(filename, "rb") as f:
        image = f.read()
      captions = generator.beam_search(sess, image)
      print("Captions for image %s:" % os.path.basename(filename))
      if FLAGS.mode == True:
          img_name = os.path.basename(filename)
          if img_name not in val_image_names:
              continue
          img_id = id_to_filename[img_name]
          caption = captions[0]
          temp = [vocab.id_to_word(w) for w in caption.sentence[1:-1]]
          sentence = ''
          for word in temp:
              if sentence != '':
                  sentence += ' '
              for c in word:
                  sentence += c
          data.append({'image_id' : img_id, 'caption' : sentence})


      for i, caption in enumerate(captions):
        # Ignore begin and end words.
        sentence = [vocab.id_to_word(w) for w in caption.sentence[1:-1]]
        sentence = " ".join(sentence)
        print("  %d) %s (p=%f)" % (i, sentence, math.exp(caption.logprob)))

    with open('captions_val2014_im2txt_results.json', 'w') as outfile:
        json.dump(data, outfile)


if __name__ == "__main__":
  tf.app.run()
