from tkinter import *
import tkinter, tkinter.constants, tkinter.filedialog
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

import math
import os
import tensorflow as tf
import configuration

import inference_wrapper
import texttospeech
from inference_utils import caption_generator
from inference_utils import vocabulary

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("checkpoint_path", "./model/train/",
            "Model checkpoint file or directory containing a "
            "model checkpoint file.")
tf.flags.DEFINE_string("vocab_file", "./data/word_counts.txt", "Text file containing the vocabulary.")
tf.flags.DEFINE_string(flag_name="rnn_type", default_value= "lstm",
            docstring="RNN cell type lstm/gru .")

tf.logging.set_verbosity(tf.logging.INFO)

root = Tk()
root.title("Image Captioning")

# Add a grid
mainframe = Frame(root)
mainframe.grid(column=0,row=0, sticky=(N,W,E,S))
mainframe.columnconfigure(0, weight = 1)
mainframe.rowconfigure(0, weight = 1)
mainframe.pack(pady = 50, padx = 100)

browse_button = Button(mainframe, text="Browse")
gen_button = Button(mainframe, text="Show Caption")
voice_button = Button(mainframe, text="Read Caption")

path = ""
sentences = []

def browse_file(event):
  global path
  path = tkinter.filedialog.askopenfilename(filetypes=(("All files", "*.type"), ("All files", "*")))
  img = Image.open(path)
  img = img.resize((500, 500))
  img.show()
  inference()

def gen_click(event):
  img = Image.open(path)
  img = img.resize((500,450))
  background = Image.new('RGB', (500, 500), (255, 255, 255))
  background.paste(img, (0, 50))
  img = background
  draw = ImageDraw.Draw(img)
  font = ImageFont.truetype("Aaargh.ttf", 20)
  sentence = sentences[0]
  y = 0
  temp = ''
  cnt = 0
  sens = []
  for c in sentence:
    if c != ' ':
      temp += c
    else:
      sens.append(temp)
      temp = ''
  if temp != '' and temp != ' ':
    sens.append(temp)
  line = ''
  for sen in sens:
    if len(line + ' ' + sen) <= 45:
      line += ' ' + sen
    else:
      draw.text((0, y), line, (0, 0, 0), font=font)
      y += 25
      line = ' ' + sen
  draw.text((0, y), line, (0, 0, 0), font=font)
  img.show()

def voice_click(event):
  caption = sentences[0]
  texttospeech.convert_to_voice(caption)

browse_button.grid(row=1, column=1)
gen_button.grid(row=2, column=1)
voice_button.grid(row=3, column=1)

browse_button.bind("<Button-1>", browse_file)
gen_button.bind("<Button-1>", gen_click)
voice_button.bind("<Button-1>", voice_click)


def inference():
  # build the inference graph
  g = tf.Graph()
  with g.as_default():
    model = inference_wrapper.InferenceWrapper(FLAGS.rnn_type)
    restore_fn = model.build_graph_from_config(configuration.ModelConfig(), FLAGS.checkpoint_path)
  g.finalize()

  # Create the vocabulary.
  vocab = vocabulary.Vocabulary(FLAGS.vocab_file)

  filename = path

  with tf.Session(graph=g) as sess:
    # Load the model from checkpoint.
    restore_fn(sess)

    # Prepare the caption generator. Here we are implicitly using the default
    # beam search parameters. See caption_generator.py for a description of the
    # available beam search parameters.
    generator = caption_generator.CaptionGenerator(model, vocab)

    with tf.gfile.GFile(filename, "rb") as f:
      image = f.read()
    captions = generator.beam_search(sess, image)
    print("Captions for image %s:" % os.path.basename(filename))
    global sentences
    sentences = []
    for i, caption in enumerate(captions):
      # Ignore begin and end words.
      sentence = [vocab.id_to_word(w) for w in caption.sentence[1:-1]]
      sentence = " ".join(sentence)
      sentences.append(sentence)
      print(" %d) %s (p=%f)" % (i, sentence, math.exp(caption.logprob)))

root.mainloop()