import numpy as np
from tkinter import *
import tkinter, tkinter.constants, tkinter.filedialog
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

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
tf.flags.DEFINE_string(flag_name="rnn_type", default_value= "lstm",
                       docstring="RNN cell type lstm/gru .")

#??
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
gen_button = Button(mainframe, text="Generate Caption")

path = ""
sentences = []

def browse_file(event):
    global path
    path = tkinter.filedialog.askopenfilename(filetypes=(("All files", "*.type"), ("All files", "*")))
    img = Image.open(path)
    img.show()
    inference()

def gen_click(event):
    img = Image.open(path)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("Aaargh.ttf", 20)
    sentence = ""
    for c in sentences[0]:
        if c != '<':
            sentence = sentence + c
        else:
            break
    y = 0
    temp = ""
    cnt = 0
    for c in sentence:
        temp = temp + c
        if c == ' ':
            cnt = cnt + 1
            if cnt == 6:
                cnt = 0
                draw.text((0, y), temp, (0, 0, 0), font=font)
                y = y + 25
                temp = ""

    if temp != "":
        draw.text((0, y), temp, (0, 0, 0), font=font)
    img.show()


browse_button.grid(row=1, column=1)
gen_button.grid(row = 2, column=1)

browse_button.bind("<Button-1>", browse_file)
gen_button.bind("<Button-1>", gen_click)

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


def inference():
    #Build the inference graph
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

    filename = path

    with tf.Session(graph=g) as sess:
        #load the model from checkpoint
        restore_fn(sess)

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
            print("  %d) %s (p=%f)" % (i, sentence, math.exp(caption.logprob)))

#if __name__ == "__main__":
    #tf.app.run()

root.mainloop()
