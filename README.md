# ImageCaptionByVoice
Implementation of a model that automaticaly generate caption for images.

this implementation is based on this repo: https://github.com/tensorflow/models/blob/master/research/im2txt

following this paper:

"Show and Tell: Lessons learned from the 2015 MSCOCO Image Captioning Challenge.", http://arxiv.org/abs/1609.06647

The model can be trained on GRU cell or LSTM by choosing the type of rnn cell before training 

This link contains weights for training the model: https://www.dropbox.com/sh/ty8zsc1zkyl9hqx/AADsI_PgzXpqCu9H7VaHBttAa?dl=0

LSTM was trained for almost 700000 steps and GRU for 600000

GRU showed slightly better numbers as an output from the evaluation metrics.

Automatic score of our model for the following metric respectively CIDER, METEOR, ROUGE, and BLEU-4

Inception-LSTM model: 0.888, 0.239, 0.512, 0.284

Inception-GRU  model: 0.891, 0.241, 0.515, 0.287


# User Manual
# 1. Install required packages

  -TensorFlow 1.0 or greater

  -Pillow

  -Heapq

  -OS

  -Json

  -Glop

  -NumPy

  -Natural Language Toolkit (NLTK)

  -First install NLTK

  Then install the NLTK data package "punkt"

  -PlaySound

  -Gtts

  -Tkinter


# 2. Training
 
 2.1 Prepare the Training Data
 
 downloading dataset
 
 Location to save the MSCOCO data :
 
 MSCOCO_DIR="${HOME}/im2txt/data/mscoco"
 
 ./download_and_preprocess_mscoco "${MSCOCO_DIR}"

 running build_mscoco
  
  Note that :
  train_shards = ( train_dataset size * 5 ) /2300
  val_shards = (val_dataset size * 5 ) /2300
  test_shards = ( test_dataset size * 5 ) /2300

  OUTPUT_DIR="${HOME}/im2txt/data/mscoco"
  
  SCRATCH_DIR="${OUTPUT_DIR}/raw-data"
  
  TRAIN_IMAGE_DIR="${SCRATCH_DIR}/train2014"
  
  VAL_IMAGE_DIR="${SCRATCH_DIR}/val2014"
  
  TRAIN_CAPTIONS_FILE="${SCRATCH_DIR}/annotations/captions_train2014.json"
  
  VAL_CAPTIONS_FILE="${SCRATCH_DIR}/annotations/captions_val2014.json"
  
  python3 build_mscoco_data.py --train_image_dir="${TRAIN_IMAGE_DIR}" --val_image_dir="${VAL_IMAGE_DIR}"   --train_captions_file="${TRAIN_CAPTIONS_FILE}"   --val_captions_file="${VAL_CAPTIONS_FILE}"   --output_dir="${OUTPUT_DIR}" --word_counts_output_file="${OUTPUT_DIR}/word_counts.txt" --train_shards=256 --val_shards=4 --test_shards=8


 2.2 Download the Inception v3 Checkpoint
  
  Location to save the Inception v3 checkpoint.
  
  INCEPTION_DIR="${HOME}/im2txt/data"
  
  mkdir -p ${INCEPTION_DIR}
  
  wget"http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz"
  
  tar -xvf "inception_v3_2016_08_28.tar.gz" -C ${INCEPTION_DIR}
  
  rm "inception_v3_2016_08_28.tar.gz”

2.3 Training a Model
  
  MODEL_DIR="${HOME}/im2txt/model"
  
  INCEPTION_CHECKPOINT="${HOME}/im2txt/data/inception_v3.ckpt
  
  MSCOCO_DIR="${HOME}/im2txt/data/mscoco"

  python3 train.py   --input_file_pattern="${MSCOCO_DIR}/train-?????-of-00256" --inception_checkpoint_file="${INCEPTION_CHECKPOINT}"  --train_dir="${MODEL_DIR}/train" --train_inception=false   --number_of_steps=70000 --rnn_type="lstm"



# 3. Testing
  
  3.1 Test one image (using GUI)
   
   Press “Browse” button and browse an image anywhere on your PC.
   
   Press “Show Caption” button and the caption will be written on top of the image.
   
   Press “Read Caption” button and get the result as a sound.
  
  3.2 Test a folder of images
    
    run run_inference script:
	  
    python3 run_inference.py
    
    --input_files = “” #the folder directory of the images you want to test
    --mode = False

