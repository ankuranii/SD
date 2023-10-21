# General
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import argparse
import shutil
import tensorflow as tf
# Image Processing
from PIL import Image
import clip

# Image Generation
from transformers import CLIPTokenizer
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import torch

# Image similarity
from Simililarity_Score import ImageSimilarity


# Arguments 
parser=argparse.ArgumentParser(description='Python Script to Generate Stable Diffusion Images')

parser.add_argument("--CSV_PATH",type=str,default = '/local/data/sghosh_dg/AISCC/AGID_0/AGID/Data/NYT.csv', help='Pass the path of CSV file (e.g /content/data.csv)')
parser.add_argument('--ORIGINAL_DIR',type=str,default = '/local/data/sghosh_dg/AISCC/AGID_0/AGID/Data/tweetImages/', help='Directory path of Images (e.g /content/images/)')
parser.add_argument('--NUM_IMAGES_PER_PROMPT',type=int,default=1,help='Number of Images per prompt')
parser.add_argument('--THRESHOLD_CLIP',type=float,default=0.60,help='Threshold for similarity using CLIP Embeddings')
parser.add_argument('--THRESHOLD_VGG',type=float,default=0.65,help='Threshold for similarity using VGG Embeddings')

args = parser.parse_args()

# Access the parsed arguments
csv_path = args.CSV_PATH
ORIGINAL_DIR = args.ORIGINAL_DIR

#if not os.path.exists(ORIGINAL_DIR):
#  os.makedirs(ORIGINAL_DIR)

NUM_IMAGES_PER_PROMPT = args.NUM_IMAGES_PER_PROMPT
THRESHOLD_CLIP = args.THRESHOLD_CLIP
THRESHOLD_VGG = args.THRESHOLD_VGG

print('Reading the Data!!')
df=pd.read_csv(csv_path)
prompts=list(df.tweetContentProcessed)

# Directory to store the Images
ORIGINAL_USEFUL='ORIGINAL_USEFUL/'

FAKE_USEFUL='FAKE_USEFUL/'

FAKE_ALL='FAKE_ALL/'


if not os.path.exists(FAKE_ALL):
    os.makedirs(FAKE_ALL)

if not os.path.exists(FAKE_USEFUL):
    os.makedirs(FAKE_USEFUL)

if not os.path.exists(ORIGINAL_USEFUL):
    os.makedirs(ORIGINAL_USEFUL)


DEVICE = 'cuda:1' if torch.cuda.is_available else 'cpu' # device
STRENGTH = 0.3 # The noise to add to original image
NUM_INFERENCE_STEPS = 100 # Number of inference steps to the Diffusion Model

print(f'Loading the Models to {DEVICE}')
DIFFUSION_MODEL_PATH = "stabilityai/stable-diffusion-2-1" # Set the model path to load the diffusion model from
CLIP,preprocess = clip.load("ViT-B/32", device=DEVICE)

scheduler = EulerDiscreteScheduler.from_pretrained(DIFFUSION_MODEL_PATH, subfolder="scheduler")

model = StableDiffusionPipeline.from_pretrained(DIFFUSION_MODEL_PATH, scheduler=scheduler)#, torch_dtype=torch.float16)
model = model.to(DEVICE) #("cuda")

model.set_progress_bar_config(disable=True) #To disable progress bar


similarity=ImageSimilarity()

print('Start Generating images')

with torch.no_grad():
  try:
    for i,prompt in enumerate(tqdm(prompts)):
      max_score = 0
      idx = str(df.id[i])
      image_path =  ORIGINAL_DIR + idx + '.jpg'
      original_image = Image.open(image_path)

      if len(np.array(original_image).shape) != 3:
         continue
      
      else:
         


        print(np.array(original_image).shape, 'o')

        for j in range(NUM_IMAGES_PER_PROMPT):

          if not os.path.isfile(f'{FAKE_ALL}/{idx}.jpg'):
            output_image = model(prompt, num_inference_steps=NUM_INFERENCE_STEPS).images[0]
            print(f"output_image = {output_image}, i = {i}")
            print(np.array(output_image).shape) #This may tell about the error
            #exit(1)

            clip_similarity = similarity.get_similarity_score(output_image, original_image, model_name="CLIP")
            vgg_similarity = similarity.get_similarity_score(output_image, original_image, model_name="VGG")
            output_image.save(f'{FAKE_ALL}/{idx}.jpg')

            if clip_similarity > THRESHOLD_CLIP and vgg_similarity > THRESHOLD_VGG:
              output_image.save(f'{FAKE_USEFUL}/{idx}.jpg')
              original_image.save(f'{ORIGINAL_USEFUL}/{idx}.jpg')
              break
          else:
            continue

  except Exception as e:
    print(f'The Error is {e}')






