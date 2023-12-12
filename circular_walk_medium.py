import keras_cv
from tensorflow import keras
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import math
from PIL import Image

'''
Portions of this file are derived from KERAS random walk implementation
'''

#### PROMPT ####
prompt = "A painting of a lion in Africa"

model = keras_cv.models.StableDiffusion(jit_compile=True)

e = tf.squeeze(model.encode_text(prompt)) # Encode the text prompt
num_batch = 150 / 3 # Number of batches was found by testing time complexity 

x = tf.random.normal((64, 64, 4), dtype=tf.float64) # Picture dimensinos are 64 x 64 x 4
y = tf.random.normal((64, 64, 4), dtype=tf.float64)

scaled_x = tf.cos(tf.linspace(0, 2, 150) * math.pi) # Scale x by cos
scaled_y = tf.sin(tf.linspace(0, 2, 150) * math.pi) # Scale y by sin

noise = tf.add(tf.tensordot(scaled_x, x, axes=0), tf.tensordot(scaled_y, y, axes=0)) # Total noise sum
noise_split = tf.split(noise, num_batch) # Split noise into batches

# Generate pictures from the Stable Diffusion Model
pics = []
for b in range(num_batch):
    pics += [Image.fromarray(pic) for pic in model.generate_image(e,  num_steps=20, batch_size=3, diffusion_noise=noise_split[b], )]

# Same pictures as gif for visualization
pics[0].save("circular_medium.gif", save_all=True, append_images=pics[1:], loop=0, duration = 100,)
