"""
main.py
written by theo
"""

import time
import numpy as np
import cv2
import imageio.v2 as imageio
# import wave # https://docs.python.org/3/library/wave.html
from scipy.io import wavfile
from PIL import Image # https://pillow.readthedocs.io/en/stable/index.html

# Init start_time
start_time = time.time()

# Constants
FILE_SOUND = "lovetones.wav"
FILE_IMAGE = "cursed.tiff"

def divide_chunks(l, n):
	for i in range(0, len(l), n):
		yield l[i:i + n]

# Open files
# sound = wave.open(FILE_SOUND, "r")
sound = wavfile.read(FILE_SOUND)
image = Image.open(FILE_IMAGE)

# Get information
# sound_params = sound.getparams()._asdict()
sound_data = np.array(sound[1], dtype=float)[0:, 0]
image_data = np.asarray(image)
image_h, image_w, *_ = image_data.shape
fps = 60

numpixels = image.height * image.width
numframes = len(sound_data)
framesperpixel = np.uint(np.floor(numframes/numpixels))

samplerate = sound[0]
maxframe = np.abs(sound_data.max())
minframe = np.abs(sound_data.min())
biggerminmax = max(maxframe, minframe)
mult = 255/biggerminmax

soundlength = numframes / samplerate
samplesperframe = np.uint(np.floor(samplerate/fps))

print()
print(f"Number of pixels: {image.height} * {image.width} = {numpixels}")
print(f"Number of audio frames: {numframes}")
print(f"Frames per pixel: {framesperpixel}")
print(f"Sample rate: {samplerate}")
print(f"Sound length: {soundlength:.3f}s")

# 24 frames per second, 44100 samples per second
# 1837 samples per frame
print(f"Samples per frame: {samplesperframe}")

# ========= TRANSFORMATIONS ========= #

# take the average of 1837 samples at a time
# use that average to scale the red
# blue, green = 0, 0

averageframes = []
for i in range(0, numframes, samplesperframe):
	averageframes.append(np.average(sound_data[i:i+samplesperframe]))

extremeaverage = max(np.abs(min(averageframes)), np.abs(max(averageframes)))
mult = 500/extremeaverage
print(f"Multiplier: {mult}")

images = []
for av in averageframes:
	img = np.copy(image_data) 
	img[:,:,1] = 0
	img[:,:,0] = img[:,:,0] * mult * av
	images.append(img)


# =================================== #

print(f"\n----- edit:  {time.time() - start_time} seconds -----\n")
start_time = time.time()

print("Done editing image. Writing image...")

# Convert image_data back into an image and save output

image = Image.fromarray(images[np.random.randint(0, len(images))])
image.save("output.png")

out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'MP42'), fps, (image_w, image_h))
for img in images:
	out.write(img)

out.release()

# Clean up
# sound.close()
image.close()

# Timer
print(f"\n----- write: {time.time() - start_time} seconds -----\n")

"""
Notes
=====

cursed.jpg
	shape (672, 504, 4)
	number of pixels = 338688
	take every 11th audio frame for a pixel

chant.wav
    nchannels = 2
    sampwidth = 2
    framerate = 48000
    nframes = 3963506
    comptype = NONE
    compname = not compressed

	4 bytes per frame: b'\x00\x00\x00\x00'
	240 000 frames for 5 seconds, 960 000 bytes

TIFF:
	bit depth - 2^n different colours, 8->256

RGBA
	(red, green, blue, alpha)
	all from 0-255
	alpha is transparent->opaque

Visualisation ideas:

	- Move down through image, editing it row by row with audio frame data
"""