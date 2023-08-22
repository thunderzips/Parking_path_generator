from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from numpy import expand_dims
from matplotlib import pyplot

from time import time

def load_image(filename, size=(256,256)):
	pixels = load_img(filename, target_size=size)
	pixels = img_to_array(pixels)
	pixels = (pixels - 127.5) / 127.5
	pixels = expand_dims(pixels, 0)
	return pixels

def append_images(im1, im2):

	for i in range(im1.shape[0]):
		for j in range(im1.shape[1]):
			#check if pixel is not white
			if im1[i,j,0] < 0.7 or im1[i,j,1] < 0.7 or im1[i,j,2] < 0.7:
				im2[i,j] = im1[i,j]
	
	return im2
				



# load source image

im = 74, 76, 121


model = load_model('models/model_f_aug9_8.h5')

avg_time = 0

for im in range(1,150):

	src_image = load_image(f'images/testing/input_images/environment{im}.png')
	print('Loaded', src_image.shape)

	start_time = time()

	gen_image = model.predict(src_image)

	end_time = time()

	avg_time += end_time-start_time

	print(f"Time taken: {end_time - start_time}")

	print(f"avg time: {avg_time/im}")

	# scale from [-1,1] to [0,1]
	gen_image = (gen_image + 1) / 2.0

	gen_image = append_images(gen_image[0], src_image[0])

	pyplot.subplot(1,2,1)
	pyplot.imshow(gen_image)
	pyplot.subplot(1,2,2)
	pyplot.imshow(load_image(f'images/testing/output_images_truth/environment{im}.png')[0])
	pyplot.axis('off')
	pyplot.savefig(f'images/testing/results_processed/environment{im}.png')
	# pyplot.show()