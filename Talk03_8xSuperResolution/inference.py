import keras
from keras.models import load_model, Model
from keras.utils import plot_model
import numpy as np
import glob
import cv2
import matplotlib.pyplot as plt

print('Keras:',keras.__version__)

def compare_pics(name,x,y,z):
    fig = plt.figure(figsize=(50,30))
    a=fig.add_subplot(1,3,1)
    imgplot = plt.imshow(x)
    a=fig.add_subplot(1,3,2)
    imgplot = plt.imshow(y)
    a=fig.add_subplot(1,3,3)
    imgplot = plt.imshow(z)
    plt.savefig('results/{}.png'.format(name[-10:-4]))
    plt.close()

def select_test_images():

	images_fname = []
	test_images = []
	original_images = []

	path = '/media/dennis/197bed8d-a187-4bde-97a3-44305bd7396f/CelebA/Img/img_align_celeba_smallres_test/'

	for fname in glob.glob(path+'*.jpg'):
		images_fname.append(fname)
		test_image = cv2.cvtColor(cv2.imread(fname), cv2.COLOR_BGR2RGB)
		original_image = cv2.cvtColor(cv2.imread(fname.replace('img_align_celeba_smallres_test', 'img_align_celeba_test')), cv2.COLOR_BGR2RGB)
		test_images.append(test_image)
		original_images.append(original_image)

	return images_fname, test_images, original_images

rn_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32) #RGB

model = load_model('weights/celba_2k_8x_100.hdf5', custom_objects={'rn_mean': rn_mean})
model.summary()

#print(model.get_layer('model_3').get_config())
plot_model(model.get_layer('model_3'), to_file='model_3.png', show_shapes=True, show_layer_names=True)

superres_model = Model(model.get_layer('model_3').get_layer('input_1').input, model.get_layer('model_3').get_layer('lambda_1').output) #conv2d_13
plot_model(superres_model, to_file='superres.png', show_shapes=True, show_layer_names=True)

images_fname, test_images, original_images = select_test_images()

resized_images = []
for img in test_images:
	resized_images.append(cv2.resize(img, (22,22), interpolation = cv2.INTER_CUBIC))

result_images = superres_model.predict(np.array(resized_images))
	
# for i in range(len(result_images)):
# 	result_images[i] = cv2.resize(result_images[i], (178,218), interpolation = cv2.INTER_CUBIC)

for i in range(len(images_fname)):
	#print(result_images[i][0][0].astype(int))
	print(result_images[i].astype(np.uint8).shape)
	resulttest = cv2.resize(result_images[i].astype(np.uint8), (178,218), interpolation = cv2.INTER_CUBIC)
	compare_pics(images_fname[i],test_images[i],resulttest,original_images[i])


plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)
