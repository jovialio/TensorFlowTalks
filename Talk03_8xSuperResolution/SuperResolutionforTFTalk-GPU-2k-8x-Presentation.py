
# coding: utf-8

# ### Imports

# In[1]:


#get_ipython().magic(u'matplotlib inline')


import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3" 

import numpy as np
import glob
import cv2
import pickle
from random import shuffle

from PIL import Image
import os
import sys
import bcolz
import matplotlib.pyplot as plt

import tensorflow as tf
import keras

from keras_tqdm import TQDMNotebookCallback
from keras import initializers
from keras.applications.resnet50 import ResNet50, decode_predictions
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image

from keras.layers import *
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import decode_predictions, preprocess_input

from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.utils import multi_gpu_model
from keras.layers import Input
from keras.layers.convolutional import Convolution2D, AveragePooling2D, MaxPooling2D
from keras.utils.data_utils import get_file
from keras.utils.layer_utils import convert_all_kernels_in_model
import keras.backend as K

from vgg16_avg import VGG16_Avg


# ### SDK Versions
# 

# In[2]:


print('TensorFlow:',tf.__version__)
print('Keras:',keras.__version__)


# # Celeb Faces

# In[6]:


#def save_array(fname, arr): c=bcolz.carray(arr, rootdir=fname, mode='w'); c.flush()
#def load_array(fname): return bcolz.open(fname)[:]


# In[ ]:

folder = '../CelebA/img_align_celeba/*jpg'

original_img_fnames = []

for fname in glob.glob(folder, recursive=True):
    original_img_fnames.append(fname)
    
def batchGenerator(image_fnames, batch_size=128):
    
    targ = np.zeros((batch_size, 128))

    while True:

        orginal_image_batch = []
        small_res_image_batch = []
        shuffle(image_fnames)
        
        for fname in image_fnames:
            
            small_res_image = cv2.resize(cv2.cvtColor(cv2.imread(fname.replace('img_align_celeba', 'img_align_celeba_smallres')), cv2.COLOR_BGR2RGB),(22,22), interpolation = cv2.INTER_CUBIC)
            original_image = cv2.resize(cv2.cvtColor(cv2.imread(fname), cv2.COLOR_BGR2RGB),(176,176), interpolation = cv2.INTER_CUBIC)
                
            orginal_image_batch.append(original_image)
            small_res_image_batch.append(small_res_image)
            
            if len(orginal_image_batch) == batch_size:
                #image_batch = [PIL.Image.fromarray(i) for i in image_batch]
                #val_data = [i.resize((INPUT_YOLO_FEATURE_SIZE, INPUT_YOLO_FEATURE_SIZE), PIL.Image.BICUBIC) for i in val_data]
                #image_batch = [np.array(image, dtype=np.float) for image in image_batch]
                #image_batch = [image/255. for image in image_batch]
                yield([np.array(small_res_image_batch), np.array(orginal_image_batch)], targ)
                orginal_image_batch = []
                small_res_image_batch = []
  
#    if len(orginal_image_batch) != 0:
#        yield([np.array(small_res_image_batch), np.array(orginal_image_batch)], targ)


# In[7]:


"""dpath = '/Users/samwitteveen/Dropbox/ai_learning/Key DL Learning'
#dpath = '/home/paperspace/Dropbox/ai_learning/Key DL Learning'
bcolz_hr = '/celeba-176_2k.bc'
bcolz_lr = '/celeba-44_2k.bc'
bcolz_elr = '/celeba-22_2k.bc'
bcolz_test_lr = '/celeba-44_test.bc'
bcolz_test_hr = '/celeba-176_test.bc'
bcolz_test_elr ='/celeba-22_test.bc'

#Original Image - Training
arr_hr = load_array(dpath+bcolz_hr)
#Small Res Image - Training
arr_elr = load_array(dpath+bcolz_elr)
#Original Image - Test
arr_test_hr = load_array(dpath+bcolz_test_hr)
#Small Res Image - Test
arr_test_elr = load_array(dpath+bcolz_test_elr)

#arr_lr = load_array(dpath+bcolz_lr)"""


# In[5]:


#arr_elr.shape


# 
# #### Image Preproc

# In[8]:


# vgg preproc
rn_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32) #RGB
preproc = lambda x: (x - rn_mean)[:, :, :, ::-1] #Switch back to BGR


# ### Set up Network parts
# 
# ConvBlock  
# ResBlock

# In[9]:


def conv_block(x, num_filters, filter_size, stride=(2,2), mode='same', act=True):
    x = Convolution2D(num_filters, filter_size, filter_size, subsample=stride, border_mode=mode)(x)
    x = BatchNormalization()(x)
    return Activation('relu')(x) if act else x

def res_block(initial_input, num_filters=64):
    x = conv_block(initial_input, num_filters, 3, (1,1))
    x = conv_block(x, num_filters, 3, (1,1), act=False)
    return add([x, initial_input])


# Deconvolution / Transposed Conv / Fractionally Strident Convs

# In[10]:


# Up Sampling block aka Decon
def up_block(x, num_filters, size):
    x = keras.layers.UpSampling2D()(x)
    x = Convolution2D(num_filters, size, size, border_mode='same')(x)
    x = BatchNormalization()(x)
    return Activation('relu')(x)


# ### Set up Deconv network - Upsampling network

# In[1]:


def get_upsampling_model_8x():
    inp=Input([22,22,3])
    x=conv_block(inp, 64, 9, (1,1))
    x=res_block(x)
    x=res_block(x)
    x=res_block(x)
    x=res_block(x)
    x=up_block(x, 64, 3)
    x=up_block(x, 64, 3)
    x=up_block(x, 64, 3)
    x=Convolution2D(3, 9, 9, activation='tanh', border_mode='same')(x)
    outp=Lambda(lambda x: (x+1)*127.5)(x)
    return inp,outp


# In[13]:


#up_model = Lambda(get_upsampling_model_4x(arr_elr))


# In[14]:


#up_model.summary()


# In[16]:


# this gets the output 
upsampled_inp,upsampled_output = get_upsampling_model_8x()


# In[17]:

with tf.device('/cpu:0'):
    up_model2 = Model(upsampled_inp,upsampled_output)
    up_model2.summary()


# ### VGG network
# 
# this is only used to for calculating our loss
# 

# In[18]:


#vgg input
vgg_inp=Input([176,176,3])

#vgg network
vgg= VGG16(include_top=False, input_tensor=vgg_inp)
for l in vgg.layers: l.trainable=False


# In[19]:


# Lambda makes a layer of a function/ this makes the preprocessing a layer
preproc_layer = Lambda(preproc)


# In[20]:


# get the vgg output 
vgg_out_layer = vgg.get_layer('block2_conv2').output

# making model Model(inputs, outputs)
with tf.device('/cpu:0'):
    vgg_content = Model(vgg_inp, vgg_out_layer)
    vgg_content.summary()

    # In[21]:

    # this is the VGG model with the HR input
    vgg_hr_image = vgg_content(preproc_layer(vgg_inp))

    # this is the upsampled network
    vgg_it_op = vgg_content(preproc_layer(upsampled_output))

    # ### Loss and Optimisers
    # 

    # In[22]:
    loss = Lambda(lambda x: K.sqrt(K.mean((x[0]-x[1])**2, (1,2))))([vgg_hr_image, vgg_it_op])


# In[23]:

with tf.device('/cpu:0'):
    sr_model = Model([upsampled_inp, vgg_inp], loss)

parallel_model = multi_gpu_model(sr_model, gpus=4)

parallel_model.compile('adam', 'mse')
#sr_model.compile('adam', 'mse')

# ### Training

# In[82]:

tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

checkpointer = ModelCheckpoint(filepath='./weights/celba_2k_8x_{epoch:02d}.hdf5', verbose=1)
history = parallel_model.fit_generator(batchGenerator(original_img_fnames),
        steps_per_epoch=1582, epochs=100, callbacks=[checkpointer, tensorboard])
with open('trainHistoryDict', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)
#sr_model.fit_generator(batchGenerator(original_img_fnames), 8, 20)


# ### Saver
# 

# In[83]:


it_model = Model(upsampled_inp, upsampled_output)
it_model.save_weights('./weights/'+'celba_2k_8x.h5')


# In[24]:


it_model = Model(upsampled_inp, upsampled_output)
it_model.load_weights('./weights/'+'celba_2k_8x.h5')


# ### Examples
# 
# show 1. low res 2. hi-res 3. ground truth
# 

# In[25]:


#get_ipython().magic(u'time p = it_model.predict(arr_elr[0:50])')
#p.shape


# In[26]:


def compare_pics(x,y):
    fig = plt.figure(figsize=(30,30))
    a=fig.add_subplot(1,2,1)
    imgplot = plt.imshow(x)
    a=fig.add_subplot(1,2,2)
    imgplot = plt.imshow(y)


# In[27]:


compare_pics(arr_elr[10].astype('uint8'), p[10].astype('uint8'))


# In[32]:


compare_pics(arr_elr[13].astype('uint8'), p[13].astype('uint8'))


# In[33]:


compare_pics(arr_hr[13].astype('uint8'),p[13].astype('uint8'))


# # Predicting on Test set that the model hasn't seen

# In[34]:


get_ipython().magic(u'time p = it_model.predict(arr_test_elr[0:50])')
p.shape


# In[35]:


compare_pics(arr_test_elr[24].astype('uint8'),p[24].astype('uint8'))


# In[45]:


compare_pics(arr_test_hr[24].astype('uint8'),p[24].astype('uint8'))


# # Let's Predict on the Prediction or 64x SR

# In[46]:


new_upsampled_inp,new_upsampled_output = get_upsampling_model_4x(p[20:25])


# In[47]:


new_up_model = Model(new_upsampled_inp,new_upsampled_output)
new_up_model.summary()


# In[48]:


new_new_up_model = Model(new_upsampled_inp, new_upsampled_output)
new_new_up_model.load_weights('./weights/'+'celba_2k_8x.h5')


# In[50]:


get_ipython().magic(u'time new_p = new_new_up_model.predict(p[24:25])')


# In[51]:


new_p.shape


# In[52]:


compare_pics(new_p[0].astype('uint8'),p[24].astype('uint8'))


# In[ ]:


compare_pics(new_p[4].astype('uint8'),arr_test_hr[24].astype('uint8'))


# ### Credits
# 
# Papers: "Perceptual Losses for Real-Time Style Transfer and Super-Resolution" by Johnson, et.al
# http://arxiv.org/abs/1603.08155
# 
# "A Neural Algorithm of Artistic Style" by Gatys et.al 
# http://arxiv.org/abs/1508.06576v2
# 
# Code ideas inspired by Jermey Howard's SFData Institute Advanced Deep Learning Course 
