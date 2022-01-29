import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"


import tensorflow as tf
from tensorflow.python.keras.backend import set_session
config = tf.compat.v1.ConfigProto() 
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess2 = tf.compat.v1.Session(config=config)
set_session(sess2)  # set this TensorFlow session as the default

import numpy as np
import matplotlib.pyplot as plt
import sys
import pdb
from keras_utils import train_model, evaluate_model
from classification_models.classification_models.keras import Classifiers


def identity_block(x, filter):
    # copy tensor to variable called x_skip
    x_skip = x
    # Layer 1
    x = tf.keras.layers.Conv2D(filter, (3,3), padding = 'same')(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    x = tf.keras.layers.Activation('relu')(x)
    # Layer 2
    x = tf.keras.layers.Conv2D(filter, (3,3), padding = 'same')(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    # Add Residue
    x = tf.keras.layers.Add()([x, x_skip])     
    x = tf.keras.layers.Activation('relu')(x)
    return x

def convolutional_block(x, filter):
    # copy tensor to variable called x_skip
    x_skip = x
    # Layer 1
    x = tf.keras.layers.Conv2D(filter, (3,3), padding = 'same', strides = (2,2))(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    x = tf.keras.layers.Activation('relu')(x)
    # Layer 2
    x = tf.keras.layers.Conv2D(filter, (3,3), padding = 'same')(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    # Processing Residue with conv(1,1)
    x_skip = tf.keras.layers.Conv2D(filter, (1,1), strides = (2,2))(x_skip)
    # Add Residue
    x = tf.keras.layers.Add()([x, x_skip])     
    x = tf.keras.layers.Activation('relu')(x)
    return x

def model_fn(shape = (400, 400, 2)):

    
    # Step 1 (Setup Input Layer)
    x_input = tf.keras.layers.Input(shape)
    x = tf.keras.layers.ZeroPadding2D((3, 3))(x_input)
    # Step 2 (Initial Conv layer along with maxPool)
    x = tf.keras.layers.Conv2D(64, kernel_size=7, strides=2, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')(x)
    # Define size of sub-blocks and initial filter size
    block_layers = [3, 4, 6, 3]
    filter_size = 64
    # Step 3 Add the Resnet Blocks
    for i in range(4):
        if i == 0:
            # For sub-block 1 Residual/Convolutional block not needed
            for j in range(block_layers[i]):
                x = identity_block(x, filter_size)
        else:
            # One Residual/Convolutional Block followed by Identity blocks
            # The filter size will go on increasing by a factor of 2
            filter_size = filter_size*2
            x = convolutional_block(x, filter_size)
            for j in range(block_layers[i] - 1):
                x = identity_block(x, filter_size)
    # Step 4 End Dense Network
    #x = tf.keras.layers.AveragePooling2D((2,2), padding = 'same')(x)
    
    """
    Resnet34, _ = Classifiers.get('resnet34')
    inner_model = Resnet34(input_shape=shape, weights='imagenet', include_top=False)
    inner_model.trainable=False
    x_input = tf.keras.layers.Input(shape)
    x = inner_model(x_input, training=False)
    """
    last_conv_sz = [512,256,128,64,32]
    
    for i in last_conv_sz:
        x = tf.keras.layers.Conv2D(i, kernel_size=3)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        #y = Dropout(0.1)(y)
 
    x = tf.keras.layers.Conv2D(6, kernel_size=1, use_bias=False)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(1, activation = 'linear')(x)

    model = tf.keras.models.Model(inputs = x_input, outputs = x, name = "ResNet34_Alt")
    model.summary()
    return model



eval_mode = False
frames = 2
if len(sys.argv) > 1:
    eval_mode = True
    
base_weights_dir = '%s_%d_cells_weights/'


weights_dir = base_weights_dir

if not os.path.exists('weights/' + weights_dir):
    os.makedirs('weights/' + weights_dir)

# try:
#pdb.set_trace()
model = model_fn(shape=(400,400,frames))

# comment out the training code to only evaluate !

if not eval_mode:
    train_model(model, epochs=2000, batch_size=128, frames=frames)


acc = evaluate_model(model, batch_size=128, frames=frames)
                     

