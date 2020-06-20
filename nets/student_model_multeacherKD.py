from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Activation, Lambda
from tensorflow.keras.layers import AveragePooling2D, Input, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.initializers import he_normal
import numpy as np
import os

SEED = 6

def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder

    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)

    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer=he_normal(seed=SEED),
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x

def resnet_v1(input_shape, depth, num_classes=10):
    """ResNet Version 1 Model builder [a]

    Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
    Last ReLU is after the shortcut connection.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filters is
    doubled. Within each stage, the layers have the same number filters and the
    same number of filters.
    Features maps sizes:
    stage 0: 32x32, 16
    stage 1: 16x16, 32
    stage 2:  8x8,  64
    The Number of parameters is approx the same as Table 6 of [a]:
    ResNet20 0.27M
    ResNet32 0.46M
    ResNet44 0.66M
    ResNet56 0.85M
    ResNet110 1.7M

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs)
    # Instantiate the stack of residual units
    for stack in range(3):
        if stack==2:
            num_res_blocks -= 1
            
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            y = resnet_layer(inputs=x,
                             num_filters=num_filters,
                             strides=strides)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters,
                             activation=None)
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = keras.layers.add([x, y])
            x = Activation('relu')(x)
        num_filters *= 2

    return (inputs, x, num_filters)

def resnet_student(input_shape, temperature, body_depth, num_head, num_blocks_head, num_classes=10):
    
    (inputs, x, num_filters) = resnet_v1(input_shape, body_depth, num_classes)
    heads = []
    
    num_filters = int(num_filters/2)
    
    if num_blocks_head!=0:
        for i in range(num_head):
            x_1 = x
            #num_filters = int(num_filters/2)
            for res_block in range(num_blocks_head):
                strides = 1
                y = resnet_layer(inputs=x_1,
                                    num_filters=num_filters,
                                    strides=strides)
                y = resnet_layer(inputs=y,
                                    num_filters=num_filters,
                                    activation=None)   
                x_1 = keras.layers.add([x_1, y])
                x_1 = Activation('relu')(x_1)
            
            x_1 = AveragePooling2D(pool_size=8)(x_1)
            y = Flatten()(x_1)
            normal_logits = Dense(num_classes,
                            kernel_initializer=he_normal(seed=SEED), name= 'logits_'+str(i+1))(y)
            normal_prob = Activation('softmax', name='softmax_'+str(i+1))(normal_logits)
        
            soft_logits = Lambda(lambda k:k/temperature)(normal_logits)
            soft_prob = Activation('softmax', name= 'soft_prob_'+str(i+1))(soft_logits)
            head = tf.keras.layers.concatenate([normal_prob, soft_prob])
            heads.append(head)
            
        model = Model(inputs=inputs, outputs=heads)
        
    elif num_blocks_head==0:
        model = Model(inputs=inputs, outputs=x)
        
    return model
    
    
    
    
    

    



