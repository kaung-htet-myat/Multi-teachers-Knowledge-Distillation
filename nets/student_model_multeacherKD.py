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

def resnet20_student(input_shape, temperature, num_classes=10):
    
    (inputs, x, num_filters) = resnet_v1(input_shape, 20, num_classes)
    
    # Head 1
    x_1 = x
    num_filters = int(num_filters/2)
    for res_block in range(2):
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
                    kernel_initializer=he_normal(seed=SEED), name= 'logits_1')(y)
    normal_prob = Activation('softmax', name='softmax_1')(normal_logits)
    
    soft_logits = Lambda(lambda k:k/temperature)(normal_logits)
    soft_prob = Activation('softmax', name= 'soft_prob_1')(soft_logits)
    head_1 = tf.keras.layers.concatenate([normal_prob, soft_prob])
    
    # Head 2
    x_2 = x
    for res_block in range(2):
        strides = 1
        y = resnet_layer(inputs=x_2,
                             num_filters=num_filters,
                             strides=strides)
        y = resnet_layer(inputs=y,
                             num_filters=num_filters,
                             activation=None)   
        x_2 = keras.layers.add([x_2, y])
        x_2 = Activation('relu')(x_2)
        
    x_2 = AveragePooling2D(pool_size=8)(x_2)
    y = Flatten()(x_2)
    normal_logits = Dense(num_classes,
                    kernel_initializer=he_normal(seed=SEED), name= 'logits_2')(y)
    normal_prob = Activation('softmax', name='softmax_2')(normal_logits) 
    
    soft_logits = Lambda(lambda k:k/temperature)(normal_logits)
    soft_prob = Activation('softmax', name= 'soft_prob_2')(soft_logits)
    head_2 = tf.keras.layers.concatenate([normal_prob, soft_prob])
    
    # Head 3
    x_3 = x
    for res_block in range(2):
        strides = 1
        y = resnet_layer(inputs=x_3,
                             num_filters=num_filters,
                             strides=strides)
        y = resnet_layer(inputs=y,
                             num_filters=num_filters,
                             activation=None)   
        x_3 = keras.layers.add([x_3, y])
        x_3 = Activation('relu')(x_3)
        
    x_3 = AveragePooling2D(pool_size=8)(x_3)
    y = Flatten()(x_3)
    normal_logits = Dense(num_classes,
                    kernel_initializer=he_normal(seed=SEED), name= 'logits_3')(y)
    normal_prob = Activation('softmax', name='softmax_3')(normal_logits) 
    
    soft_logits = Lambda(lambda k:k/temperature)(normal_logits)
    soft_prob = Activation('softmax', name= 'soft_prob_3')(soft_logits)
    head_3 = tf.keras.layers.concatenate([normal_prob, soft_prob])
    
    # Head 4
    x_4 = x
    for res_block in range(2):
        strides = 1
        y = resnet_layer(inputs=x_4,
                             num_filters=num_filters,
                             strides=strides)
        y = resnet_layer(inputs=y,
                             num_filters=num_filters,
                             activation=None)   
        x_4 = keras.layers.add([x_4, y])
        x_4 = Activation('relu')(x_4)
        
    x_4 = AveragePooling2D(pool_size=8)(x_4)
    y = Flatten()(x_4)
    normal_logits = Dense(num_classes,
                    kernel_initializer=he_normal(seed=SEED), name= 'logits_4')(y)
    normal_prob = Activation('softmax', name='softmax_4')(normal_logits) 
    
    soft_logits = Lambda(lambda k:k/temperature)(normal_logits)
    soft_prob = Activation('softmax', name= 'soft_prob_4')(soft_logits)
    head_4 = tf.keras.layers.concatenate([normal_prob, soft_prob])
    
    model = Model(inputs=inputs, outputs=[head_1, head_2, head_3, head_4])
    
    return model
    
    
    

    



