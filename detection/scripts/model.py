import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Add, Lambda

def conv_layer(x, filters, kernel_size=(3, 3), dilation_rate=1,
              activation='relu', padding='same', *, name):
    x = Conv2D(
        filters, 
        kernel_size=kernel_size, 
        activation=activation, 
        padding=padding, 
        name=name, 
        dilation_rate=dilation_rate
        )(x)
    return x

def pool_layer(x, pool_size=(2, 2), strides=(2, 2), padding='valid', *, name):
    x = MaxPooling2D(
        pool_size, 
        strides=strides,
        name=name,
        padding=padding
        )(x)
    return x

def sum_layers(x1, x2):
    return Add()([x1, x2])

def lambda_layer(func, x, name):
    return Lambda(func, name=name)(x)

def upsample(x):
    shape = (x.shape[1]*2, x.shape[2]*2)
    return tf.image.resize(x, shape, tf.image.ResizeMethod.BILINEAR)

def vgg16(input_shape, dilation=True):
    input_layer = tf.keras.Input(shape=input_shape)

    # Block 1
    x = conv_layer(input_layer, 64, name='conv1_1')
    x = conv_layer(x, 64, name='conv1_2')
    x = pool_layer(x, name='pool1')

    # Block 2 
    x = conv_layer(x, 128, name='conv2_1')
    x = conv_layer(x, 128, name='conv2_2')
    x = pool_layer(x, name='pool2')

    # Block 3
    x = conv_layer(x, 256, name='conv3_1')
    x = conv_layer(x, 256, name='conv3_2')
    x = conv_layer(x, 256, name='conv3_3')
    x = pool_layer(x, name='pool3')

    # Block 4
    x = conv_layer(x, 512, name='conv4_1')
    x = conv_layer(x, 512, name='conv4_2')
    x = conv_layer(x, 512, name='conv4_3')
    x = pool_layer(x, name='pool4')

    # Block 5
    x = conv_layer(x, 512, name='conv5_1')
    x = conv_layer(x, 512, name='conv5_2')
    x = conv_layer(x, 512, name='conv5_3')
    x = pool_layer(x, pool_size=(3, 3), strides=(1, 1), padding='same', name='pool5')

    # fc6
    if dilation:
        x = conv_layer(x, 1024, dilation_rate=6, activation=None, name='fc6')
    else:
        x = conv_layer(x, 1024, activation=None, name='fc6')
    
    # fc7
    x = conv_layer(x, 1024, kernel_size=(1, 1), activation=None, name='fc7')

    model = tf.keras.Model(inputs=input_layer, outputs=x)

    return model

def get_cls_and_link(x):
    name = x.name.split('/')[0]
    cls = conv_layer(x, 2, kernel_size=(1, 1), activation=None, name=f'pixel_cls_{name}')
    link = conv_layer(x, 16, kernel_size=(1, 1), activation=None, name=f'pixel_link_{name}')
    return cls, link

def pixel_link_model(input_shape):
    backbone = vgg16(input_shape)
    
    conv3_3 = backbone.get_layer('conv3_3').output
    conv4_3 = backbone.get_layer('conv4_3').output
    conv5_3 = backbone.get_layer('conv5_3').output
    fc7 = backbone.get_layer('fc7').output

    conv3_3_cls, conv3_3_link = get_cls_and_link(conv3_3)
    conv4_3_cls, conv4_3_link = get_cls_and_link(conv4_3)
    conv5_3_cls, conv5_3_link = get_cls_and_link(conv5_3)
    fc7_cls, fc7_link = get_cls_and_link(fc7)
    
    up1_cls, up1_link = \
        lambda_layer(upsample, sum_layers(conv5_3_cls, fc7_cls), name='upsample1_cls'), \
        lambda_layer(upsample, sum_layers(conv5_3_link, fc7_link), name='upsample1_link')

    up2_cls, up2_link = \
        lambda_layer(upsample, sum_layers(up1_cls, conv4_3_cls), name='upsample2_cls'), \
        lambda_layer(upsample, sum_layers(up1_link, conv4_3_link), name='upsample2_link')

    up3_cls, up3_link = \
        sum_layers(up2_cls, conv3_3_cls), \
        sum_layers(up2_link, conv3_3_link)

    out_cls, out_link = up3_cls, up3_link

    model = tf.keras.Model(inputs=backbone.input, 
                           outputs=[out_cls, out_link],
                           name='pixellink')
  
    return model
