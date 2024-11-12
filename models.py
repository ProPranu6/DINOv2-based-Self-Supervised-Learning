# https://arxiv.org/pdf/1804.03999.pdf - attention-UNet paper

import tensorflow as tf
import tensorflow.keras.layers as L
from tensorflow.keras.models import Model
tf.config.experimental_run_functions_eagerly(True)
from keras_unet_collection import models as pretrained_models

def conv_block(x, num_filters):
    x = L.Conv2D(num_filters, 3, padding="same")(x)
    x = L.BatchNormalization()(x)
    x = L.Activation("relu")(x)

    x = L.Conv2D(num_filters, 3, padding="same")(x)
    x = L.BatchNormalization()(x)
    x = L.Activation("relu")(x)

    return x

def encoder_block(x, num_filters):
    x = conv_block(x, num_filters)
    p = L.MaxPool2D((2, 2))(x)
    return x, p

def attention_gate(g, s, num_filters):
    Wg = L.Conv2D(num_filters, 1, padding="same")(g)
    Wg = L.BatchNormalization()(Wg)

    Ws = L.Conv2D(num_filters, 1, padding="same")(s)
    Ws = L.BatchNormalization()(Ws)

    out = L.Activation("relu")(Wg + Ws)
    out = L.Conv2D(1, 1, padding="same")(out)  #num filters set to 1 following original paper
    out = L.Activation("sigmoid")(out)

    return out * s

def decoder_block(x, s, num_filters):
    x = L.UpSampling2D(interpolation="bilinear")(x)
    s = attention_gate(x, s, num_filters)
    x = L.Concatenate()([x, s])
    x = conv_block(x, num_filters)
    return x

def attention_unet(input_shape):
    inputs = L.Input(input_shape)

    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)

    b1 = conv_block(p3, 512)

    d1 = decoder_block(b1, s3, 256)
    d2 = decoder_block(d1, s2, 128)
    d3 = decoder_block(d2, s1, 64)

    outputs = L.Conv2D(1, 1, padding="same", activation='sigmoid')(d3)

    model = Model(inputs, outputs, name="Attention-UNET")
    return model    

def pretrained_encoder_attention_unet(input_shape=(512, 512)):
  return pretrained_models.att_unet_2d(
    (*input_shape, 3), 
    filter_num=[32, 64, 128, 256, 512], 
    n_labels=1,
    stack_num_down=2, 
    stack_num_up=2,
    activation='ReLU', 
    atten_activation='ReLU', 
    attention='add', 
    output_activation='Sigmoid',
    batch_norm=True,
    pool=False,
    unpool=False,
    backbone='VGG16', 
    weights='imagenet', 
    freeze_backbone=True,
    freeze_batch_norm=True,
    name='attunet'
)    

def pretrained_transunet_(input_shape=(512, 512)):
    return pretrained_models.transunet_2d(
    input_size=(*input_shape, 3), 
    filter_num=[64, 128, 256, 512], 
    n_labels=1, 
    stack_num_down=2, 
    stack_num_up=2, 
    embed_dim=768, 
    num_heads=12, 
    num_transformer=12, 
    num_mlp=3072, 
    activation='ReLU', 
    output_activation='Sigmoid', 
    batch_norm=True, 
    pool='max', 
    unpool='bilinear', 
    name='transunet',
    backbone='VGG16',
    freeze_backbone=True,
    freeze_batch_norm=True,
    weights='imagenet'
)

from tensorflow.keras.layers import Lambda
import tensorflow as tf
from keras_unet_collection import models as pretrained_models

def pretrained_transunet__(input_shape=(512, 512)):
    # Create the transunet model
    model = pretrained_models.transunet_2d(
        input_size=(*input_shape, 3),
        filter_num=[64, 128, 256, 512],
        n_labels=1,
        stack_num_down=2,
        stack_num_up=2,
        embed_dim=768,
        num_heads=12,
        num_transformer=12,
        num_mlp=3072,
        activation='ReLU',
        output_activation='Sigmoid',
        batch_norm=True,
        pool='max',
        unpool='bilinear',
        name='transunet',
        backbone='VGG16',
        freeze_backbone=True,
        freeze_batch_norm=True,
        weights='imagenet'
    )

    # Wrap the model in a Lambda layer to handle reshape or other TF functions
    output = Lambda(lambda x: tf.reshape(x, (-1, *x.shape[1:])))(model.output)
    return tf.keras.Model(inputs=model.input, outputs=output)



from tensorflow.keras.layers import Layer
from tensorflow.keras import Model, Input
import tensorflow as tf
from keras_unet_collection import models as pretrained_models

class ReshapeLayer(Layer):
    def __init__(self, target_shape, **kwargs):
        super(ReshapeLayer, self).__init__(**kwargs)
        self.target_shape = target_shape

    def call(self, inputs):
        return tf.reshape(inputs, self.target_shape)

def pretrained_transunet(input_shape=(512, 512)):
    # Create the transunet model
    base_model = pretrained_models.transunet_2d(
        input_size=(*input_shape, 3),
        filter_num=[64, 128, 256, 512],
        n_labels=1,
        stack_num_down=2,
        stack_num_up=2,
        embed_dim=768,
        num_heads=12,
        num_transformer=12,
        num_mlp=3072,
        activation='ReLU',
        output_activation='Sigmoid',
        batch_norm=True,
        pool='max',
        unpool='bilinear',
        name='transunet',
        backbone='VGG16',
        freeze_backbone=True,
        freeze_batch_norm=True,
        weights='imagenet'
    )

    # Define the output shape (replace with your actual target shape)
    target_shape = (-1, 16, 16, 768)  # Update this based on your requirements

    # Apply the custom Reshape layer to the model output
    reshaped_output = ReshapeLayer(target_shape=target_shape)(base_model.output)

    # Create the final model
    model = Model(inputs=base_model.input, outputs=reshaped_output)
    return model