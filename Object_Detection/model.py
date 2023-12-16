import tensorflow as tf
from .layer import *
from keras.layers import Input,UpSampling2D
from keras.models import Model,load_model


def get_model(model_path=None):
    if model_path is not None:
        custom_objects = {'YoloConvLayer': YoloConvLayer,'YoloC2FLayer': YoloC2FLayer,'YoloSPPFLayer': YoloSPPFLayer,'YoloBottleneckLayer': YoloBottleneckLayer}
        model = load_model("model.keras",custom_objects=custom_objects)
        return model
    input = Input(shape=(224,224,3))
    P0 = YoloConvLayer(k=3,s=2,p=1,c=64)(input)
    P1 = YoloConvLayer(k=3,s=2,p=1,c=128)(P0)
    P2 = YoloC2FLayer(n=3,c=128,boolean_lambda=True)(P1)
    P3 = YoloConvLayer(k=3,s=2,p=1,c=256)(P2)
    P4 = YoloC2FLayer(n=6,c=256,boolean_lambda=True)(P3)
    P5 = YoloConvLayer(k=3,s=2,p=1,c=512)(P4)
    P6 = YoloC2FLayer(n=3,c=512,boolean_lambda=True)(P5)
    P7 = YoloConvLayer(k=3,s=2,p=1,c=768)(P6)
    P8 = YoloC2FLayer(n=3,c=1024,boolean_lambda=True)(P7)
    P_extra = UpSampling2D(size=(4,4))(P8)
    P9 = YoloSPPFLayer(c0=1024)(P_extra)
    P10 = UpSampling2D(size=(2,2))(P9)
    P11 = tf.concat([P6,P10],axis=3)
    P12 = YoloC2FLayer(n=3,c=512,boolean_lambda=False)(P11)
    P13 = UpSampling2D(size=(2,2))(P12)
    P14 = tf.concat([P4,P13],axis=3)
    P15 = YoloC2FLayer(n=3,c=256,boolean_lambda=False)(P14)
    D1 = YoloConvLayer(k=1,s=1,p=0,c=18,name="y1")(P15)
    P16 = YoloConvLayer(k=3,s=2,p=1,c=256)(P15)
    P17 = tf.concat([P12,P16],axis=3)
    P18 = YoloC2FLayer(n=3,c=512,boolean_lambda=False)(P17)
    D2 = YoloConvLayer(k=1,s=1,p=0,c=18,name="y2")(P18)
    P19 = YoloConvLayer(k=3,s=2,p=1,c=512)(P18)
    P20 = tf.concat([P9,P19],axis=3)
    D3 = YoloConvLayer(k=1,s=1,p=0,c=18,name="y3")(P20)
    model = Model(inputs=input,outputs = [D1,D2,D3])
    return model