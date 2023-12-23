import tensorflow as tf
from .layer import *
from keras.layers import Input,UpSampling2D
from keras.models import Model,load_model


def get_model(num_classes=0,num_anchors=0,model_path=None):
    if model_path is not None:
        custom_objects = {'YoloConvLayer': YoloConvLayer,'YoloC2FLayer': YoloC2FLayer,'YoloSPPFLayer': YoloSPPFLayer,'YoloBottleneckLayer': YoloBottleneckLayer}
        model = load_model("model.keras",custom_objects=custom_objects,safe_mode=False)
        return model
    
    if num_classes<=0:
        raise ValueError("invalid number of classes, classes should be greater than 0")
    if num_anchors<=0:
        raise ValueError("Invalid number of anchors, anchors should be greater than 0")
    
    act = "sigmoid"
    if num_classes>1:
        act = "softmax"
    input = Input(shape=(224,224,3))
    P0 = YoloConvLayer(k=3,s=2,p=1,c=64,name = "P0")(input)
    P1 = YoloConvLayer(k=3,s=2,p=1,c=128,name = "P1")(P0)
    P2 = YoloC2FLayer(n=3,c=128,boolean_lambda=True,name = "P2")(P1)
    P3 = YoloConvLayer(k=3,s=2,p=1,c=256,name = "P3")(P2)
    P4 = YoloC2FLayer(n=6,c=256,boolean_lambda=True,name = "P4")(P3)
    P5 = YoloConvLayer(k=3,s=2,p=1,c=512,name = "P5")(P4)
    P6 = YoloC2FLayer(n=3,c=512,boolean_lambda=True,name = "P6")(P5)
    P7 = YoloConvLayer(k=3,s=2,p=1,c=768,name = "P7")(P6)
    P8 = YoloC2FLayer(n=3,c=1024,boolean_lambda=True,name = "P8")(P7)
    P_extra = UpSampling2D(size=(4,4))(P8)
    P9 = YoloSPPFLayer(c0=1024,name = "P9")(P_extra)
    P10 = UpSampling2D(size=(2,2),name = "P10")(P9)
    P11 = tf.concat([P6,P10],axis=3)
    P12 = YoloC2FLayer(n=3,c=512,boolean_lambda=False,name = "P12")(P11)
    P13 = UpSampling2D(size=(2,2),name = "P13")(P12)
    P14 = tf.concat([P4,P13],axis=3)
    P15 = YoloC2FLayer(n=3,c=256,boolean_lambda=False,name = "P15")(P14)
    P16 = YoloConvLayer(k=3,s=2,p=1,c=256,name = "P16")(P15)
    P17 = tf.concat([P12,P16],axis=3)
    P18 = YoloC2FLayer(n=3,c=512,boolean_lambda=False,name = "P18")(P17)
    P19 = YoloConvLayer(k=3,s=2,p=1,c=512,name = "P19")(P18)
    P20 = tf.concat([P9,P19],axis=3)

    D1 = YoloConvLayer(k=1,s=1,p=0,c=(num_classes+5)*num_anchors,name='y1')(P15)
    D2 = YoloConvLayer(k=1,s=1,p=0,c=(num_classes+5)*num_anchors,name='y2')(P18)
    D3 = YoloConvLayer(k=1,s=1,p=0,c=(num_classes+5)*num_anchors,name='y3')(P20)

    model = Model(inputs=input,outputs = {'y1':D1,'y2':D2,'y3':D3})
    return model

def predict(model,image):
    img = tf.expand_dims(tf.convert_to_tensor(image,dtype=tf.float32)/255.0,axis=0)
    op0,op1,op2 = model.predict(img)
    op0 = tf.squeeze(op0)
    op1 = tf.squeeze(op1)
    op2 = tf.squeeze(op2)
    return [op0,op1,op2]