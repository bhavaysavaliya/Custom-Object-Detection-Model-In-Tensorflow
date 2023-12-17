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

    
    P15_temp = P15
    layer_1 = []
    for i in range(num_anchors):
        P15_temp = Conv2D(kernel_size=1, strides=1, padding="valid", filters=1, activation="sigmoid", name=f"1{i+1}1")(P15_temp)
        layer_1.append(P15_temp)
        P15_temp = Conv2D(kernel_size=1, strides=1, padding="valid", filters=4, activation="relu", name=f"1{i+1}2")(P15_temp)
        layer_1.append(P15_temp)
        P15_temp = Conv2D(kernel_size=1, strides=1, padding="valid", filters=num_classes, activation=act, name=f"1{i+1}3")(P15_temp)
        layer_1.append(P15_temp)
    concatenate_1 = tf.keras.layers.Concatenate(axis=3)(layer_1)
    D1 = Lambda(lambda x: x, name="y1")(concatenate_1)


    P18_temp = P18
    layer_2 = []
    for i in range(num_anchors):
        P18_temp = Conv2D(kernel_size=1, strides=1, padding="valid", filters=1, activation="sigmoid", name=f"2{i+1}1")(P18_temp)
        layer_2.append(P18_temp)
        P18_temp = Conv2D(kernel_size=1, strides=1, padding="valid", filters=4, activation="relu", name=f"2{i+1}2")(P18_temp)
        layer_2.append(P18_temp)
        P18_temp = Conv2D(kernel_size=1, strides=1, padding="valid", filters=num_classes, activation=act, name=f"2{i+1}3")(P18_temp)
        layer_2.append(P18_temp)
    concatenate_2 = tf.keras.layers.Concatenate(axis=3)(layer_2)
    D2 = Lambda(lambda x: x, name="y2")(concatenate_2)


    P20_temp = P20
    layer_3 = []
    for i in range(num_anchors):
        P20_temp = Conv2D(kernel_size=1, strides=1, padding="valid", filters=1, activation="sigmoid", name=f"3{i+1}1")(P20_temp)
        layer_3.append(P20_temp)
        P20_temp = Conv2D(kernel_size=1, strides=1, padding="valid", filters=4, activation="relu", name=f"3{i+1}2")(P20_temp)
        layer_3.append(P20_temp)
        P20_temp = Conv2D(kernel_size=1, strides=1, padding="valid", filters=num_classes, activation=act, name=f"3{i+1}3")(P20_temp)
        layer_3.append(P20_temp)
    concatenate_3 = tf.keras.layers.Concatenate(axis=3)(layer_3)
    D3 = Lambda(lambda x: x, name="y3")(concatenate_3)


    model = Model(inputs=input,outputs = [D1,D2,D3])
    return model

def predict(model,image):
    img = tf.expand_dims(tf.convert_to_tensor(image,dtype=tf.float32)/255.0,axis=0)
    op0,op1,op2 = model.predict(img)
    op0 = tf.squeeze(op0)
    op1 = tf.squeeze(op1)
    op2 = tf.squeeze(op2)
    return [op0,op1,op2]