import tensorflow as tf
from keras.layers import ZeroPadding2D,Conv2D,BatchNormalization,MaxPooling2D,UpSampling2D,Input
from keras.models import load_model,Model

def yolo_conv(prev, k, s, p, c):
    x1 = ZeroPadding2D(padding=(p, p))(prev)
    x1 = Conv2D(kernel_size=k, filters=c, strides=s, padding="valid",activation=tf.nn.leaky_relu)(x1)
    x1 = BatchNormalization()(x1)
    x1 = tf.nn.silu(x1)
    return x1

def yolo_bottleneck(prev,shortcut=True):
    _,_,_,d = prev.shape
    x1 = yolo_conv(prev=prev,k=1,s=1,p=0,c=d)
    x2 = yolo_conv(prev=x1,k=1,s=1,p=0,c=d)
    if shortcut:
        op = prev + x2
    else:
        op = x2
    return op

def yolo_c2f(prev,n,c,boolean_lambda=True):
    x1 = yolo_conv(prev,k=1,s=1,p=0,c=c)
    new_cout = c // 2
    output_1 = x1[:, :, :, :new_cout]
    output_2 = x1[:, :, :, new_cout:]
    concat_layer = tf.concat([output_1,output_2], axis=3)
    loop = output_2
    for i in range(n):
        loop = yolo_bottleneck(prev=loop,shortcut=boolean_lambda)
        concat_layer = tf.concat([concat_layer,loop],axis=3)
    op = yolo_conv(prev=concat_layer,k=1,s=1,p=0,c=c)
    return op

def yolo_sppf(prev,c):
    op = yolo_conv(prev=prev,k=1,s=1,p=0,c=c)
    x1 = MaxPooling2D()(op)
    x2 = MaxPooling2D()(x1)
    _,bp,cp,dp = op.shape
    _,b1,c1,d1 = x1.shape
    _,b2,c2,_ = x2.shape
    prev_reshape = tf.reshape(prev,shape=[-1,b2,c2,int((bp*cp*dp)/(b2*c2))])
    x1_reshape = tf.reshape(x1,shape=[-1,b2,c2,int((b1*c1*d1)/(b2*c2))])
    concatenated_tensor = tf.concat([prev_reshape, x1_reshape, x2], axis=3)
    final_op = yolo_conv(concatenated_tensor,k=1,s=1,p=0,c=c)
    return final_op

def get_model_fn(num_classes=0,num_anchors=0,model_path=None):
    if model_path is not None:
        model = load_model(model_path)
        return model
    
    if num_classes<=0:
        raise ValueError("invalid number of classes, classes should be greater than 0")
    if num_anchors<=0:
        raise ValueError("Invalid number of anchors, anchors should be greater than 0")
    
    act = "sigmoid"
    if num_classes>1:
        act = "softmax"
    input = Input(shape=(224,224,3))
    P0 = yolo_conv(prev=input,k=3,s=2,p=1,c=64)
    P1 = yolo_conv(prev=P0,k=3,s=2,p=1,c=128)
    P2 = yolo_c2f(prev=P1,n=3,c=128,boolean_lambda=True)
    P3 = yolo_conv(prev=P2,k=3,s=2,p=1,c=256)
    P4 = yolo_c2f(prev=P3,n=6,c=256,boolean_lambda=True)
    P5 = yolo_conv(prev=P4,k=3,s=2,p=1,c=512)
    P6 = yolo_c2f(prev=P5,n=3,c=512,boolean_lambda=True)
    P7 = yolo_conv(prev=P6,k=3,s=2,p=1,c=768)
    P8 = yolo_c2f(prev=P7,n=3,c=1024,boolean_lambda=True)
    P_extra = UpSampling2D(size=(4,4))(P8)
    P9 = yolo_sppf(prev=P_extra,c=1024)
    P10 = UpSampling2D(size=(2,2))(P9)
    P11 = tf.concat([P6,P10],axis=3)
    P12 = yolo_c2f(prev=P11,n=3,c=512,boolean_lambda=False)
    P13 = UpSampling2D(size=(2,2))(P12)
    P14 = tf.concat([P4,P13],axis=3)
    P15 = yolo_c2f(prev=P14,n=3,c=256,boolean_lambda=False)
    P16 = yolo_conv(prev=P15,k=3,s=2,p=1,c=256)
    P17 = tf.concat([P12,P16],axis=3)
    P18 = yolo_c2f(prev=P17,n=3,c=512,boolean_lambda=False)
    P19 = yolo_conv(prev=P18,k=3,s=2,p=1,c=512)
    P20 = tf.concat([P9,P19],axis=3)

    D1 = yolo_conv(prev=P15,k=1,s=1,p=0,c=(num_classes+5)*num_anchors)
    D2 = yolo_conv(prev=P18,k=1,s=1,p=0,c=(num_classes+5)*num_anchors)
    D3 = yolo_conv(prev=P20,k=1,s=1,p=0,c=(num_classes+5)*num_anchors)

    model = Model(inputs=input,outputs = {'y1':D1,'y2':D2,'y3':D3})
    return model
