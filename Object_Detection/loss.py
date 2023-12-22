import tensorflow as tf
from .utils import intersection_over_union
from keras.losses import Loss,MeanSquaredError,BinaryCrossentropy,SparseCategoricalCrossentropy

class YoloLoss(Loss):
    def __init__(self,anchors_norm):
        super().__init__()
        self.mse = MeanSquaredError()
        self.bce = BinaryCrossentropy(from_logits=True)
        self.entropy = SparseCategoricalCrossentropy(from_logits=True)
        self.sigmoid = tf.keras.activations.sigmoid
        self.anchors = anchors_norm

        # Constants signifying how much to pay for each respective part of the loss
        self.lambda_class = 1
        self.lambda_noobj = 10
        self.lambda_obj = 1
        self.lambda_box = 10
        self.num_anchors=anchors_norm.shape[0]

    def transform_to_bbox(self,input_tensor,anchor_box_size):
        x_and_y = tf.sigmoid(input_tensor[...,0:2])
        w_and_h = tf.multiply(tf.exp(input_tensor[...,2:4]),anchor_box_size)
        return tf.concat([x_and_y,w_and_h],axis=-1)
    
    def no_obj_loss(self,y_true,y_pred):
        mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM)
        noobj = tf.equal(y_true[...,0],0.0)
        return mse(y_true[...,0:1][noobj],y_pred[...,0:1][noobj])

    def class_loss(self,y_true,y_pred):
        obj = tf.equal(y_true[...,0],1.0)
        softmax_tensor2 = tf.nn.softmax(y_pred[5:][obj])
        cross_entropy = tf.reduce_sum(-1.0*y_true[...,5:][obj] * tf.math.log(1e-16+softmax_tensor2))
        return cross_entropy
    
    def box_loss(self,y_true,y_pred,anchor_box_size):
        mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM)
        obj = tf.equal(y_true[...,0],1.0)
        box_preds = self.transform_to_bbox(input_tensor=y_pred[...,1:5][obj],anchor_box_size=anchor_box_size)
        return mse(y_true[...,1:5],box_preds)
    
    def obj_loss(self,y_true,y_pred,anchor_box_size):
        mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM)
        obj = tf.equal(y_true[...,0],1.0)
        box_preds = self.transform_to_bbox(input_tensor=y_pred[...,1:5][obj],anchor_box_size=anchor_box_size)
        ious = intersection_over_union(boxes_preds=box_preds,boxes_labels=y_true[...,1:5][obj])
        ious = tf.stop_gradient(ious)
        object_loss = mse(tf.sigmoid(y_pred[..., 0:1][obj]), ious * y_true[..., 0:1][obj])
        return object_loss
    
    @tf.function
    def total_loss(self,y_true,y_pred,anchor_box_size):
        return self.no_obj_loss(y_true=y_true,y_pred=y_pred) + self.class_loss(y_true=y_true,y_pred=y_pred) + self.box_loss(y_true=y_true,y_pred=y_pred,anchor_box_size=anchor_box_size) + self.obj_loss(y_true=y_true,y_pred=y_pred,anchor_box_size=anchor_box_size)
        
    @tf.function
    def call(self, y_true, y_pred):
        pred = tf.split(y_pred, num_or_size_splits=self.num_anchors, axis=-1)
        truth = tf.split(y_true, num_or_size_splits=self.num_anchors, axis=-1)
        loss = 0.0
        for i in range(len(pred)):
            loss = loss + self.total_loss(y_pred=pred[i],y_true=truth[i],anchor_box_size=self.anchors[i])
        return loss
