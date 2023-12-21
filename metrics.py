import tensorflow as tf
from collections import Counter

def intersection_over_union(boxes_preds, boxes_labels, box_format="xywh",epsilon = 1e-6):
    if box_format == "xywh":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    elif box_format == "xyxy":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]
    
    else:
        raise TypeError("Invalid box_format type")

    x1 = tf.maximum(box1_x1, box2_x1)
    y1 = tf.maximum(box1_y1, box2_y1)
    x2 = tf.minimum(box1_x2, box2_x2)
    y2 = tf.minimum(box1_y2, box2_y2)

    intersection = tf.maximum(0.0, x2 - x1) * tf.maximum(0.0, y2 - y1)
    box1_area = tf.abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = tf.abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + epsilon)

# box1 = tf.convert_to_tensor([
#                 [0, 0, 2, 2],
#                 [0, 0, 2, 2],
#                 [0, 0, 2, 2],
#                 [0, 0, 2, 2],
#                 [0, 0, 2, 2],
#                 [0, 0, 3, 2],
#             ],dtype=tf.float32)
# box2 = tf.convert_to_tensor([
#                 [3, 0, 5, 2],
#                 [3, 0, 5, 2],
#                 [0, 3, 2, 5],
#                 [2, 0, 5, 2],
#                 [1, 1, 3, 3],
#                 [1, 1, 3, 3],
#             ],dtype=tf.float32)
# intersection_over_union(boxes_preds=box1,boxes_labels=box2,box_format="xyxy")

def non_max_suppression(bboxes,iou_threshold,nms_threshold,box_format="xyxy"):
    # predictions = [[1,0.9,x1,y1,x2,y2]]
    bboxes = [box for box in bboxes if box[1]>nms_threshold]
    bboxes_after_nms = []
    bboxes = sorted(bboxes,key=lambda x:x[1],reverse=True)

    while bboxes:
        chosen_box = bboxes.pop(0)
        bboxes = [box 
                  for box in bboxes
                  if box[0] !=chosen_box[0]
                  or intersection_over_union(
                      tf.convert_to_tensor(chosen_box[2:]),
                      tf.convert_to_tensor(box[2:]),
                      box_format=box_format
                    ) < iou_threshold
                  ]
        bboxes_after_nms.append(chosen_box)
    
    return bboxes_after_nms

# non_max_suppression(bboxes=[
#             [1, 1, 0.5, 0.45, 0.4, 0.5],
#             [1, 0.8, 0.5, 0.5, 0.2, 0.4],
#             [1, 0.7, 0.25, 0.35, 0.3, 0.1],
#             [1, 0.05, 0.1, 0.1, 0.1, 0.1],
#         ],iou_threshold=0.35,nms_threshold=0.2,box_format="xywh")

@tf.function
def mAP(pred_boxes,true_boxes,num_classes,iou_threshold=0.5,box_format="xyxy",epsilon=1e-6):
    # pred_boxes (list): [[train_idx,class_pred,prob_score,x1,y1,x2,y2],[],[]]

    average_precision=[]
    for c in range(num_classes):
        detections=[]
        ground_truths=[]
        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)
        
        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)
        
        # img 0 has 3 bboxes
        # img 1 has 5 bboxes
        # amount_bboxes = {0:3, 1:5}
        amount_bboxes = Counter(gt[0] for gt in ground_truths)

        for key,val in amount_bboxes.items():
            amount_bboxes[key] = tf.zeros(val)
        
        # amount_boxes = {0:tf.tensor([0,0,0]),1:tf.tensor([0,0,0,0,0])}
            detection = sorted(detections,key=lambda x:x[2],reverse=True)
            TP = tf.zeros((len(detections)))
            FP = tf.zeros((len(detections)))
            total_true_boxes = len(ground_truths)
        
        for detection_idx, detection in enumerate(detections):
            ground_truth_img = [bbox for bbox in ground_truths if bbox[0] == detection[0]]

            best_iou = 0.0
            best_gt_idx = -1
            for idx, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(
                    tf.convert_to_tensor(detection[3:],dtype=tf.float32),
                    tf.convert_to_tensor(gt[3:],dtype=tf.float32),
                    box_format=box_format
                )
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx
            
            if best_iou > iou_threshold:
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    TP = tf.tensor_scatter_nd_update(TP, [[detection_idx]], [1])
                    amount_bboxes[detection[0]] = tf.tensor_scatter_nd_update(amount_bboxes[detection[0]], [[best_gt_idx]], [1])
                else:
                    FP = tf.tensor_scatter_nd_update(FP, [[detection_idx]], [1])
            else:
                FP = tf.tensor_scatter_nd_update(FP, [[detection_idx]], [1])
                
        TP_cumsum = tf.cumsum(TP,axis=0)
        FP_cumsum = tf.cumsum(FP,axis=0)
        recalls = TP_cumsum / (total_true_boxes + epsilon)
        precisions = tf.divide(TP_cumsum,(TP_cumsum + FP_cumsum + epsilon))
        precisions = tf.concat([tf.convert_to_tensor([1], dtype=tf.float32), precisions], axis=0)
        recalls = tf.concat([tf.convert_to_tensor([0], dtype=tf.float32), recalls], axis=0)


        def trapz(x, y):
            dx = x[1:] - x[:-1]
            return tf.reduce_sum((y[:-1] + y[1:]) * dx / 2)

        average_precision.append(trapz(recalls,precisions))
    
    return sum(average_precision) / len(average_precision)

# pred_boxes = [
#             [0, 0, 0.9, 0.55, 0.2, 0.3, 0.2],
#             [0, 0, 0.8, 0.35, 0.6, 0.3, 0.2],
#             [0, 0, 0.7, 0.8, 0.7, 0.2, 0.2],
#         ]
# true_boxes = [
#             [0, 0, 0.9, 0.55, 0.2, 0.3, 0.2],
#             [0, 0, 0.8, 0.35, 0.6, 0.3, 0.2],
#             [0, 0, 0.7, 0.8, 0.7, 0.2, 0.2],
#         ]
# mAP(pred_boxes=pred_boxes,true_boxes=true_boxes,iou_threshold=0.5,box_format="xywh",num_classes=5)