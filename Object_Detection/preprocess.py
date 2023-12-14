import tensorflow as tf
import numpy as np

def grid_starting_position(grid_no,total_grids):
    grid_h,grid_w=grid_no
    if grid_w>total_grids or grid_h>total_grids:
        raise ValueError("Grid number should not be greater than total grids")
    Cx = (1/total_grids)*(grid_w-1)
    Cy = (1/total_grids)*(grid_h-1)
    return Cx,Cy


def get_tw_and_th_wrt_anchor_box(anchor_box_size_norm,bbox_w_and_h):
    bbox_w,bbox_h=bbox_w_and_h
    anchor_box_wnorm,anchor_box_hnorm = anchor_box_size_norm
    return bbox_w/anchor_box_wnorm,bbox_h/anchor_box_hnorm

def find_grid_no(yolo_x_and_y,total_grids):
    x,y = yolo_x_and_y
    return int(y*total_grids)+1,int(x*total_grids)+1

def get_tx_and_ty_wrt_grid(yolo_x_and_y,grid_l,Cx,Cy):
    x,y=yolo_x_and_y
    dx,dy=x-Cx,y-Cy
    tx,ty = dx/grid_l,dy/grid_l
    return (tx,ty)

def convert_to_yolo_bbox_format(yolo_box,image_size,anchor_box_size,total_grids):
    grid_no = find_grid_no(yolo_x_and_y=yolo_box[1:3],total_grids=total_grids)
    Cx,Cy = grid_starting_position(grid_no=grid_no,total_grids=total_grids)
    tx,ty = get_tx_and_ty_wrt_grid(yolo_x_and_y=yolo_box[1:3],grid_l=1/total_grids,Cx=Cx,Cy=Cy)
    tw,th=get_tw_and_th_wrt_anchor_box(anchor_box_size_norm=(anchor_box_size[0]/image_size,anchor_box_size[1]/image_size),bbox_w_and_h=yolo_box[3:])
    return [tx,ty,tw,th]

@tf.function
def final_yolo_output_for_an_anchor(total_classes,yolo_box,image_size,anchor_box_size,total_grids):
    bbox = convert_to_yolo_bbox_format(yolo_box=yolo_box,image_size=image_size,anchor_box_size=anchor_box_size,total_grids=total_grids)
    class_prob = np.zeros(total_classes)
    class_prob[int(yolo_box[0])]=1.0
    return tf.concat([tf.constant([1], dtype=tf.float32),bbox,tf.constant(class_prob, dtype=tf.float32)], axis=0)

def read_yolo_box_from_txt_file(path):
    numbers_list = []
    with open(path, 'r') as file:
        for line in file:
            numbers = list(map(float, line.strip().split()))
            numbers_list.append(numbers)
    return numbers_list

@tf.function
def get_final_output_for_an_image_per_grid_shape(path, total_classes, image_size, anchors, total_grids):
    output = tf.zeros(shape=(total_grids, total_grids, (5 + total_classes) * len(anchors)), dtype=tf.float32)
    yolo_box = read_yolo_box_from_txt_file(path=path)
    for box in yolo_box:
        a, b = find_grid_no(yolo_x_and_y=box[1:3], total_grids=total_grids)
        op = []
        for anchor in anchors:
            op.append(final_yolo_output_for_an_anchor(
                total_classes=total_classes,
                yolo_box=box,
                image_size=image_size,
                anchor_box_size=anchor,
                total_grids=total_grids
            ))

        updates = tf.expand_dims(tf.concat(op, axis=-1), axis=0)
        indices = tf.expand_dims(tf.convert_to_tensor([a, b]), axis=0)
        output = tf.tensor_scatter_nd_update(output, indices, updates)
    return output