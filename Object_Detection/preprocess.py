import tensorflow as tf
import numpy as np

def grid_starting_position(grid_no, total_grids):
    grid_w, grid_h = tf.unstack(tf.cast(grid_no, tf.float32))
    total_grids = tf.cast(total_grids, tf.float32)
    if tf.reduce_any(grid_w > total_grids) or tf.reduce_any(grid_h > total_grids):
        raise ValueError("Grid number should not be greater than total grids")
    Cx = (1.0 / total_grids) * grid_w
    Cy = (1.0 / total_grids) * grid_h
    return Cx, Cy


def get_tw_and_th_wrt_anchor_box(anchor_box_size_norm, bbox_w_and_h):
    anchor_box_wnorm, anchor_box_hnorm = anchor_box_size_norm
    bbox_w, bbox_h = bbox_w_and_h
    tw = tf.divide(bbox_w, anchor_box_wnorm)
    th = tf.divide(bbox_h, anchor_box_hnorm)
    return tw, th

def find_grid_no(yolo_x_and_y, total_grids):
    x, y = yolo_x_and_y
    grid_x = tf.cast(tf.floor(x*total_grids), tf.int32)
    grid_y = tf.cast(tf.floor(y*total_grids), tf.int32)
    return grid_x, grid_y

def get_tx_and_ty_wrt_grid(yolo_x_and_y,grid_l,Cx,Cy):
    x,y=yolo_x_and_y
    dx,dy=x-Cx,y-Cy
    tx,ty = dx/grid_l,dy/grid_l
    return (tx,ty)

def convert_to_yolo_bbox_format(yolo_box, image_size, anchor_box_size, total_grids):
    grid_no = find_grid_no(yolo_box[1:3], total_grids)
    Cx, Cy = grid_starting_position(grid_no, total_grids)
    tx, ty = get_tx_and_ty_wrt_grid(yolo_box[1:3], 1 / total_grids, Cx, Cy)
    tw, th = get_tw_and_th_wrt_anchor_box((anchor_box_size[0] / image_size, anchor_box_size[1] / image_size), yolo_box[3:])
    tx = tf.convert_to_tensor(tx, dtype=tf.float32)
    ty = tf.convert_to_tensor(ty, dtype=tf.float32)
    tw = tf.convert_to_tensor(tw, dtype=tf.float32)
    th = tf.convert_to_tensor(th, dtype=tf.float32)
    bbox_tensor = tf.stack([tx, ty, tw, th], axis=0)
    return bbox_tensor

def final_yolo_output_for_an_anchor(total_classes, yolo_box, image_size, anchor_box_size, total_grids):
    bbox = convert_to_yolo_bbox_format(yolo_box=yolo_box, image_size=image_size, anchor_box_size=anchor_box_size, total_grids=total_grids)
    class_prob = np.zeros(total_classes)
    class_prob[int(yolo_box[0])] = 1.0
    bbox = tf.convert_to_tensor(bbox, dtype=tf.float32)
    class_prob = tf.convert_to_tensor(class_prob, dtype=tf.float32)
    return tf.concat([tf.constant([1], dtype=tf.float32), bbox, class_prob], axis=0)

def read_yolo_box_from_txt_file(path):
    numbers_list = []
    if tf.is_tensor(path):
        path = path.numpy()
    with tf.io.gfile.GFile(path, 'r') as file:
        for line in file:
            numbers = list(map(float, line.strip().split()))
            numbers_list.append(numbers)
    return np.array(numbers_list)

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
    return tf.convert_to_tensor(output)

# used to generate y_train for a paarticular grid size
def get_y(path,total_classes, image_size, anchors, grid_size):
    file = tf.strings.regex_replace(path, '\\..+$', '.txt')
    return get_final_output_for_an_image_per_grid_shape(path=file, total_classes=total_classes, image_size=image_size, anchors=anchors, total_grids=grid_size)

# used to generate x_train
def get_x(path,image_size):
    image = tf.io.read_file(path)
    image = tf.image.decode_image(image, channels=3)
    resized_image = tf.image.resize(image, (int(image_size), int(image_size)))
    return tf.cast(resized_image, tf.float32) / 255.0

def list_files_using_path(base_path):
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
    file_list = tf.io.gfile.glob(base_path + '/*')
    image_files = [file for file in file_list if any(file.lower().endswith(ext) for ext in image_extensions)]
    return image_files

# main function that generates dataset
def generate_dataset(base_path, total_classes, image_size, anchors, batch_size=32):
    grid_sizes=tf.convert_to_tensor([28.0,14.0,7.0])
    paths = list_files_using_path(base_path)

    x_train = tf.convert_to_tensor([get_x(path=path,image_size=image_size) for path in paths])
    y1_train = tf.convert_to_tensor([get_y(path=path,total_classes=total_classes, image_size=image_size, anchors=anchors, grid_size=grid_sizes[0]) for path in paths])
    y2_train = tf.convert_to_tensor([get_y(path=path,total_classes=total_classes, image_size=image_size, anchors=anchors, grid_size=grid_sizes[1]) for path in paths])
    y3_train = tf.convert_to_tensor([get_y(path=path,total_classes=total_classes, image_size=image_size, anchors=anchors, grid_size=grid_sizes[2]) for path in paths])
    dataset = tf.data.Dataset.from_tensor_slices((x_train, {'y1': y1_train, 'y2': y2_train, 'y3': y3_train}))

    dataset = dataset.shuffle(buffer_size=len(x_train)).batch(batch_size)
    return dataset