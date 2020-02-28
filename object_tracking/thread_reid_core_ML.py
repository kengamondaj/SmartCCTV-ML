from __future__ import division, print_function, absolute_import

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import os
import sys
import cv2
import csv
import tensorflow as tf
import tensorflow.contrib.slim as slim

import pandas as pd
import time
from tqdm import tqdm

from collections import defaultdict
from io import StringIO

import argparse
from application_util import preprocessing
from application_util import visualization
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from time import sleep


sys.path.insert(0, os.path.abspath("./"))
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

from math import factorial
from deep_sort.iou_matching import iou
from matplotlib import pyplot as plt
from tqdm import tqdm
from PIL import Image

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import normalize
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression

from mtcnn.mtcnn import MTCNN

# ================================================================================

def detection(sess, video_path):
    points_objs = []
    start = time.time()
    
    skip = 0
    id_frame = 1
    id_center = 0
    
    cap = cv2.VideoCapture(video_path)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    detector = MTCNN()

    pbar = tqdm(total=length)
    while(True):
        ret, frame = cap.read()
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        
        image_np = np.array(frame)
        if(image_np.shape == ()): break

        print('Frame ID:', id_frame, '\tTime:', '{0:.2f}'.format(time.time()-start), 'seconds')

        image_np_expanded = np.expand_dims(image_np, axis=0)

        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        scores = detection_graph.get_tensor_by_name('detection_scores:0')
        classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')

        (boxes, scores, classes, num_detections) = sess.run([boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})
        
        boxes = np.squeeze(boxes)
        classes = np.squeeze(classes).astype(np.int32)
        scores = np.squeeze(scores)

        count_boxes = 0
        thresh = 0.2
        max_boxes = 50
        
        for i, c in enumerate(classes):
            if (c == 1 and (scores[i] > thresh) and (count_boxes < max_boxes)):
                im_height = image_np.shape[0]
                im_width = image_np.shape[1]
                ymin, xmin, ymax, xmax = boxes[i]
                
                (left, right, top, bottom) = (int(xmin*im_width),  int(xmax*im_width),
                                              int(ymin*im_height), int(ymax*im_height))
                
#                 # detect faces in the image
                people = image_np[top:bottom,left:right]
                # people = people[...,::-1]
                face = detector.detect_faces(people)
                
                xF, yF, widthF, heightF = 0,0,0,0
                    
                if len(face) != 0 :
                    if face[0]['confidence'] >= 0.8:
                        xF, yF, widthF, heightF = face[0]['box']
                        xF = int(xF*0.9)
                        yF = int(yF*0.9)
                        widthF = int(widthF*1.2)
                        heightF = int(heightF*1.15)

                        img = people[yF:yF+heightF,xF:xF+widthF]
                        nameF = './face/'+str(id_frame)+'.jpg'
                        cv2.imwrite(nameF,img)
                
                points_objs.append([
                    id_frame, -1,
                    left, top, right-left, bottom-top,
                    scores[i],
                    -1,-1,-1,
                    xF+left, yF+top, widthF, heightF,-1
                ])
                count_boxes += 1

        id_frame += 1
        pbar.update(1)
    pbar.close()

    cap.release()
    cv2.destroyAllWindows()

    # write detection
    return np.array(points_objs)
    # with open(det_path[:-3]+'csv', 'w') as file:
    #     writer = csv.writer(file, lineterminator='\n')
    #     writer.writerows(points_objs)



def _batch_norm_fn(x, scope=None):
    if scope is None:
        scope = tf.get_variable_scope().name + "/bn"
    return slim.batch_norm(x, scope=scope)


def create_link(
        incoming, network_builder, scope, nonlinearity=tf.nn.elu,
        weights_initializer=tf.truncated_normal_initializer(stddev=1e-3),
        regularizer=None, is_first=False, summarize_activations=True):
    if is_first:
        network = incoming
    else:
        network = _batch_norm_fn(incoming, scope=scope + "/bn")
        network = nonlinearity(network)
        if summarize_activations:
            tf.summary.histogram(scope+"/activations", network)

    pre_block_network = network
    post_block_network = network_builder(pre_block_network, scope)

    incoming_dim = pre_block_network.get_shape().as_list()[-1]
    outgoing_dim = post_block_network.get_shape().as_list()[-1]
    if incoming_dim != outgoing_dim:
        assert outgoing_dim == 2 * incoming_dim, \
            "%d != %d" % (outgoing_dim, 2 * incoming)
        projection = slim.conv2d(
            incoming, outgoing_dim, 1, 2, padding="SAME", activation_fn=None,
            scope=scope+"/projection", weights_initializer=weights_initializer,
            biases_initializer=None, weights_regularizer=regularizer)
        network = projection + post_block_network
    else:
        network = incoming + post_block_network
    return network


def create_inner_block(
        incoming, scope, nonlinearity=tf.nn.elu,
        weights_initializer=tf.truncated_normal_initializer(1e-3),
        bias_initializer=tf.zeros_initializer(), regularizer=None,
        increase_dim=False, summarize_activations=True):
    n = incoming.get_shape().as_list()[-1]
    stride = 1
    if increase_dim:
        n *= 2
        stride = 2

    incoming = slim.conv2d(
        incoming, n, [3, 3], stride, activation_fn=nonlinearity, padding="SAME",
        normalizer_fn=_batch_norm_fn, weights_initializer=weights_initializer,
        biases_initializer=bias_initializer, weights_regularizer=regularizer,
        scope=scope + "/1")
    if summarize_activations:
        tf.summary.histogram(incoming.name + "/activations", incoming)

    incoming = slim.dropout(incoming, keep_prob=0.6)

    incoming = slim.conv2d(
        incoming, n, [3, 3], 1, activation_fn=None, padding="SAME",
        normalizer_fn=None, weights_initializer=weights_initializer,
        biases_initializer=bias_initializer, weights_regularizer=regularizer,
        scope=scope + "/2")
    return incoming


def residual_block(incoming, scope, nonlinearity=tf.nn.elu,
                   weights_initializer=tf.truncated_normal_initializer(1e3),
                   bias_initializer=tf.zeros_initializer(), regularizer=None,
                   increase_dim=False, is_first=False,
                   summarize_activations=True):

    def network_builder(x, s):
        return create_inner_block(
            x, s, nonlinearity, weights_initializer, bias_initializer,
            regularizer, increase_dim, summarize_activations)

    return create_link(
        incoming, network_builder, scope, nonlinearity, weights_initializer,
        regularizer, is_first, summarize_activations)


def _create_network(incoming, num_classes, reuse=tf.AUTO_REUSE, l2_normalize=True,
                   create_summaries=True, weight_decay=1e-8):
    nonlinearity = tf.nn.elu
    conv_weight_init = tf.truncated_normal_initializer(stddev=1e-3)
    conv_bias_init = tf.zeros_initializer()
    conv_regularizer = slim.l2_regularizer(weight_decay)
    fc_weight_init = tf.truncated_normal_initializer(stddev=1e-3)
    fc_bias_init = tf.zeros_initializer()
    fc_regularizer = slim.l2_regularizer(weight_decay)

    def batch_norm_fn(x):
        return slim.batch_norm(x, scope=tf.get_variable_scope().name + "/bn")

    network = incoming
    network = slim.conv2d(
        network, 32, [3, 3], stride=1, activation_fn=nonlinearity,
        padding="SAME", normalizer_fn=batch_norm_fn, scope="conv1_1",
        weights_initializer=conv_weight_init, biases_initializer=conv_bias_init,
        weights_regularizer=conv_regularizer)
    if create_summaries:
        tf.summary.histogram(network.name + "/activations", network)
        tf.summary.image("conv1_1/weights", tf.transpose(
            slim.get_variables("conv1_1/weights:0")[0], [3, 0, 1, 2]),
                         max_images=128)
    network = slim.conv2d(
        network, 32, [3, 3], stride=1, activation_fn=nonlinearity,
        padding="SAME", normalizer_fn=batch_norm_fn, scope="conv1_2",
        weights_initializer=conv_weight_init, biases_initializer=conv_bias_init,
        weights_regularizer=conv_regularizer)
    if create_summaries:
        tf.summary.histogram(network.name + "/activations", network)

    network = slim.max_pool2d(network, [3, 3], [2, 2], scope="pool1")

    network = residual_block(
        network, "conv2_1", nonlinearity, conv_weight_init, conv_bias_init,
        conv_regularizer, increase_dim=False, is_first=True,
        summarize_activations=create_summaries)
    network = residual_block(
        network, "conv2_3", nonlinearity, conv_weight_init, conv_bias_init,
        conv_regularizer, increase_dim=False,
        summarize_activations=create_summaries)

    network = residual_block(
        network, "conv3_1", nonlinearity, conv_weight_init, conv_bias_init,
        conv_regularizer, increase_dim=True,
        summarize_activations=create_summaries)
    network = residual_block(
        network, "conv3_3", nonlinearity, conv_weight_init, conv_bias_init,
        conv_regularizer, increase_dim=False,
        summarize_activations=create_summaries)

    network = residual_block(
        network, "conv4_1", nonlinearity, conv_weight_init, conv_bias_init,
        conv_regularizer, increase_dim=True,
        summarize_activations=create_summaries)
    network = residual_block(
        network, "conv4_3", nonlinearity, conv_weight_init, conv_bias_init,
        conv_regularizer, increase_dim=False,
        summarize_activations=create_summaries)

    feature_dim = network.get_shape().as_list()[-1]
    print("feature dimensionality: ", feature_dim)
    network = slim.flatten(network)

    network = slim.dropout(network, keep_prob=0.6)
    network = slim.fully_connected(
        network, feature_dim, activation_fn=nonlinearity,
        normalizer_fn=batch_norm_fn, weights_regularizer=fc_regularizer,
        scope="fc1", weights_initializer=fc_weight_init,
        biases_initializer=fc_bias_init)

    features = network

    if l2_normalize:
        # Features in rows, normalize axis 1.
        features = slim.batch_norm(features, scope="ball", reuse=reuse)
        feature_norm = tf.sqrt(
            tf.constant(1e-8, tf.float32) +
            tf.reduce_sum(tf.square(features), [1], keep_dims=True))
        features = features / feature_norm

        with slim.variable_scope.variable_scope("ball", reuse=reuse):
            weights = slim.model_variable(
                "mean_vectors", (feature_dim, num_classes),
                initializer=tf.truncated_normal_initializer(stddev=1e-3),
                regularizer=None)
            scale = slim.model_variable(
                "scale", (num_classes, ), tf.float32,
                tf.constant_initializer(0., tf.float32), regularizer=None)
            if create_summaries:
                tf.summary.histogram("scale", scale)

            scale = tf.nn.softplus(scale)

        # Each mean vector in columns, normalize axis 0.
        weight_norm = tf.sqrt(
            tf.constant(1e-8, tf.float32) +
            tf.reduce_sum(tf.square(weights), [0], keep_dims=True))
        logits = scale * tf.matmul(features, weights / weight_norm)

    else:
        logits = slim.fully_connected(
            features, num_classes, activation_fn=None,
            normalizer_fn=None, weights_regularizer=fc_regularizer,
            scope="softmax", weights_initializer=fc_weight_init,
            biases_initializer=fc_bias_init)

    return features, logits


def _network_factory(num_classes, is_training, weight_decay=1e-8):

    def factory_fn(image, reuse, l2_normalize):
            with slim.arg_scope([slim.batch_norm, slim.dropout],
                                is_training=is_training):
                with slim.arg_scope([slim.conv2d, slim.fully_connected,
                                     slim.batch_norm, slim.layer_norm],
                                    reuse=reuse):
                    features, logits = _create_network(
                        image, num_classes, l2_normalize=l2_normalize,
                        reuse=reuse, create_summaries=is_training,
                        weight_decay=weight_decay)
                    return features, logits

    return factory_fn


def _preprocess(image, is_training=False, enable_more_augmentation=True):
    image = image[:, :, ::-1]  # BGR to RGB
    if is_training:
        image = tf.image.random_flip_left_right(image)
        if enable_more_augmentation:
            image = tf.image.random_brightness(image, max_delta=50)
            image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
            image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
    return image


def _run_in_batches(f, data_dict, out, batch_size):
    data_len = len(out)
    num_batches = int(data_len / batch_size)

    s, e = 0, 0
    for i in range(num_batches):
        s, e = i * batch_size, (i + 1) * batch_size
        batch_data_dict = {k: v[s:e] for k, v in data_dict.items()}
        out[s:e] = f(batch_data_dict)
    if e < len(out):
        batch_data_dict = {k: v[e:] for k, v in data_dict.items()}
        out[e:] = f(batch_data_dict)


def extract_image_patch(image, bbox, patch_shape):
    bbox = np.array(bbox)
    if patch_shape is not None:
        # correct aspect ratio to patch shape
        target_aspect = float(patch_shape[1]) / patch_shape[0]
        new_width = target_aspect * bbox[3]
        bbox[0] -= (new_width - bbox[2]) / 2
        bbox[2] = new_width

    # convert to top left, bottom right
    bbox[2:] += bbox[:2]
    bbox = bbox.astype(np.int)

    # clip at image boundaries
    bbox[:2] = np.maximum(0, bbox[:2])
    bbox[2:] = np.minimum(np.asarray(image.shape[:2][::-1]) - 1, bbox[2:])
    if np.any(bbox[:2] >= bbox[2:]):
        return None
    sx, sy, ex, ey = bbox
    image = image[sy:ey, sx:ex]
    image = cv2.resize(image, patch_shape[::-1])

    return image


def _create_image_encoder(preprocess_fn, factory_fn, image_shape, batch_size=32,
                         session=None, checkpoint_path=None,
                         loss_mode="cosine"):
    image_var = tf.placeholder(tf.uint8, (None, ) + image_shape)

    preprocessed_image_var = tf.map_fn(
        lambda x: preprocess_fn(x, is_training=False),
        tf.cast(image_var, tf.float32))

    l2_normalize = loss_mode == "cosine"
    feature_var, _ = factory_fn(
        preprocessed_image_var, l2_normalize=l2_normalize, reuse=tf.AUTO_REUSE)
    feature_dim = feature_var.get_shape().as_list()[-1]

    if session is None:
        session = tf.Session()
    if checkpoint_path is not None:
        slim.get_or_create_global_step()
        init_assign_op, init_feed_dict = slim.assign_from_checkpoint(
            checkpoint_path, slim.get_variables_to_restore())
        session.run(init_assign_op, feed_dict=init_feed_dict)

    def encoder(data_x):
        out = np.zeros((len(data_x), feature_dim), np.float32)
        _run_in_batches(
            lambda x: session.run(feature_var, feed_dict=x),
            {image_var: data_x}, out, batch_size)
        return out

    return encoder


def create_image_encoder(model_filename, batch_size=32, loss_mode="cosine",
                         session=None):
    image_shape = 128, 64, 3
    factory_fn = _network_factory(num_classes=1501, is_training=False, weight_decay=1e-8)

    return _create_image_encoder(_preprocess, factory_fn, image_shape, batch_size, session,
        model_filename, loss_mode)


def create_box_encoder(model_filename, batch_size=32, loss_mode="cosine"):
    image_shape = 128, 64, 3
    image_encoder = create_image_encoder(model_filename, batch_size, loss_mode)

    def encoder(image, boxes):
        image_patches = []
        for box in boxes:
            patch = extract_image_patch(image, box, image_shape[:2])
            if patch is None:
                print("WARNING: Failed to extract image patch: %s." % str(box))
                patch = np.random.uniform(
                    0., 255., image_shape).astype(np.uint8)
            image_patches.append(patch)
        image_patches = np.asarray(image_patches)
        return image_encoder(image_patches)

    return encoder


def generate_detections(encoder, video_dir, det_data ,test_video):    
    videos = os.listdir(video_dir)
    videos.sort()
    for video_name in videos:
        if(video_name != test_video and test_video != '' ): 
            continue

        print("Processing %s" % video_name)

        # detection_file = os.path.join(det_dir, video_name[:-3]+'csv')
        # detections_in = np.loadtxt(detection_file, delimiter=',')
        detections_in = det_data
        detections_out = []

        cap = cv2.VideoCapture(os.path.join(video_dir, video_name))

        frame_indices = detections_in[:, 0].astype(np.int)
        min_frame_idx = frame_indices.astype(np.int).min()
        max_frame_idx = frame_indices.astype(np.int).max()
        for frame_idx in tqdm(range(min_frame_idx, max_frame_idx + 1)):
            print("Frame %05d/%05d" % (frame_idx, max_frame_idx))
            mask = frame_indices == frame_idx
            rows = detections_in[mask]
            ret, bgr_image = cap.read()
            features = encoder(bgr_image, rows[:, 2:6].copy())
            # edit
            rows = rows[:,0:14]
            detections_out += [np.r_[(row, feature)] for row, feature in zip(rows, features)]

        # feature_filename = os.path.join(feat_dir, "%s.npy" % video_name[:-4])
        # np.save(feature_filename, np.asarray(detections_out), allow_pickle=False)
        return np.asarray(detections_out)

def gather_sequence_info(video_name, video_path, feat_data):
    detections = feat_data

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    image_size = np.array(frame).shape[:-1] 
    # print(image_size)

    min_frame_idx = int(detections[:, 0].min())
    max_frame_idx = int(detections[:, 0].max())

    feature_dim = detections.shape[1] - 10 if detections is not None else 0
    seq_info = {
        "sequence_name": video_name,
        "detections": detections,
        "image_size": image_size,
        "min_frame_idx": min_frame_idx,
        "max_frame_idx": max_frame_idx,
        "feature_dim": feature_dim,
        "update_ms": None
    }
    return seq_info


def create_detections(detection_mat, frame_idx, min_height=0):
    frame_indices = detection_mat[:, 0].astype(np.int)
    mask = frame_indices == frame_idx

    detection_list = []
    for row in detection_mat[mask]:
        bbox, confidence, feature , f_bbox = row[2:6], row[6], row[14:] ,row[10:14]
        if bbox[3] < min_height:
            continue
        detection_list.append(Detection(bbox, confidence, feature ,f_bbox))
    return detection_list


def join(df_track):
    prev_frame_idx = min(df_track['track_id'].index)
    results = []
    for frame_idx, currrent_row in df_track.iterrows():
        gap = frame_idx - prev_frame_idx
        if(gap > 1):
            results.append(str(prev_frame_idx)+' -> '+ str(frame_idx))
            currrent_row = np.array(currrent_row)
            previous_row = np.array(df_track.loc[prev_frame_idx].values)
            steps = (currrent_row - previous_row) / gap

            for i, frame in enumerate(range(prev_frame_idx+1,frame_idx)):
                df_track.loc[frame] = np.array(previous_row + (i+1) * steps).astype(int)

        prev_frame_idx = frame_idx
    df_track = df_track.sort_index()

    misses = np.squeeze(list(set(range(min(df_track.index), 
                                       max(df_track.index) + 1)).difference(df_track.index)))
    if(len(misses)==0 and len(results) > 0):
        print('Track:', int(df_track['track_id'].iloc[0]),', concatenation complete, ',results)
    elif(len(misses)!=0):
        print('Warning!! Frame:', int(df_track['track_id'].iloc[0]), ', concatenation incomplete\n')
    return df_track

def run_concatenate(results):

    df_all = pd.DataFrame(results)
    df = df_all.iloc[:,:15]
    df.columns = ['frame_id','track_id','xmin','ymin','width','height', 
                  'confidence','neg_1', 'neg_2', 'neg_3','xF','yF','widthF', 'heightF', 'neg_5']
    df.index = df['frame_id']
    df = df.drop(['frame_id'], axis=1)

    concat = []
    From, To = min(df['track_id']), max(df['track_id'])+1
    for track_id in range(From, To):
        concat.append(join(df.loc[df['track_id']==track_id].copy()))
        
    df_concat = pd.concat(concat)
    df_concat = df_concat.sort_index()
    df_concat['filename'] = df_all.iloc[0,15]
    # df_concat.to_csv(concat_track_file, header=None)
    print('=================')
    return df_concat

def gather_sequence_info_2(video_name, video_path, feat_path):
    detections = np.load(feat_path)

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    image_size = np.array(frame).shape[:-1] 
    # print(image_size)

    min_frame_idx = int(detections[:, 0].min())
    max_frame_idx = int(detections[:, 0].max())

    feature_dim = detections.shape[1] - 10 if detections is not None else 0
    seq_info = {
        "sequence_name": video_name,
        "detections": detections,
        "image_size": image_size,
        "min_frame_idx": min_frame_idx,
        "max_frame_idx": max_frame_idx,
        "feature_dim": feature_dim,
        "update_ms": None
    }
    return seq_info

# ========================================================================================================


def capture(video_path, cap_dir, results, seq_info, is_plot=False):

    if os.path.exists(cap_dir):
        shutil.rmtree(cap_dir)
    os.makedirs(cap_dir)

    cap = cv2.VideoCapture(video_path)

    N_track = int(max(results[:,1]))
    subplot_x = 6
    subplot_y = int(math.ceil(N_track/subplot_x))
    print('Total Tracks:', N_track)
    print('Subplot', subplot_y, subplot_x)

    image_size = seq_info['image_size']
    points = {}
    captured = []

    with tf.Session() as sess:
        for frame_idx in tqdm(range(
                            seq_info['min_frame_idx'], 
                            seq_info['max_frame_idx'] + 1), 'capturing output'):
        
            image_np = np.array(cap.read()[1])

            mask = results[:, 0].astype(np.int) == frame_idx
            track_ids = results[mask, 1].astype(np.int)
            boxes = results[mask, 2:6]

            for track_id, box in zip(track_ids, boxes):
                if(track_id not in captured):
                    captured.append(track_id)

                    l,t,w,h = np.array(box).astype(int)
                    if(l<0): l=0 # if xmin is negative 
                    if(t<0): t=0 # if ymin is negative

                    if(l+w > image_size[1]): w=image_size[1]-l # if xmax exceeds width
                    if(t+h > image_size[0]): h=image_size[0]-t # if ymax exceeds height

                    cropped_image = sess.run(tf.image.crop_to_bounding_box(image_np, t, l, h, w))
                    cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)

                    img = Image.fromarray(cropped_image)
                    img.save(os.path.join(cap_dir, str(track_id)+'.jpg'))

                    if(is_plot):
                        plt.subplot(subplot_y, subplot_x, len(captured))
                        plt.imshow(cropped_image)
                        plt.title(str(track_id)+', '+str(frame_idx))

    cap.release()

    if(is_plot):
        plt.subplots_adjust(top=0.92, bottom=0.08, left=0.1, right=0.95, hspace=0.5, wspace=0.8)
        plt.show()

# ========================================================================================================
   
def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    window_size = np.abs(np.int(window_size))
    order = np.abs(np.int(order))
    
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2: # order should be less than or equal window-2
        raise TypeError("window_size is too small for the polynomials order")
        
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    
    # pad the signal at the extremes with
    # values taken from the signal itself
    
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    
    return np.convolve( m[::-1], y, mode='valid')

# ========================================================================================================

def golay_filter(df_track, window_size=45, order=5):
    if(len(df_track) <= window_size):
        return df_track
    df_track[2] = savitzky_golay(df_track[2].values, window_size=window_size, order=order, deriv=0, rate=1)
    df_track[3] = savitzky_golay(df_track[3].values, window_size=window_size, order=order, deriv=0, rate=1)
    df_track[4] = savitzky_golay(df_track[4].values, window_size=window_size, order=order, deriv=0, rate=1)
    df_track[5] = savitzky_golay(df_track[5].values, window_size=window_size, order=order, deriv=0, rate=1)
    return df_track
    
def poly_interpolate(df_track):
    model = make_pipeline(PolynomialFeatures(5), Ridge(solver='svd'))
    X = np.array(df_track.index).reshape(-1, 1)
    df_track[2] = model.fit(X, df_track[2]).predict(X)
    df_track[3] = model.fit(X, df_track[3]).predict(X)
    df_track[4] = model.fit(X, df_track[4]).predict(X)
    df_track[5] = model.fit(X, df_track[5]).predict(X)
    return df_track

def moving_avg(df_track, window=5):
    df_haed = df_track[[2,3,4,5]][:window-1]
    df_tail = df_track[[2,3,4,5]].rolling(window=window).mean()[window-1:]
    df_track[[2,3,4,5]] = pd.concat([df_haed, df_tail], axis=0)
    return df_track

def smooth(df, smooth_method):
    polynomials = []
    From, To = min(df[1]), max(df[1])+1

    for track_id in range(From, To):
        df_track = df.loc[df[1]==track_id].copy()

        if(smooth_method == 'poly'): df_track = poly_interpolate(df_track)
        elif(smooth_method == 'moving'): df_track = moving_avg(df_track)
        elif(smooth_method == 'golay'): df_track = golay_filter(df_track)
            
        polynomials.append(df_track)

#     do_range = df[0].unique()
#     for track_id in do_range:
#         df_track = df.loc[df[0]==track_id].copy()

#         if(smooth_method == 'poly'): df_track = poly_interpolate(df_track)
#         elif(smooth_method == 'moving'): df_track = moving_avg(df_track)
#         elif(smooth_method == 'golay'): df_track = golay_filter(df_track)
            
#         polynomials.append(df_track)

    df_smooth = pd.concat(polynomials)
    df_smooth = df_smooth.sort_index()
    return df_smooth
    # return df_smooth.values
    
    
############################################################################################



loss_mode = "cosine"

model = "object_tracking/resources/networks/mars-small128.ckpt-68577"

max_cosine_distance = 0.35

nn_budget = None



# ==============================================================================
fs = []
count_human_lasts = []
metrics = []
trackers = []

for i in range(4):
    fs.append(create_box_encoder(model, batch_size=32, loss_mode=loss_mode))
    count_human_lasts.append([])
# --------------
    metrics.append(nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget))
    trackers.append(Tracker(metrics[i], max_age=50, n_init=5))

# ==============================================================================

print ('loading model..')
PATH_TO_CKPT = 'object_detection/faster_rcnn_resnet101_coco_11_06_2017/frozen_inference_graph.pb'
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
label_map = label_map_util.load_labelmap(os.path.join('object_detection','data', 'mscoco_label_map.pbtxt'))
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=90, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# ==============================================================================

def run_process(sess,video_name,f,count_human_last,metric,tracker,output,output_feat,cam):

    all_results = pd.DataFrame([])

    video_dir = "dataset/videos"

    det_dir = "dataset/detections"

    feat_dir = "dataset/features"

    loss_mode = "cosine"

    model = "object_tracking/resources/networks/mars-small128.ckpt-68577"

    tracks_dir= "dataset/tracks"

    min_confidence = 0.7

    min_detection_height = 0

    nms_max_overlap = 1.0

    max_cosine_distance = 0.4

    nn_budget = None


    
    videos = os.listdir(video_dir)
    videos.sort()
    results = []
    feature_bbox = []

    if video_name in videos:
        video_path = os.path.join(video_dir, video_name)
        #dont need
        feat_path  = os.path.join(feat_dir, video_name[:-3]+'npy')
        print('Processing Video:', video_name + '..')
        det_data = detection(sess, 
                    video_path=os.path.join(video_dir, video_name)
                )
        pd.DataFrame(det_data).to_csv(det_dir+'/'+video_name[:-3]+'csv',index = False)
        # ==============================================================================
        feat_data = generate_detections(f, video_dir, det_data ,video_name)
        
        feature_filename = os.path.join(feat_dir, "%s.npy" % video_name[:-4])
        np.save(feature_filename, feat_data, allow_pickle=False)
        # ==============================================================================
        seq_info = gather_sequence_info(video_name, video_path, feat_data)
    
        print('Video Path:', video_path,'\tFeatures:', feat_path)

        def frame_callback(vis, frame_idx):
            # print("Processing frame %05d" % frame_idx)

            # Load image and generate detections.
            detections = create_detections(seq_info["detections"], frame_idx, min_detection_height)
            detections = [d for d in detections if d.confidence >= min_confidence]

            # Run non-maxima suppression.
            boxes = np.array([d.tlwh for d in detections])
            scores = np.array([d.confidence for d in detections])
            indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
            detections = [detections[i] for i in indices]

            # Update tracker.
            tracker.predict()
            tracker.update(detections)

            # Update visualization.
            count_human = vis.draw_trackers(tracker.tracks)

            # Store results.
            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                bbox = track.to_tlwh()
                f_bbox = track.f_bbox
                bbox_feat = track.bbox_feat
                ID = count_human.index(track.track_id) + 1
#                 ID = str(cam)+'_'+str(ID)
                results.append([frame_idx, ID, bbox[0], bbox[1], bbox[2], bbox[3],1,-1,-1,-1
                                ,f_bbox[0], f_bbox[1], f_bbox[2], f_bbox[3],-1,video_path])
                feature_bbox.append(bbox_feat)

        visualizer = visualization.NoVisualization( seq_info, count_human_last)
        visualizer.run(frame_callback)

    # sent (results) to database here6
    # results = run_concatenate(results)
    results = pd.DataFrame(results)
    results.columns = range(results.shape[1])
    results = smooth(results, smooth_method='golay')
#     edit
    results[1] = str(cam)+'_' + results[1].astype(str)
    
    output_feat.put((int(cam), feature_bbox))
    output.put((cam, results))
    # sum for test
    # all_results = pd.concat([all_results, results])

    file_name = 'dataset/tracks/'+ video_name[:-3]+'csv'
    results.to_csv(file_name, header=None,index=False)

def sortFirst(val): 
    return val[0] 

import multiprocessing as mp
output = mp.Queue()
output_feat = mp.Queue()
reid_dict = {}
reid_life = 0

from scipy.spatial import distance
import threading 

# l1 = ['test_0-5.avi','test_5-10.avi'] 
# l2 = ['test_5-10.avi','test_10-15.avi']
# l3 = ['Front-HW-20200203143657.avi','Front-HW-20200203143713.avi']
l1 = ['front-509-20200227152652.avi','front-509-20200227152708.avi','front-509-20200227152724.avi'] 
l2 = ['Front-HW-20200227152653.avi','Front-HW-20200227152709.avi','Front-HW-20200227152725.avi']


# l1 = ['front-509-20200203143657.avi','Front-HW-20200203143657.avi'] 
# l2 = ['Front-HW-20200203143657.avi','front-509-20200203143657.avi']

l3 = ['test_0-555.avi','front-509-20200203143657.avi','front-509-20200203143657.avi']
with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        for l1_v,l2_v,l3_v in zip(l1,l2,l3):
            print((l1_v,l2_v,l3_v))
            t1 = threading.Thread(target=run_process, args=(sess,l1_v,fs[0],count_human_lasts[0],metrics[0],trackers[0],output,output_feat,1,)) 
            t2 = threading.Thread(target=run_process, args=(sess,l2_v,fs[1],count_human_lasts[1],metrics[1],trackers[1],output,output_feat,2,)) 
            # t3 = threading.Thread(target=run_process, args=(sess,[l3_v],fs[2],count_human_lasts[2],metrics[2],trackers[2],output,output_feat,3,))

            # starting thread 1 
            t1.start() 
            # starting thread 2 
            t2.start() 

            # wait until thread 1 is completely executed 
            t1.join() 
            # wait until thread 2 is completely executed 
            t2.join() 

            # reid cam1&2
            results = [output.get() for p in range(output.qsize())]
            results_feat = [output_feat.get() for p in range(output_feat.qsize())]

            results.sort(key=sortFirst)  
            results = [r[1] for r in results]
            results_feat.sort(key=sortFirst)  
            results_feat = [r[1] for r in results_feat]

            results0 = pd.DataFrame(results[1])
            results1 = pd.DataFrame(results[0])
            results_feat0 = pd.DataFrame(results_feat[1])
            results_feat1 = pd.DataFrame(results_feat[0])

            cam_0 = pd.concat([results0, results_feat0], axis=1)
            cam_0.columns = range(cam_0.shape[1])
            cam_1 = pd.concat([results1, results_feat1], axis=1)
            cam_1.columns = range(cam_1.shape[1])

            scope_cam_1 = cam_1[cam_1[3].between(220, 625)]
            scope_cam_0 = cam_0[cam_0[3].between(220, 625)] 

            # reid from previous video
            # temp_reid_dict = reid_dict.copy()
            # if len(temp_reid_dict) != 0 :
            #     for prev_id in temp_reid_dict :
            #         if prev_id in cam_1[1].unique() :
            #             cam_1[1] = cam_1[1].replace(row1[1], reid_dict[prev_id])
            #         else :
            #             if reid_life > 1 :
            #                 del reid_dict[prev_id]
            #                 reid_life = 0
            # reid_life += 1
            
            done_check = []
            re_cam_1 = scope_cam_1.copy()
            for index1, row1 in tqdm(re_cam_1.iterrows()):
                min_dis = [1000,'']
                for index0, row0 in scope_cam_0.iterrows(): 
                    if row1[1] in done_check:
                        continue
                    dis = distance.cosine(np.array(row1.iloc[16:]).astype(float),np.array(row0.iloc[16:]).astype(float))
                    if dis < min_dis[0] :
                        min_dis[0] = dis
                        min_dis[1] = row0[1]
                if min_dis[0] < 0.3 :
                    done_check.append(row1[1])
                    print((row1[1], min_dis[1],min_dis[0]))
                    # update result
                    reid_dict[row1[1]] = min_dis[1]
                    cam_1[1] = cam_1[1].replace(row1[1], min_dis[1])
            
            cam_1 = cam_1.iloc[:,:16]
            cam_1.to_csv('dataset/tracks/' + l1_v[:-3] + 'csv',index = False,header = False)

# both threads completely executed 
print(reid_dict)
print("Done!") 



