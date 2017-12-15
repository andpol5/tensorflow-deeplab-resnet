"""Run DeepLab-ResNet on a given image.

This script computes a segmentation mask for a given image.
"""

from __future__ import print_function

import argparse
import os
import sys
import time

from PIL import Image
import numpy as np
import tensorflow as tf
import cv2

from deeplab_resnet import DeepLabResNetModel, ImageReader, decode_labels, prepare_label

SAVE_DIR = './output/'

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLabLFOV Network Inference.")
    parser.add_argument("-i", "--input_file", type=str, required=True,
                help="Path to the RGB image file.")
    parser.add_argument("-w", "--weights", type=str, required=True,
                help="Path to the file with model weights.")
    parser.add_argument("-o", "--output_dir", type=str, default=SAVE_DIR,
                help="Where to save predicted mask.")

    args = parser.parse_args()
    return args.input_file, args.weights, args.output_dir

def load(saver, sess, ckpt_path):
    '''Load trained weights.

    Args:
      saver: TensorFlow saver object.
      sess: TensorFlow session.
      ckpt_path: path to checkpoint file with parameters.
    '''
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))

def main():
    """Create the model and start the evaluation process."""
    input_file, weights, output_dir = get_arguments()

    # Prepare image.
    img = tf.image.decode_jpeg(tf.read_file(input_file), channels=3)
    # Convert RGB to BGR.
    img_r, img_g, img_b = tf.split(axis=2, num_or_size_splits=3, value=img)
    img = tf.cast(tf.concat(axis=2, values=[img_b, img_g, img_r]), dtype=tf.float32)
    # Normalize image from uint8
    img = tf.scalar_mul(1.0/255.0, img)

    # Create network.
    net = DeepLabResNetModel({'data': tf.expand_dims(img, axis=0)}, is_training=False)

    # Which variables to load.
    restore_var = tf.global_variables()

    # Predictions.
    raw_output = net.layers['fc1_voc12']
    raw_output_up = tf.image.resize_bilinear(raw_output, tf.shape(img)[0:2,])
    raw_output_up = tf.argmax(raw_output_up, axis=3)
    pred = tf.expand_dims(raw_output_up, axis=3)


    # Set up TF session and initialize variables.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()

    sess.run(init)

    # Load weights.
    loader = tf.train.Saver(var_list=restore_var)
    load(loader, sess, weights)

    # Perform inference.
    preds = sess.run(pred)

    out_filename =os.path.join(output_dir,
        os.path.splitext(os.path.basename(input_file))[0] + ".png")
    msk = decode_labels(preds)
    im = Image.fromarray(msk[0])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    im.save(out_filename)

    print('The output file has been saved to {}'.format(out_filename))

    in_rgb = cv2.imread(input_file)
    out_mask = cv2.imread(out_filename)
    res = np.vstack((in_rgb, out_mask*255))
    cv2.imwrite(os.path.join(output_dir, 'combined.jpg'), res)

if __name__ == '__main__':
    main()

