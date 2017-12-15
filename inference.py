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

from deeplab_resnet import DeepLabResNetModel, ImageReader, decode_labels, prepare_label

SAVE_DIR = './output/'
IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLabLFOV Network Inference.")
    parser.add_argument("--img-path", type=str,
                        help="Path to the RGB image file.")
    parser.add_argument("--weights", type=str,
                        help="Path to the file with model weights.")
    parser.add_argument("--save-dir", type=str, default=SAVE_DIR,
                        help="Where to save predicted mask.")
    return parser.parse_args()

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
    args = get_arguments()

    # Prepare image.
    img = tf.image.decode_jpeg(tf.read_file(args.img_path), channels=3)
    # Convert RGB to BGR.
    img_r, img_g, img_b = tf.split(axis=2, num_or_size_splits=3, value=img)
    img = tf.cast(tf.concat(axis=2, values=[img_b, img_g, img_r]), dtype=tf.float32)
    # Extract mean.
    img -= IMG_MEAN

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
    load(loader, sess, args.model_weights)

    # Perform inference.
    preds = sess.run(pred)

    out_filename =os.path.join(args.save_dir,
        os.path.splitext(os.path.basename(args.img_path))[0] + ".png")
    msk = decode_labels(preds)
    import bpdb; bpdb.set_trace()
    im = Image.fromarray(msk[0]*255)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    im.save(out_filename)

    print('The output file has been saved to {}'.format(out_filename))


if __name__ == '__main__':
    main()
