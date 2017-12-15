"""
Implement some custom layers, not provided by TensorFlow.

Trying to follow as much as possible the style/standards used in
tf.contrib.layers
"""
import uuid

import numpy as np
import tensorflow as tf

from tensorflow.contrib.framework.python.ops import add_arg_scope
from tensorflow.contrib.framework.python.ops import variables
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.python.framework import ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import variable_scope


def focal_loss(predictions,
               labels,
               gamma=2,
               alpha=0.25,
               weights=1.0,
               epsilon=1e-7,
               scope=None):
    """Adds a Focal Loss term to the training procedure.
    For each value x in `predictions`, and the corresponding l in `labels`,
    the following is calculated:
    ```
    pt = 1 - x                  if l == 0
    pt = x                      if l == 1
    focal_loss = - a * (1 - pt)**g * log(pt)
    ```
    where g is `gamma`, a is `alpha`.
    See: https://arxiv.org/pdf/1708.02002.pdf
    `weights` acts as a coefficient for the loss. If a scalar is provided, then
    the loss is simply scaled by the given value. If `weights` is a tensor of size
    [batch_size], then the total loss for each sample of the batch is rescaled
    by the corresponding element in the `weights` vector. If the shape of
    `weights` matches the shape of `predictions`, then the loss of each
    measurable element of `predictions` is scaled by the corresponding value of `weights`.

    Args:
        labels: The ground truth output tensor, same dimensions as 'predictions'.
        predictions: The predicted outputs.
        gamma, alpha: parameters.
        weights: Optional `Tensor` whose rank is either 0, or the same rank as
                `labels`, and must be broadcastable to `labels` (i.e., all dimensions must
                be either `1`, or the same as the corresponding `losses` dimension).
        epsilon: A small increment to add to avoid taking a log of zero.
        scope: The scope for the operations performed in computing the loss.
        loss_collection: collection to which the loss will be added.
        reduction: Type of reduction to apply to loss.
    :param predictions: The predicted outputs.(Operated by Softmax)
    :param labels: Ground Truth labels for predictions.
    :param gamma: Focal loss parameter -- check paper link above.
    :param alpha: Focal loss parameter -- check paper link above.
    :param weights: Class weights tensor whose rank is either 0, or the same rank as `labels`
    :param epsilon: Small value to prevent Nan operation while computing log.
    :param scope: Named scoped value.
    :return: Forward and Backward pass functionality.
    """
    with ops.name_scope(scope, "focal_loss", (predictions, labels, weights)) as scope:
        predictions = math_ops.to_float(predictions)
        labels = math_ops.to_float(labels)
        predictions.get_shape().assert_is_compatible_with(labels.get_shape())
        preds = array_ops.where(math_ops.equal(labels, 1), predictions, 1. - predictions)
        losses = -alpha * (1. - preds)**gamma * math_ops.log(preds + epsilon)
        return compute_weighted_loss(losses, weights, scope=scope)

