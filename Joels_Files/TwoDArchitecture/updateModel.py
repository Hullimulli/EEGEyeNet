import tensorflow as tf
from keras.losses import LossFunctionWrapper
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.util import dispatch

# class AngleLoss(LossFunctionWrapper):
#   """Computes the mean of squares of errors between labels and predictions.
#
#   `loss = square(y_true - y_pred)`
#
#   Standalone usage:
#
#   >>> y_true = [[0., 1.], [0., 0.]]
#   >>> y_pred = [[1., 1.], [1., 0.]]
#   >>> # Using 'auto'/'sum_over_batch_size' reduction type.
#   >>> mse = tf.keras.losses.MeanSquaredError()
#   >>> mse(y_true, y_pred).numpy()
#   0.5
#
#   >>> # Calling with 'sample_weight'.
#   >>> mse(y_true, y_pred, sample_weight=[0.7, 0.3]).numpy()
#   0.25
#
#   >>> # Using 'sum' reduction type.
#   >>> mse = tf.keras.losses.MeanSquaredError(
#   ...     reduction=tf.keras.losses.Reduction.SUM)
#   >>> mse(y_true, y_pred).numpy()
#   1.0
#
#   >>> # Using 'none' reduction type.
#   >>> mse = tf.keras.losses.MeanSquaredError(
#   ...     reduction=tf.keras.losses.Reduction.NONE)
#   >>> mse(y_true, y_pred).numpy()
#   array([0.5, 0.5], dtype=float32)
#
#   Usage with the `compile()` API:
#
#   ```python
#   model.compile(optimizer='sgd', loss=tf.keras.losses.MeanSquaredError())
#   ```
#   """
#
#   def __init__(self,
#                reduction=losses_utils.ReductionV2.AUTO,
#                name='angle_loss'):
#     """Initializes `MeanSquaredError` instance.
#
#     Args:
#       reduction: Type of `tf.keras.losses.Reduction` to apply to
#         loss. Default value is `AUTO`. `AUTO` indicates that the reduction
#         option will be determined by the usage context. For almost all cases
#         this defaults to `SUM_OVER_BATCH_SIZE`. When used with
#         `tf.distribute.Strategy`, outside of built-in training loops such as
#         `tf.keras` `compile` and `fit`, using `AUTO` or `SUM_OVER_BATCH_SIZE`
#         will raise an error. Please see this custom training [tutorial](
#           https://www.tensorflow.org/tutorials/distribute/custom_training) for
#             more details.
#       name: Optional name for the instance. Defaults to 'mean_squared_error'.
#     """
#     super().__init__(angle_loss, name=name, reduction=reduction)

@dispatch.add_dispatch_support
def angle_loss(y_pred, y_true):
    """
    Custom loss function for models that predict the angle on the fix-sacc-fix dataset
    Angles -pi and pi should lead to 0 loss, since this is actually the same angle on the unit circle
    Angles pi/2 and -pi/2 should lead to a large loss, since this is a difference by pi on the unit circle
    Therefore we compute the absolute error of the "shorter" direction on the unit circle
    """

    return tf.reduce_mean(tf.math.square(tf.abs(tf.atan2(tf.sin(y_pred - y_true), tf.cos(y_pred - y_true)))))



def mse(pred,ground):
    return tf.math.reduce_mean(tf.square(pred - ground))

@tf.function
def mseUpdate(model,input, ground, seed):
    loss = tf.keras.losses.MeanSquaredError()
    with tf.GradientTape() as tape:
        tf.random.set_seed(seed)
        pred = model(input, training=True)
        loss_value = loss(pred, ground)
    grads = tape.gradient(loss_value, model.trainable_variables)
    model.optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss_value

@tf.function
def angleLossUpdate(model,input, ground, seed):
    loss = angle_loss

    with tf.GradientTape() as tape:
        tf.random.set_seed(seed)
        pred = model(input, training=True)
        loss_value = loss(pred, ground)
    grads = tape.gradient(loss_value, model.trainable_variables)
    model.optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return loss_value