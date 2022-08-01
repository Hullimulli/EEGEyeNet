import tensorflow as tf

def angle_loss(a, b):
    """
    Custom loss function for models that predict the angle on the fix-sacc-fix dataset
    Angles -pi and pi should lead to 0 loss, since this is actually the same angle on the unit circle
    Angles pi/2 and -pi/2 should lead to a large loss, since this is a difference by pi on the unit circle
    Therefore we compute the absolute error of the "shorter" direction on the unit circle
    """
    return tf.reduce_mean(tf.math.square(tf.abs(tf.atan2(tf.sin(a - b), tf.cos(a - b)))))


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
    with tf.GradientTape() as tape:
        tf.random.set_seed(seed)
        pred = model(input, training=True)
        loss_value = 100*angle_loss(pred, ground)
    grads = tape.gradient(loss_value, model.trainable_variables)
    model.optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return loss_value