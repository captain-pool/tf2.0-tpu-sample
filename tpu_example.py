import os
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.python.tpu import tpu_estimator
from tensorflow.python.tpu import tpu_optimizer
from functools import partial


use_tpu = True
iterations = 2
class MNIST(tf.keras.Model):
    def __init__(self, params):
        super(MNIST, self).__init__()
        self.layer_1 = tf.keras.layers.Dense(
            params.get(
                'layer_1', 128), activation=params.get(
                'layer_1_activation', "linear"))
        self.layer_2 = tf.keras.layers.Dense(
            params.get(
                'layer_2', 128), activation=params.get(
                'layer_2_activation', "linear"))
        self.out = tf.keras.layers.Dense(params.get(
            'num_classes', 10), activation="softmax")

    @tf.function(
        input_signature=[
            tf.TensorSpec(
                shape=[
                    None,
                    28,
                    28,
                    1],
                dtype=tf.float32)])
    def call(self, inputs, training=True):
        x = tf.keras.layers.Flatten()(inputs)
        x = self.layer_1(x)
        x.trainable = training
        x = self.layer_2(x)
        x.trainable = training
        o = self.out(x)
        o.trainable = training
        return o


def input_fn(mode, batch_size):

    dataset = tfds.load(
        "mnist",
        as_supervised=True,
        split="train" if mode == tf.estimator.ModeKeys.TRAIN else "test")

    def scale(image, label):
        image = tf.cast(image, tf.float32)
        image = image / tf.reduce_max(image)
#    label = tf.one_hot(label, 10)
        return image, label

    dataset = dataset.map(scale).shuffle(
        1000).repeat(iterations).batch(batch_size, drop_remainder=True)
    return dataset


def model_fn(features, labels, mode, params):
    model = MNIST(params)
    optimizer = None
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.optimizers.Adam(params.get("learing_rate", 1e-3))
        if use_tpu:
          optimizer = tpu_optimizer.CrossShardOptimizer(optimizer)

    with tf.GradientTape() as tape:
        logits = model(features)
        if mode == tf.estimator.ModeKeys.PREDICT:
            preds = {
                "predictions": logits
            }
            return tpu_estimator.TPUEstimatorSpec(mode, predictions=preds)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True)(labels, logits)
    if mode == tf.estimator.ModeKeys.EVAL:
        return tpu_estimator.TPUEstimatorSpec(mode, loss=loss)

    def train_fn():
        assert optimizer is not None
        gradient = tape.gradient(loss, model.trainable_variables)
        global_step = tf.compat.v1.train.get_global_step()
        update_global_step = tf.compat.v1.assign(global_step, global_step + 1, name='update_global_step')
        with tf.control_dependencies([update_global_step]):
          apply_grads = optimizer.apply_gradients(
              zip(gradient, model.trainable_variables))
        return apply_grads

    if mode == tf.estimator.ModeKeys.TRAIN:
        return tpu_estimator.TPUEstimatorSpec(mode, loss=loss, train_op=train_fn())


input_fn = partial(input_fn, batch_size=10)
cluster = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=tpu)
run_config =  tpu_config.RunConfig(
  model_dir = "mnist/",
  cluster=cluster,
  tpu_config=tpu_config.TPUConfig(iterations))

classifier = tpu_estimator.TPUEstimator(
  model_fn=model_fn,
  use_tpu=use_tpu,
  train_batch_size=10,
  eval_batch_size=10,
  config=run_config
)
# classifier = tf.estimator.Estimator(model_fn=model_fn, model_dir="mnist/")
tf.estimator.train_and_evaluate(
    classifier,
    train_spec=tf.estimator.TrainSpec(input_fn=input_fn),
    eval_spec=tf.estimator.EvalSpec(input_fn=input_fn)
)
