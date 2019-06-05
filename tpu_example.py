import os
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.python.tpu import tpu_estimator
from tensorflow.python.tpu import tpu_optimizer
from tensorflow.python.tpu import tpu_config
from functools import partial
from absl import flags, app

FLAGS = flags.FLAGS
flags.DEFINE_string("tpu", None, "TPU Address")
flags.DEFINE_integer("iterations",2, "Number of Itertions")
flags.DEFINE_integer("batch_size", 10, "Size of eahc Batch")
flags.DEFINE_boolean("use_tpu", True, " Use TPU")
flags.DEFINE_string("model_dir", "mnist/", "Directory to Save the Models and Checkpoint")
flags.DEFINE_string("data_dir", None, "Directory to Save Data to")
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
        return x
        x = self.layer_2(x)
        o = self.out(x)
        return o


def input_(mode, batch_size, iterations, **kwargs):

    dataset = tfds.load(
        "mnist",
        as_supervised=True,
        split="train" if mode == tf.estimator.ModeKeys.TRAIN else "test",
        data_dir = kwargs["data_dir"]
        )

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
    #model = tf.keras.Sequential([
    #    tf.keras.layers.Flatten(input_shape=[28, 28, 1]),
    #    tf.keras.layers.Dense(128),
    #    tf.keras.layers.Dense(10, activation="softmax")
    #])
    optimizer = None
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.optimizers.Adam(params.get("learing_rate", 1e-3))
        if params.get("use_tpu", True):
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
          apply_grads = optimizer.apply_gradients(zip(gradient, model.trainable_variables))
        return apply_grads

    if mode == tf.estimator.ModeKeys.TRAIN:
        return tpu_estimator.TPUEstimatorSpec(mode, loss=loss, train_op=train_fn())

def main(_):
  input_fn = partial(input_, iterations=FLAGS.iterations)
  cluster = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=FLAGS.tpu)
  run_config =  tpu_config.RunConfig(
    model_dir = FLAGS.model_dir,
    cluster=cluster,
    tpu_config=tpu_config.TPUConfig(FLAGS.iterations))

  classifier = tpu_estimator.TPUEstimator(
    model_fn=model_fn,
    use_tpu=FLAGS.use_tpu,
    train_batch_size=FLAGS.batch_size,
    eval_batch_size=FLAGS.batch_size,
    config=run_config,
    params={
      "use_tpu": FLAGS.use_tpu,
      "data_dir": FLAGS.data_dir
    }
  )
  # classifier = tf.estimator.Estimator(model_fn=model_fn, model_dir="mnist/")
  #tf.estimator.train_and_evaluate(
  #    classifier,
  #    train_spec=tf.estimator.TrainSpec(input_fn=input_fn),
  #    eval_spec=tf.estimator.EvalSpec(input_fn=input_fn),
  #    max_steps=3000
  #)
  classifier.train(
          input_fn=lambda params:input_fn(
              mode=tf.estimator.ModeKeys.TRAIN,
              **params),
          max_steps=2000)#.evaluate(
                  #input_fn=lambda params:input_fn(
                  #    mode=tf.estimator.ModeKeys.EVAL,
                  #    **params),
                  #steps=1000)
if __name__ == "__main__":
  app.run(main)
