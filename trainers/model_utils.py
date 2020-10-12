import tensorflow as tf


def layerSummary(value):
  tf.summary.scalar("fraction_of_zero_values", tf.nn.zero_fraction(value))
  tf.summary.histogram("activation", value)


def getBinaryPredictions(logits):
  with tf.name_scope("predictions"):
    logistic = tf.sigmoid(logits)
    classIds = tf.cast(logistic > 0.5, tf.int32)

  predictions = {
    "logits": logits,
    "logistic": logistic,
    "probabilities": logistic,
    "class_id": classIds,
  }
  return predictions


def getBinaryLosses(labels, predictions):
  with tf.name_scope("losses"):
    labels = tf.expand_dims(labels, -1)
    unreducedLoss = tf.compat.v1.losses.sigmoid_cross_entropy(labels, predictions["logits"],
                                                    reduction=tf.losses.Reduction.NONE)
    averageLoss = tf.reduce_mean(unreducedLoss)
    loss = tf.reduce_sum(unreducedLoss)

  losses = {
    "unreduced_loss": unreducedLoss,
    "average_loss": averageLoss,
    "loss": loss,
  }
  return losses


def getBinaryMetricOps(labels, predictions, losses):
  with tf.name_scope("metrics"):
    labels = tf.expand_dims(labels, -1)
    averageLoss = tf.compat.v1.metrics.mean(losses["unreduced_loss"])
    accuracy = tf.compat.v1.metrics.accuracy(labels, predictions["class_id"], name="accuracy")
    auc = tf.compat.v1.metrics.auc(labels, predictions["probabilities"], name="auc")
    aucPrecisionRecall = tf.compat.v1.metrics.auc(labels, predictions["probabilities"],
                                        curve="PR", name="auc_precision_recall")

  metrics = {
    "accuracy": accuracy,
    "auc": auc,
    "auc_precision_recall": aucPrecisionRecall,
    "average_loss": averageLoss,
  }
  return metrics


def getOptimizer(optimizerName="Adam", learningRate=0.001):
  optimizerClasses = {
    "Adagrad": tf.compat.v1.train.AdagradOptimizer,
    "Adam": tf.compat.v1.train.AdamOptimizer,
    "Ftrl": tf.compat.v1.train.FtrlOptimizer,
    "RMSProp": tf.compat.v1.train.RMSPropOptimizer,
    "SGD": tf.compat.v1.train.GradientDescentOptimizer,
  }
  optimizer = optimizerClasses[optimizerName](learning_rate=learningRate)
  return optimizer


def getTrainOp(loss, optimizer):
  with tf.name_scope("train"):
    trainOp = optimizer.minimize(loss, global_step=tf.train.get_global_step())
  return trainOp
