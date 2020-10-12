import tensorflow as tf

EVAL_INTERVAL = 60


def getRunConfig():
  return tf.estimator.RunConfig(
    save_checkpoints_secs=EVAL_INTERVAL,
    keep_checkpoint_max=5
  )


def getTrainSpec(inputFn, trainSteps):
  return tf.estimator.TrainSpec(
    input_fn=inputFn,
    max_steps=trainSteps
  )


def getExporter(servingInputFn):
  return tf.estimator.LatestExporter(
    name="exporter",
    serving_input_receiver_fn=servingInputFn
  )


def getEvalSpec(inputFn, exporter):
  return tf.estimator.EvalSpec(
    input_fn=inputFn,
    steps=None,  # until OutOfRangeError from input_fn
    exporters=exporter,
    start_delay_secs=min(EVAL_INTERVAL, 120),
    throttle_secs=EVAL_INTERVAL
  )
