import shutil
from argparse import ArgumentParser

import tensorflow as tf

from trainers.conf_utils import getRunConfig, getTrainSpec, getExporter, getEvalSpec
from trainers.ml_100k import getFeatureColumns, getInput, servingInput
from trainers.model_utils import layerSummary, getOptimizer


def modelFn(features, labels, mode, params):
  # feature columns
  categoricalColumns = params.get("categorical_columns", [])
  numericColumns = params.get("numeric_columns", [])
  # structure components
  useLinear = params.get("use_linear", True)
  useMf = params.get("use_mf", True)
  useDnn = params.get("use_dnn", True)
  # structure params
  embeddingSize = params.get("embedding_size", 4)
  hiddenUnits = params.get("hidden_units", [16, 16])
  activationFn = params.get("activation", tf.nn.relu)
  dropout = params.get("dropout", 0)
  # training params
  optimizer = params.get("optimizer", "Adam")
  learningRate = params.get("learning_rate", 0.001)

  # check params
  categoricalDim = len(categoricalColumns)
  numericDim = len(numericColumns)
  if (categoricalDim + numericDim) == 0:
    raise ValueError("At least 1 feature column of categorical_columns or numeric_columns must be specified.")
  if not (useLinear or useMf or useDnn):
    raise ValueError("At least 1 of linear, mf or dnn component must be used.")

  logits = 0
  if useLinear:
    with tf.compat.v1.variable_scope("linear"):
      linear_logit = tf.compat.v1.feature_column.linear_model(features, categoricalColumns + numericColumns)
      # [None, 1]

      with tf.name_scope("linear"):
        layerSummary(linear_logit)
      logits += linear_logit
      # [None, 1]

  if useMf or useDnn:
    with tf.compat.v1.variable_scope("input_layer"):
      # categorical input
      categoricalDim = len(categoricalColumns)
      if categoricalDim > 0:
        embeddingColumns = [tf.feature_column.embedding_column(col, embeddingSize)
                            for col in categoricalColumns]
        embeddingInputs = tf.compat.v1.feature_column.input_layer(features, embeddingColumns)
        # [None, c_d * embedding_size]
        inputLayer = embeddingInputs
        # [None, c_d * embedding_size]

      # numeric input
      numericDim = len(numericColumns)
      if numericDim > 0:
        numericInputs = tf.expand_dims(tf.compat.v1.feature_column.input_layer(features, numericColumns), -1)
        # [None, n_d, 1]
        numericEmbeddings = tf.compat.v1.get_variable("numeric_embeddings", [1, numericDim, embeddingSize])
        # [1, n_d, embedding_size]
        numericEmbeddingInputs = tf.reshape(numericEmbeddings * numericInputs,
                                            [-1, numericDim * embeddingSize])
        # [None, n_d * embedding_size]
        inputLayer = numericEmbeddingInputs
        # [None, n_d * embedding_size]

        if categoricalDim > 0:
          inputLayer = tf.concat([embeddingInputs, numericEmbeddingInputs], 1)
          # [None, d * embedding_size]

    if useMf:
      with tf.compat.v1.variable_scope("mf"):
        # reshape flat embedding input layer to matrix
        embeddingMat = tf.reshape(inputLayer, [-1, categoricalDim + numericDim, embeddingSize])
        # [None, d, embedding_size]
        sumSquare = tf.square(tf.reduce_sum(embeddingMat, 1))
        # [None, embedding_size]
        squareSum = tf.reduce_sum(tf.square(embeddingMat), 1)
        # [None, embedding_size]

        with tf.name_scope("logits"):
          mfLogit = 0.5 * tf.reduce_sum(sumSquare - squareSum, 1, keepdims=True)
          # [None, 1]
          layerSummary(mfLogit)
        logits += mfLogit
        # [None, 1]

    if useDnn:
      with tf.compat.v1.variable_scope("dnn/dnn"):
        net = inputLayer
        # [None, d * embedding_size]

        for i, hidden_size in enumerate(hiddenUnits):
          with tf.compat.v1.variable_scope("hiddenlayer_%s" % i):
            net = tf.compat.v1.layers.dense(net, hidden_size, activation=activationFn)
            # [None, hidden_size]
            if dropout > 0 and mode == tf.estimator.ModeKeys.TRAIN:
              net = tf.compat.v1.layers.dropout(net, rate=dropout, training=True)
              # [None, hidden_size]
            layerSummary(net)

        with tf.compat.v1.variable_scope('logits'):
          dnnLogit = tf.compat.v1.layers.dense(net, 1)
          # [None, 1]
          layerSummary(dnnLogit)
        logits += dnnLogit
        # [None, 1]

  with tf.name_scope("deep_fm/logits"):
    layerSummary(logits)

  inputs = tf.keras.Input(shape=(3,))
  x = tf.keras.layers.Dense(4, activation=tf.nn.relu)(inputs)
  outputs = tf.keras.layers.Dense(5, activation=tf.nn.softmax)(x)
  model = tf.keras.Model(inputs=inputs, outputs=outputs) #TODO fix this https://colab.research.google.com/github/tensorflow/lattice/blob/master/docs/tutorials/custom_estimators.ipynb#scrollTo=n2Zrv6OPaQO2
  # model = tf.keras.Model()

  optimizer = getOptimizer(optimizer, learningRate)
  head = tf.estimator.BinaryClassHead()
  return head.create_estimator_spec(
    features=features,
    mode=mode,
    labels=labels,
    optimizer=optimizer,
    logits=logits,
    trainable_variables=model.trainable_variables
  )


def trainAndEvaluate(args):
  # paths
  trainCSV = args.train_csv
  testCSV = args.test_csv
  jobDir = args.job_dir
  restore = args.restore
  # model
  useLinear = not args.exclude_linear,
  useMf = not args.exclude_mf,
  useDnn = not args.exclude_dnn,
  embeddingSize = args.embedding_size
  hiddenUnits = args.hidden_units
  dropout = args.dropout
  # training
  batchSize = args.batch_size
  trainSteps = args.train_steps

  # init
  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
  if not restore:
    shutil.rmtree(jobDir, ignore_errors=True)

  # estimator
  featureColumns = getFeatureColumns(embeddingSize=embeddingSize)
  runConfig = getRunConfig()
  estimator = tf.estimator.Estimator(
    model_fn=modelFn,
    model_dir=jobDir,
    config=runConfig,
    params={
      "categorical_columns": featureColumns["linear"],
      "use_linear": useLinear,
      "use_mf": useMf,
      "use_dnn": useDnn,
      "embedding_size": embeddingSize,
      "hidden_units": hiddenUnits,
      "dropout": dropout,
    }
  )

  # train spec
  trainInputFn = getInput(trainCSV, batchSize=batchSize)
  trainSpec = getTrainSpec(trainInputFn, trainSteps)

  # eval spec
  evalInputFn = getInput(testCSV, tf.estimator.ModeKeys.EVAL, batchSize=batchSize)
  exporter = getExporter(servingInput)
  evalSpec = getEvalSpec(evalInputFn, exporter)

  # train and evaluate
  tf.estimator.train_and_evaluate(estimator, trainSpec, evalSpec)


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument("--train-csv", default="data/ml-100k/train.csv",
                      help="path to the training csv data (default: %(default)s)")
  parser.add_argument("--test-csv", default="data/ml-100k/test.csv",
                      help="path to the test csv data (default: %(default)s)")
  parser.add_argument("--job-dir", default="checkpoints/deep_fm",
                      help="job directory (default: %(default)s)")
  parser.add_argument("--restore", action="store_true",
                      help="whether to restore from job_dir")
  parser.add_argument("--exclude-linear", action="store_true",
                      help="flag to exclude linear component (default: %(default)s)")
  parser.add_argument("--exclude-mf", action="store_true",
                      help="flag to exclude mf component (default: %(default)s)")
  parser.add_argument("--exclude-dnn", action="store_true",
                      help="flag to exclude dnn component (default: %(default)s)")
  parser.add_argument("--embedding-size", type=int, default=4,
                      help="embedding size (default: %(default)s)")
  parser.add_argument("--hidden-units", type=int, nargs='+', default=[16, 16],
                      help="hidden layer specification (default: %(default)s)")
  parser.add_argument("--dropout", type=float, default=0.1,
                      help="dropout rate (default: %(default)s)")
  parser.add_argument("--batch-size", type=int, default=32,
                      help="batch size (default: %(default)s)")
  parser.add_argument("--train-steps", type=int, default=20000,
                      help="number of training steps (default: %(default)s)")
  args = parser.parse_args()

  trainAndEvaluate(args)
