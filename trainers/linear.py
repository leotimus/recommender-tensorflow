import shutil
from argparse import ArgumentParser

import tensorflow as tf

from trainers.conf_utils import getRunConfig, getTrainSpec, getExporter, getEvalSpec
from trainers.ml_100k import getFeatureColumns, getInput, servingInput


def trainAndEvaluate(args):
  # paths
  trainCsv = args.train_csv
  testCsv = args.test_csv
  jobDir = args.job_dir
  restore = args.restore
  # model
  embeddingSize = args.embedding_size
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
  estimator = tf.estimator.LinearClassifier(
    feature_columns=featureColumns["linear"],
    model_dir=jobDir,
    config=runConfig
  )

  # train spec
  trainInputFn = getInput(trainCsv, batchSize=batchSize)
  trainSpec = getTrainSpec(trainInputFn, trainSteps)

  # eval spec
  evalInputFn = getInput(testCsv, tf.estimator.ModeKeys.EVAL, batchSize=batchSize)
  exporter = getExporter(servingInput)
  evalSpec = getEvalSpec(evalInputFn, exporter)

  # train and evaluate
  tf.estimator.train_and_evaluate(estimator, trainSpec, evalSpec)


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument("--train-csv", default="\\\\cs.aau.dk\\Fileshares\\IT703e20\\data4project_global.csv",
                      help="path to the training csv data (default: %(default)s)")
  parser.add_argument("--test-csv", default="data/ml-100k/test.csv",
                      help="path to the test csv data (default: %(default)s)")
  parser.add_argument("--restore", action="store_true",
                      help="whether to restore from job_dir")
  parser.add_argument("--job-dir", default="checkpoints/linear",
                      help="job directory (default: %(default)s)")
  parser.add_argument("--embedding-size", type=int, default=4,
                      help="batch size (default: %(default)s)")
  parser.add_argument("--batch-size", type=int, default=32,
                      help="batch size (default: %(default)s)")
  parser.add_argument("--train-steps", type=int, default=20000,
                      help="number of training steps (default: %(default)s)")
  args = parser.parse_args()

  trainAndEvaluate(args)
