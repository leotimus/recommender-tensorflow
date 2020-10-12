import dask.dataframe as dd
import numpy as np
import tensorflow as tf

from src.logger import getLogger

logger = getLogger(__name__)


def ddTfRecord(df, tfRecordPath):
  featureFunc = {
    np.int64: lambda value: tf.train.Feature(int64_list=tf.train.Int64List(value=[value])),
    np.float64: lambda value: tf.train.Feature(float_list=tf.train.FloatList(value=[value])),
    np.object_: lambda value: tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(value)]))
  }

  # convert types to tf base types
  df = df.copy()
  for col in df.columns:
    # convert non numeric types to string
    if df[col].dtype.type not in {np.int64, np.float64}:
      df[col] = df[col].astype(str)
  logger.debug("data column types: %s.", list(df.dtypes.items()))

  # configure feature specification base on column type
  colFunc = {colName: featureFunc[colType.type]
             for colName, colType in df.head().dtypes.items()}

  # save tfrecord
  with tf.compat.v1.python_io.TFRecordWriter(tfRecordPath) as writer:
    for row in df.itertuples():
      example = tf.train.Example(
        features=tf.train.Features(
          feature={
            name: func(getattr(row, name))
            for name, func in colFunc.items()
          }))
      writer.write(example.SerializeToString())
  logger.info("tfrecord saved: %s.", tfRecordPath)


def tfCsvDataset(csvPath, labelCol, colDefaults, shuffle=False, batchSize=32):
  df = dd.read_csv(csvPath)
  # use col_defaults if specified for col, else use defaults base on col type
  typeDefaults = {np.int64: 0, np.float64: 0.0, np.object_: ""}
  recordDefaults = [[colDefaults.get(colName, typeDefaults.get(colType.type, ""))]
                    for colName, colType in df.dtypes.items()]

  def parse_csv(value):
    columns = tf.compat.v1.decode_csv(value, recordDefaults)
    features = dict(zip(df.columns.tolist(), columns))
    label = features[labelCol]
    return features, label

  # read, parse, shuffle and batch dataset
  dataset = tf.data.TextLineDataset(csvPath).skip(1)  # skip header
  if shuffle:
    dataset = dataset.shuffle(buffer_size=1024)
  dataset = dataset.map(parse_csv, num_parallel_calls=8)
  dataset = dataset.batch(batchSize)
  return dataset


def ddCreateCategoricalColumn(df, col, defaultValue=-1, numOovBuckets=1):
  return tf.feature_column.categorical_column_with_vocabulary_list(
    col,
    df[col].unique().compute().sort_values().tolist(),
    default_value=defaultValue,
    num_oov_buckets=numOovBuckets
  )
