import tensorflow as tf

COLUMNS = ("user_id,item_id,rating,timestamp,datetime,year,month,day,week,dayofweek,"
           "age,gender,occupation,zipcode,zipcode1,zipcode2,zipcode3,"
           "title,release,video_release,imdb,unknown,action,adventure,animation,children,"
           "comedy,crime,documentary,drama,fantasy,filmnoir,horror,musical,mystery,romance,"
           "scifi,thriller,war,western,release_date,release_year").split(",")
GENRE = ("unknown,action,adventure,animation,children,comedy,crime,documentary,drama,fantasy,"
         "filmnoir,horror,musical,mystery,romance,scifi,thriller,war,western").split(",")
LABEL_COL = "rating"
DEFAULTS = [[0], [0], [0], [0], ["null"], [0], [0], [0], [0], [0],
            [0], ["null"], ["null"], ["null"], ["null"], ["null"], ["null"],
            ["null"], ["null"], ["null"], ["null"], [0], [0], [0], [0], [0],
            [0], [0], [0], [0], [0], [0], [0], [0], [0], [0],
            [0], [0], [0], [0], ["null"], [0]]


def getFeatureColumns(embeddingSize=4):
  userFc = tf.feature_column.categorical_column_with_hash_bucket("CUSTOMER_ID", 1000, tf.int32)
  itemFc = tf.feature_column.categorical_column_with_hash_bucket("MATERIAL", 2000, tf.int32)

  # user features
  ageFc = tf.feature_column.numeric_column("age")
  ageBuckets = tf.feature_column.bucketized_column(ageFc, list(range(15, 66, 10)))
  genderFc = tf.feature_column.categorical_column_with_vocabulary_list(
    "gender", ["F", "M"],
    num_oov_buckets=1
  )
  occupationFc = tf.feature_column.categorical_column_with_hash_bucket("occupation", 50)
  zipCodeFc = tf.feature_column.categorical_column_with_hash_bucket("zipcode", 1000)

  # item features
  releaseYearFc = tf.feature_column.numeric_column("release_year")
  releaseYearBuckets = tf.feature_column.bucketized_column(releaseYearFc, list(range(1930, 1991, 10)))
  genreFc = [tf.feature_column.categorical_column_with_identity(col, 2) for col in GENRE]

  linearColumns = [userFc, itemFc, ageBuckets, genderFc, occupationFc, zipCodeFc, releaseYearBuckets] + genreFc
  deepColumns = [tf.feature_column.embedding_column(fc, embeddingSize) for fc in linearColumns]
  return {"linear": linearColumns, "deep": deepColumns}


def getInput(csvPath, mode=tf.estimator.ModeKeys.TRAIN, batchSize=32, cutoff=5):
  def inputFn():
    def parseCsv(value):
      columns = tf.compat.v1.decode_csv(value, DEFAULTS)
      features = dict(zip(COLUMNS, columns))
      label = features.pop(LABEL_COL)
      label = tf.math.greater_equal(label, cutoff)
      return features, label

    # read, parse, shuffle and batch dataset
    dataset = tf.data.TextLineDataset(csvPath).skip(1)  # skip header
    if mode == tf.estimator.ModeKeys.TRAIN:
      # shuffle and repeat
      dataset = dataset.shuffle(16 * batchSize).repeat()

    dataset = dataset.map(parseCsv, num_parallel_calls=8)
    dataset = dataset.batch(batchSize)
    return dataset

  return inputFn


def servingInput():
  featurePlaceholders = {
    "user_id": tf.compat.v1.placeholder(tf.int32, [None]),
    "item_id": tf.compat.v1.placeholder(tf.int32, [None]),

    "age": tf.compat.v1.placeholder(tf.int32, [None]),
    "gender": tf.compat.v1.placeholder(tf.string, [None]),
    "occupation": tf.compat.v1.placeholder(tf.string, [None]),
    "zipcode": tf.compat.v1.placeholder(tf.string, [None]),

    "release_year": tf.compat.v1.placeholder(tf.int32, [None]),
  }
  featurePlaceholders.update({
    col: tf.compat.v1.placeholder_with_default(tf.constant([0]), [None]) for col in GENRE
  })

  features = {
    key: tf.expand_dims(tensor, -1)
    for key, tensor in featurePlaceholders.items()
  }

  return tf.estimator.export.ServingInputReceiver(
    features=features,
    receiver_tensors=featurePlaceholders
  )
