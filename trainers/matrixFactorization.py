# %%

# @title Imports (run this cell)
from __future__ import print_function

import numpy as np
import pandas as pd
import collections
from matplotlib import pyplot as plt
import tensorflow.compat.v1 as tf
import altair as alt

# visualization library https://altair-viz.github.io/getting_started/installation.html

# in Tensorflow embeddings are the products M*N of matrix A factorization, so user and item embeddings - https://developers.google.com/machine-learning/recommendation/collaborative/matrix and handbook

tf.disable_v2_behavior()
tf.logging.set_verbosity(tf.logging.ERROR)

# Add some convenience functions to Pandas DataFrame. - just formatting
# DataFrame - Two-dimensional, size-mutable, potentially heterogeneous tabular data.
# Data structure also contains labeled axes (rows and columns).
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.3f}'.format


def mask(df, key, function):
  """Returns a filtered dataframe, by applying function to key"""
  return df[function(df[key])]


def flatten_cols(df):
  df.columns = [' '.join(col).strip() for col in df.columns.values]
  return df


pd.DataFrame.mask = mask
pd.DataFrame.flatten_cols = flatten_cols

alt.data_transformers.enable('default', max_rows=None)
alt.renderers.enable('colab')

# Load each data set (users, movies, and ratings).
# dataset is in separate csv file for each entity, therefore they must be loaded separately and combine into single dataset
users_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('data/ml-100k/u.user', sep='|', names=users_cols, encoding='latin-1')

ratings_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pd.read_csv('data/ml-100k/u.data', sep='\t', names=ratings_cols, encoding='latin-1')

# The movies file contains a binary feature for each genre (0,1). Therefore we have to define the genres manually and then merge with the other Movie columns
genre_cols = [
  "genre_unknown", "Action", "Adventure", "Animation", "Children", "Comedy",
  "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
  "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
]
movies_cols = [
                'movie_id', 'title', 'release_date', "video_release_date", "imdb_url"
              ] + genre_cols
movies = pd.read_csv(
  'data/ml-100k/u.item', sep='|', names=movies_cols, encoding='latin-1')

# Since the ids start at 1, we shift them to start at 0. - convenience for data manipulation
users["user_id"] = users["user_id"].apply(lambda x: str(x - 1))
movies["movie_id"] = movies["movie_id"].apply(lambda x: str(x - 1))
movies["year"] = movies['release_date'].apply(lambda x: str(x).split('-')[-1])
ratings["movie_id"] = ratings["movie_id"].apply(lambda x: str(x - 1))
ratings["user_id"] = ratings["user_id"].apply(lambda x: str(x - 1))
ratings["rating"] = ratings["rating"].apply(lambda x: float(x))

# Compute the number of movies to which a genre is assigned.
genre_occurences = movies[genre_cols].sum().to_dict()


# Since some movies can belong to more than one genre, we create different
# 'genre' columns as follows:
# - all_genres: all the active genres of the movie.
# - genre: randomly sampled from the active genres.
# to make data manipulation easier, in case a movie belongs to more than one genre, single genre is randomly selected
def mark_genres(movies, genres):
  def get_random_genre(gs):
    active = [genre for genre, g in zip(genres, gs) if g == 1]
    if len(active) == 0:
      return 'Other'
    return np.random.choice(active)

  def get_all_genres(gs):
    active = [genre for genre, g in zip(genres, gs) if g == 1]
    if len(active) == 0:
      return 'Other'
    return '-'.join(active)

  movies['genre'] = [
    get_random_genre(gs) for gs in zip(*[movies[genre] for genre in genres])]
  movies['all_genres'] = [
    get_all_genres(gs) for gs in zip(*[movies[genre] for genre in genres])]


mark_genres(movies, genre_cols)

# Create one merged DataFrame containing all the movielens data.
# combining ratings, movies and users into one big table
movielens = ratings.merge(movies, on='movie_id').merge(users, on='user_id')


# Utility to split the data into training and test sets.
def split_dataframe(df, holdout_fraction=0.1):
  """Splits a DataFrame into training and test sets.
  Args:
    df: a dataframe.
    holdout_fraction: fraction of dataframe rows to use in the test set. -- 10% of data will be used for test and 90% for training
  Returns:
    train: dataframe for training
    test: dataframe for testing
  """
  test = df.sample(frac=holdout_fraction, replace=False)
  train = df[~df.index.isin(test.index)]
  return train, test


# just for data exploration purposes
users.describe()

users.describe(include=[np.object])

# @title Altair visualization code (run this cell)
# The following functions are used to generate interactive Altair charts.
# We will display histograms of the data, sliced by a given attribute.

# Create filters to be used to slice the data.

# following code visualizes the initial dataframes using Altair - it might be useful for the report graphs
occupation_filter = alt.selection_multi(fields=["occupation"])
occupation_chart = alt.Chart().mark_bar().encode(
  x="count()",
  y=alt.Y("occupation:N"),
  color=alt.condition(
    occupation_filter,
    alt.Color("occupation:N", scale=alt.Scale(scheme='category20')),
    alt.value("lightgray")),
).properties(width=300, height=300, selection=occupation_filter)


# A function that generates a histogram of filtered data.
def filtered_hist(field, label, filter):
  """Creates a layered chart of histograms.
  The first layer (light gray) contains the histogram of the full data, and the
  second contains the histogram of the filtered data.
  Args:
    field: the field for which to generate the histogram.
    label: String label of the histogram.
    filter: an alt.Selection object to be used to filter the data.
  """
  base = alt.Chart().mark_bar().encode(
    x=alt.X(field, bin=alt.Bin(maxbins=10), title=label),
    y="count()",
  ).properties(
    width=300,
  )
  return alt.layer(
    base.transform_filter(filter),
    base.encode(color=alt.value('lightgray'), opacity=alt.value(.7)),
  ).resolve_scale(y='independent')

users_ratings = (
  ratings
    .groupby('user_id', as_index=False)
    .agg({'rating': ['count', 'mean']})
    .flatten_cols()
    .merge(users, on='user_id')
)

# Create a chart for the count, and one for the mean.
alt.hconcat(
  filtered_hist('rating count', '# ratings / user', occupation_filter),
  filtered_hist('rating mean', 'mean user rating', occupation_filter),
  occupation_chart,
  data=users_ratings)

# again data exploration - but its surely very important to do explorations of this kind with Grundfos data so itøs good for inspiration
movies_ratings = movies.merge(
  ratings
    .groupby('movie_id', as_index=False)
    .agg({'rating': ['count', 'mean']})
    .flatten_cols(),
  on='movie_id')

genre_filter = alt.selection_multi(fields=['genre'])
genre_chart = alt.Chart().mark_bar().encode(
  x="count()",
  y=alt.Y('genre'),
  color=alt.condition(
    genre_filter,
    alt.Color("genre:N"),
    alt.value('lightgray'))
).properties(height=300, selection=genre_filter)

(movies_ratings[['title', 'rating count', 'rating mean']]
 .sort_values('rating count', ascending=False)
 .head(10))

(movies_ratings[['title', 'rating count', 'rating mean']]
 .mask('rating count', lambda x: x > 20)
 .sort_values('rating mean', ascending=False)
 .head(10))

# Display the number of ratings and average rating per movie.
alt.hconcat(
  filtered_hist('rating count', '# ratings / movie', genre_filter),
  filtered_hist('rating mean', 'mean movie rating', genre_filter),
  genre_chart,
  data=movies_ratings)

def build_rating_sparse_tensor(ratings_df):
  """
  Args:
    ratings_df: a pd.DataFrame with `user_id`, `movie_id` and `rating` columns.
  Returns:
    a tf.SparseTensor representing the ratings matrix.
  """

  # a sparse tensor is used when we are working with a data that contain a lot of NULL values
  indices = ratings_df[['user_id', 'movie_id']].values
  values = ratings_df['rating'].values
  return tf.SparseTensor(
    indices=indices,
    values=values,
    dense_shape=[users.shape[0], movies.shape[0]])

def sparse_mean_square_error(sparse_ratings, user_embeddings, movie_embeddings):
  """
  Args:
    sparse_ratings: A SparseTensor rating matrix, of dense_shape [N, M]
    user_embeddings: A dense Tensor U of shape [N, k] where k is the embedding
      dimension, such that U_i is the embedding of user i.
    movie_embeddings: A dense Tensor V of shape [M, k] where k is the embedding
      dimension, such that V_j is the embedding of movie j.
  Returns:
    A scalar Tensor representing the MSE between the true ratings and the
      model's predictions.
  """
  # matmul multiplies matrix a and b, transpose_b means matrix b will be transposed - uv^t matrix from handbook
  # https://www.tensorflow.org/api_docs/python/tf/linalg/matmul

  # gather_nd gathers slices from param (uv^t) into a Tensor with shape specified by indices (user_id and movie_id)
  predictions = tf.gather_nd(
    tf.matmul(user_embeddings, movie_embeddings, transpose_b=True),
    sparse_ratings.indices)

  # MSE Computes the mean squared error between labels and predictions.
  # https://www.tensorflow.org/api_docs/python/tf/keras/losses/MSE
  loss = tf.losses.mean_squared_error(sparse_ratings.values, predictions)
  return loss


# @title CFModel helper class (run this cell)
# this CFModel class cannot be found in Tensorflow docs or elsewhere - it is not predefined.
# It was just written from scratch for the purpose of this CF task.
# However, because of the structure makes much sense we can maybe use it as a template when writing helper classes for the Grundfos project.
class CFModel(object):
  # in the beginning the args are initialized
  """Simple class that represents a collaborative filtering model"""

  def __init__(self, embedding_vars, loss, metrics=None):
    """Initializes a CFModel.
    Args:
      embedding_vars: A dictionary of tf.Variables.
      loss: A float Tensor. The loss to optimize.
      metrics: optional list of dictionaries of Tensors. The metrics in each
        dictionary will be plotted in a separate figure during training.
    """
    self._embedding_vars = embedding_vars
    self._loss = loss
    self._metrics = metrics
    self._embeddings = {k: None for k in embedding_vars}
    self._session = None

  @property
  def embeddings(self):
    """The embeddings dictionary."""
    return self._embeddings

  # gradient descent optimizer is used for the model training
  def train(self, num_iterations=100, learning_rate=1.0, plot_results=True,
            optimizer=tf.train.GradientDescentOptimizer):
    """Trains the model.
    Args:
      iterations: number of iterations to run.
      learning_rate: optimizer learning rate.
      plot_results: whether to plot the results at the end of training.
      optimizer: the optimizer to use. Default to GradientDescentOptimizer.
    Returns:
      The metrics dictionary evaluated at the last iteration.
    """
    with self._loss.graph.as_default():
      opt = optimizer(learning_rate)
      train_op = opt.minimize(self._loss)
      local_init_op = tf.group(
        tf.variables_initializer(opt.variables()),
        tf.local_variables_initializer())
      if self._session is None:
        self._session = tf.Session()
        with self._session.as_default():
          self._session.run(tf.global_variables_initializer())
          self._session.run(tf.tables_initializer())
          tf.train.start_queue_runners()

    with self._session.as_default():
      local_init_op.run()
      iterations = []

      # I am a bit unsure what the metrics does. it seems like it keeps track of the metrics for the x and y axis
      metrics = self._metrics or ({},)
      metrics_vals = [collections.defaultdict(list) for _ in self._metrics]

      # Train and append results.
      for i in range(num_iterations + 1):
        _, results = self._session.run((train_op, metrics))
        if (i % 10 == 0) or i == num_iterations:
          print("\r iteration %d: " % i + ", ".join(
            ["%s=%f" % (k, v) for r in results for k, v in r.items()]),
                end='')
          iterations.append(i)
          for metric_val, result in zip(metrics_vals, results):
            for k, v in result.items():
              metric_val[k].append(v)

      for k, v in self._embedding_vars.items():
        self._embeddings[k] = v.eval()

      if plot_results:
        # Plot the metrics.
        num_subplots = len(metrics) + 1
        fig = plt.figure()
        fig.set_size_inches(num_subplots * 10, 8)
        for i, metric_vals in enumerate(metrics_vals):
          ax = fig.add_subplot(1, num_subplots, i + 1)
          for k, v in metric_vals.items():
            ax.plot(iterations, v, label=k)
          ax.set_xlim([1, num_iterations])
          ax.legend()
      return results

def build_model(ratings, embedding_dim=3, init_stddev=1.):
  """
  Args:
    ratings: a DataFrame of the ratings
    embedding_dim: the dimension of the embedding vectors.
    init_stddev: float, the standard deviation of the random initial embeddings.
  Returns:
    model: a CFModel.
  """
  # Split the ratings DataFrame into train and test.
  train_ratings, test_ratings = split_dataframe(ratings)
  # SparseTensor representation of the train and test datasets.
  A_train = build_rating_sparse_tensor(train_ratings)
  A_test = build_rating_sparse_tensor(test_ratings)
  # Initialize the embeddings using a normal distribution.
  U = tf.Variable(tf.random_normal(
    [A_train.dense_shape[0], embedding_dim], stddev=init_stddev))
  V = tf.Variable(tf.random_normal(
    [A_train.dense_shape[1], embedding_dim], stddev=init_stddev))
  train_loss = sparse_mean_square_error(A_train, U, V)
  test_loss = sparse_mean_square_error(A_test, U, V)
  metrics = {
    'train_error': train_loss,
    'test_error': test_loss
  }
  embeddings = {
    "user_id": U,
    "movie_id": V
  }
  return CFModel(embeddings, train_loss, [metrics])

# Build the CF model and train it.
if __name__ == '__main__':
  model = build_model(ratings, embedding_dim=30, init_stddev=0.5)
  model.train(num_iterations=1000, learning_rate=10.)
