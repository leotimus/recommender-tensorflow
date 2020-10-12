from google.api_core import exceptions
from google.cloud import bigquery
from google.oauth2 import service_account

from src.logger import getLogger

logger = getLogger(__name__)


def getCredentials(serviceAccountJson):
  scopes = ["https://www.googleapis.com/auth/cloud-platform"]
  credentials = (service_account.Credentials
                 .from_service_account_file(serviceAccountJson, scopes=scopes))
  logger.info("credentials created from %s.", serviceAccountJson)
  return credentials


def getBigQueryClient(serviceAccountJson):
  scopes = ["https://www.googleapis.com/auth/cloud-platform"]
  credentials = (service_account.Credentials
                 .from_service_account_file(serviceAccountJson, scopes=scopes))
  logger.info("credentials created from %s.", serviceAccountJson)

  # get client
  client = bigquery.Client(project=credentials.project_id, credentials=credentials)
  return client


def getBigQueryTable(tableId, datasetId, client):
  # get or create dataset
  try:
    dataset = client.get_dataset(datasetId)
  except exceptions.NotFound:
    dataset = client.create_dataset(datasetId)
    logger.info("dataset %s created, since not found.", datasetId)

  # get table
  table = dataset.table(tableId)
  return table


def dfToBigQuery(df, tableId, datasetId, client):
  table = getBigQueryTable(tableId, datasetId, client)

  # set config: insert overwrite
  jobConfig = bigquery.LoadJobConfig(
    write_disposition=bigquery.job.WriteDisposition.WRITE_TRUNCATE
  )

  # insert table
  job = client.load_table_from_dataframe(
    dataframe=df.compute().rename_axis("id"),
    destination=table,
    job_config=jobConfig
  )
  job.result()
  logger.info('%s rows loaded into %s.%s.%s.', job.output_rows, job.project, datasetId, tableId)
  return table


def bigQueryToTable(query, tableId, datasetId, client):
  table = getBigQueryTable(tableId, datasetId, client)

  # set config: insert overwrite to table
  jobConfig = bigquery.QueryJobConfig(
    destination=table,
    write_disposition=bigquery.job.WriteDisposition.WRITE_TRUNCATE
  )

  # query and insert to table
  job = client.query(query, job_config=jobConfig)
  job.result()
  logger.info("query results loaded to table: %s.", table.path)
  return table


def bigQueryToGCS(tableId, datasetId, path, bucket, client):
  table = getBigQueryTable(tableId, datasetId, client)
  destination_uri = "gs://{bucket}/{path}".format(bucket=bucket, path=path)

  job = client.extract_table(table, destination_uri)
  job.result()
  logger.info("exported %s:%s.%s to %s.",
              client.project, datasetId, tableId, destination_uri)
