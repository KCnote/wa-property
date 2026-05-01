import boto3
import time

athena = boto3.client("athena", region_name="ap-southeast-2")

DATABASE = "wa_property_db"
OUTPUT = "s3://personal-wa-property-storage-337164669284-ap-southeast-2-an/athena-results/"
QUERY_FILE = "sql/clean_property.sql"

def run_query():
    with open(QUERY_FILE, "r") as f:
        query = f.read()

    response = athena.start_query_execution(
        QueryString=query,
        QueryExecutionContext={"Database": DATABASE},
        ResultConfiguration={"OutputLocation": OUTPUT},
    )

    query_id = response["QueryExecutionId"]

    while True:
        result = athena.get_query_execution(QueryExecutionId=query_id)
        status = result["QueryExecution"]["Status"]["State"]

        if status in ["SUCCEEDED", "FAILED", "CANCELLED"]:
            print(status)
            break

        time.sleep(2)

if __name__ == "__main__":
    run_query()