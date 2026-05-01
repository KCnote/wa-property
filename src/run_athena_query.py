import boto3
import time
from pathlib import Path

athena = boto3.client("athena", region_name="ap-southeast-2")

DATABASE = "wa_property_db"
OUTPUT = "s3://personal-wa-property-storage-337164669284-ap-southeast-2-an/athena-results/"

QUERY_FILES = [
    "sql/00_create_database.sql",
    "sql/01_create_raw_table.sql",
    "sql/02_clean_property.sql",
]

def run_query_file(query_file):
    query = Path(query_file).read_text()

    response = athena.start_query_execution(
        QueryString=query,
        ResultConfiguration={"OutputLocation": OUTPUT},
    )

    query_id = response["QueryExecutionId"]
    print(f"Running {query_file}: {query_id}")

    while True:
        result = athena.get_query_execution(QueryExecutionId=query_id)
        status_info = result["QueryExecution"]["Status"]
        status = status_info["State"]

        if status in ["SUCCEEDED", "FAILED", "CANCELLED"]:
            print(f"{query_file} status:", status)

            if status != "SUCCEEDED":
                reason = status_info.get("StateChangeReason", "")
                raise RuntimeError(f"{query_file} failed: {reason}")

            break

        time.sleep(2)

def main():
    for query_file in QUERY_FILES:
        run_query_file(query_file)

if __name__ == "__main__":
    main()