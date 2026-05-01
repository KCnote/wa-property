import boto3
import time
from pathlib import Path
from datetime import datetime, timezone

athena = boto3.client("athena", region_name="ap-southeast-2")

DATABASE = "wa_property_db"
OUTPUT = "s3://personal-wa-property-storage-337164669284-ap-southeast-2-an/athena-results/"

QUERY_FILES = [
    "sql/01_create_database.sql",
    "sql/02_drop_raw_table.sql",
    "sql/03_create_raw_table.sql",
    "sql/04_clean_property.sql",
]


def run_query(query, name=""):
    response = athena.start_query_execution(
        QueryString=query,
        QueryExecutionContext={"Database": DATABASE},
        ResultConfiguration={"OutputLocation": OUTPUT},
    )

    query_id = response["QueryExecutionId"]
    print(f"Running {name}: {query_id}")

    while True:
        result = athena.get_query_execution(QueryExecutionId=query_id)
        status_info = result["QueryExecution"]["Status"]
        status = status_info["State"]

        if status in ["SUCCEEDED", "FAILED", "CANCELLED"]:
            print(f"{name} status:", status)

            if status != "SUCCEEDED":
                reason = status_info.get("StateChangeReason", "")
                raise RuntimeError(f"{name} failed: {reason}")

            break

        time.sleep(2)


def main():
    # 🔥 run_id 생성 (utc deprecated 해결)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    output_path = (
        "s3://personal-wa-property-storage-337164669284-ap-southeast-2-an/"
        f"processed/property_clean/run_id={run_id}/"
    )

    table_name = f"wa_property_clean_{run_id}"

    print("Run ID:", run_id)
    print("Output path:", output_path)
    print("Table name:", table_name)

    # 1. SQL 파일 실행
    for query_file in QUERY_FILES:
        query = Path(query_file).read_text()

        query = query.format(
            output_path=output_path,
            table_name=table_name,
        )

        run_query(query, query_file)

    # 🔥 2. 최신 view 자동 생성
    view_query = f"""
    CREATE OR REPLACE VIEW wa_property_db.wa_property_latest AS
    SELECT *
    FROM wa_property_db.{table_name}
    """

    run_query(view_query, "create_latest_view")

    print("✅ Latest view updated:", table_name)


if __name__ == "__main__":
    main()