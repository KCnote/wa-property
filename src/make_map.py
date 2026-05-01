import awswrangler as wr
import folium
from folium.plugins import MarkerCluster

DATABASE = "wa_property_db"
TABLE = "wa_property_latest"

ATHENA_OUTPUT = "s3://personal-wa-property-storage-337164669284-ap-southeast-2-an/athena-results/"
DEPLOY_BUCKET = "personal-wa-property-server-337164669284-ap-southeast-2-an"
OUTPUT_HTML = "index.html"


def load_data():
    sql = f"""
    SELECT
      address,
      suburb,
      price,
      bedrooms,
      bathrooms,
      garage,
      land_area,
      floor_area,
      date_sold,
      postcode,
      latitude,
      longitude,
      nearest_sch,
      nearest_sch_dist
    FROM {DATABASE}.{TABLE}
    WHERE latitude IS NOT NULL
      AND longitude IS NOT NULL
      AND price IS NOT NULL
    """

    return wr.athena.read_sql_query(
        sql=sql,
        database=DATABASE,
        s3_output=ATHENA_OUTPUT,
    )


def price_color(price):
    if price < 500000:
        return "green"
    if price < 800000:
        return "orange"
    return "red"


def create_map(df):
    if df.empty:
        raise RuntimeError("No data found from Athena.")

    center_lat = df["latitude"].mean()
    center_lon = df["longitude"].mean()

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=10,
        tiles="OpenStreetMap",
    )

    cluster = MarkerCluster().add_to(m)

    for _, row in df.iterrows():
        popup = f"""
        <b>{row["address"]}</b><br>
        Suburb: {row["suburb"]}<br>
        Price: ${row["price"]:,.0f}<br>
        Bedrooms: {row["bedrooms"]}<br>
        Bathrooms: {row["bathrooms"]}<br>
        Garage: {row["garage"]}<br>
        Land area: {row["land_area"]}<br>
        Floor area: {row["floor_area"]}<br>
        Sold: {row["date_sold"]}<br>
        School: {row["nearest_sch"]}<br>
        School dist: {row["nearest_sch_dist"]}
        """

        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=5,
            color=price_color(row["price"]),
            fill=True,
            fill_color=price_color(row["price"]),
            fill_opacity=0.75,
            popup=folium.Popup(popup, max_width=350),
        ).add_to(cluster)

    return m


def upload_to_s3():
    wr.s3.upload(
        local_file=OUTPUT_HTML,
        path=f"s3://{DEPLOY_BUCKET}/{OUTPUT_HTML}",
    )


def main():
    df = load_data()

    print("Loaded rows:", len(df))
    print(df.head())

    m = create_map(df)
    m.save(OUTPUT_HTML)

    print(f"Generated {OUTPUT_HTML}")

    upload_to_s3()

    print(f"Uploaded to s3://{DEPLOY_BUCKET}/{OUTPUT_HTML}")


if __name__ == "__main__":
    main()