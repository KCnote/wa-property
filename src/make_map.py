import boto3
import awswrangler as wr
import folium
from folium.plugins import MarkerCluster
from branca.element import MacroElement
from jinja2 import Template

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

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
      AND bedrooms IS NOT NULL
      AND bathrooms IS NOT NULL
      AND garage IS NOT NULL
      AND land_area IS NOT NULL
      AND floor_area IS NOT NULL
      AND bedrooms <= 5
      AND bathrooms <= 3
      AND garage <= 2
      AND land_area BETWEEN 400 AND 800
    """

    return wr.athena.read_sql_query(
        sql=sql,
        database=DATABASE,
        s3_output=ATHENA_OUTPUT,
    )


def add_kmeans_cluster(df, n_clusters=6):
    features = [
        "latitude",
        "longitude",
        "price",
        "bedrooms",
        "bathrooms",
        "garage",
        "land_area",
        "floor_area",
    ]

    model_df = df.dropna(subset=features).copy()

    if model_df.empty:
        raise RuntimeError("No valid rows for KMeans clustering.")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(model_df[features])

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init="auto",
    )

    model_df["cluster_group"] = kmeans.fit_predict(X_scaled)

    df = df.copy()
    df["cluster_group"] = None
    df.loc[model_df.index, "cluster_group"] = model_df["cluster_group"]

    return df


def price_color(price):
    if price < 400000:
        return "#1a9850"
    if price < 500000:
        return "#66bd63"
    if price < 600000:
        return "#a6d96a"
    if price < 700000:
        return "#fee08b"
    if price < 800000:
        return "#fdae61"
    if price < 900000:
        return "#f46d43"
    if price < 1000000:
        return "#d73027"
    return "#a50026"


def land_radius(land_area):
    land_area = float(land_area)

    if land_area < 450:
        return 4
    if land_area < 500:
        return 5
    if land_area < 600:
        return 6
    if land_area < 700:
        return 7
    if land_area < 800:
        return 8
    return 9


def cluster_icon_function():
    return """
    function(cluster) {
        var markers = cluster.getAllChildMarkers();
        var sum = 0;
        var count = 0;

        markers.forEach(function(marker) {
            if (marker.options.price !== undefined && marker.options.price !== null) {
                sum += Number(marker.options.price);
                count += 1;
            }
        });

        var avg = count > 0 ? sum / count : 0;

        function getColor(price) {
            if (price < 400000) return "#1a9850";
            if (price < 500000) return "#66bd63";
            if (price < 600000) return "#a6d96a";
            if (price < 700000) return "#fee08b";
            if (price < 800000) return "#fdae61";
            if (price < 900000) return "#f46d43";
            if (price < 1000000) return "#d73027";
            return "#a50026";
        }

        function getTextColor(bgColor) {
            var c = bgColor.substring(1);
            var rgb = parseInt(c, 16);
            var r = (rgb >> 16) & 0xff;
            var g = (rgb >> 8) & 0xff;
            var b = rgb & 0xff;
            var brightness = (r * 299 + g * 587 + b * 114) / 1000;
            return brightness > 150 ? "black" : "white";
        }

        var color = getColor(avg);
        var textColor = getTextColor(color);
        var avgText = "$" + Math.round(avg / 1000) + "k";

        return L.divIcon({
            html:
                '<div style="' +
                'background:' + color + ';' +
                'color:' + textColor + ';' +
                'border:3px solid white;' +
                'border-radius:50%;' +
                'width:58px;' +
                'height:58px;' +
                'display:flex;' +
                'flex-direction:column;' +
                'align-items:center;' +
                'justify-content:center;' +
                'font-size:12px;' +
                'font-weight:bold;' +
                'box-shadow:0 0 6px rgba(0,0,0,0.45);' +
                '">' +
                '<div>' + avgText + '</div>' +
                '<div style="font-size:10px;">' + count + ' homes</div>' +
                '</div>',
            className: "price-cluster-icon",
            iconSize: [58, 58]
        });
    }
    """


class ClusterAreaLayer(MacroElement):
    def __init__(self, map_name, cluster_name):
        super().__init__()
        self._name = "ClusterAreaLayer"
        self.map_name = map_name
        self.cluster_name = cluster_name

        self._template = Template("""
        {% macro script(this, kwargs) %}

        var clusterAreaLayer = L.layerGroup().addTo({{ this.map_name }});

        function getAvgPriceColor(price) {
            if (price < 400000) return "#1a9850";
            if (price < 500000) return "#66bd63";
            if (price < 600000) return "#a6d96a";
            if (price < 700000) return "#fee08b";
            if (price < 800000) return "#fdae61";
            if (price < 900000) return "#f46d43";
            if (price < 1000000) return "#d73027";
            return "#a50026";
        }

        function getClusterAveragePrice(cluster) {
            var markers = cluster.getAllChildMarkers();
            var sum = 0;
            var count = 0;

            markers.forEach(function(marker) {
                if (marker.options.price !== undefined && marker.options.price !== null) {
                    sum += Number(marker.options.price);
                    count += 1;
                }
            });

            return {
                avg: count > 0 ? sum / count : 0,
                count: count
            };
        }

        function redrawClusterAreas() {
            clusterAreaLayer.clearLayers();

            {{ this.cluster_name }}._featureGroup.eachLayer(function(layer) {
                if (layer instanceof L.MarkerCluster) {
                    var result = getClusterAveragePrice(layer);
                    var avg = result.avg;
                    var color = getAvgPriceColor(avg);
                    var bounds = layer.getBounds();

                    var rectangle = L.rectangle(bounds, {
                        color: color,
                        weight: 2,
                        fillColor: color,
                        fillOpacity: 0.16,
                        interactive: false
                    });

                    rectangle.addTo(clusterAreaLayer);
                    rectangle.bringToBack();
                }
            });
        }

        {{ this.map_name }}.on("zoomend moveend", function() {
            redrawClusterAreas();
        });

        {{ this.cluster_name }}.on("animationend", function() {
            redrawClusterAreas();
        });

        setTimeout(redrawClusterAreas, 500);

        {% endmacro %}
        """)


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

    cluster = MarkerCluster(
        name="Average Property Price Cluster",
        icon_create_function=cluster_icon_function(),
        options={
            "showCoverageOnHover": False,
            "removeOutsideVisibleBounds": True,
            "spiderfyOnMaxZoom": True,
            "disableClusteringAtZoom": 17,
            "maxClusterRadius": 80,
        },
    ).add_to(m)

    for _, row in df.iterrows():
        price = float(row["price"])
        color = price_color(price)
        radius = land_radius(row["land_area"])

        popup = f"""
        <b>{row["address"]}</b><br>
        Price: ${price:,.0f}<br>
        Bedrooms: {row["bedrooms"]}<br>
        Bathrooms: {row["bathrooms"]}<br>
        Garage: {row["garage"]}<br>
        Floor area: {row["floor_area"]}<br>
        KMeans Area Group: {row["cluster_group"]}
        """

        marker = folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.75,
            weight=1,
            popup=folium.Popup(popup, max_width=300),
        )

        marker.options["price"] = price
        marker.add_to(cluster)

    area_layer = ClusterAreaLayer(
        map_name=m.get_name(),
        cluster_name=cluster.get_name(),
    )

    m.get_root().add_child(area_layer)

    folium.LayerControl().add_to(m)

    return m


def upload_to_s3():
    s3 = boto3.client("s3")

    s3.upload_file(
        OUTPUT_HTML,
        DEPLOY_BUCKET,
        OUTPUT_HTML,
        ExtraArgs={
            "ContentType": "text/html; charset=utf-8",
            "CacheControl": "no-cache",
        },
    )


def main():
    df = load_data()

    print("Loaded rows:", len(df))
    print(df.head())

    df = add_kmeans_cluster(df, n_clusters=6)

    print("KMeans cluster counts:")
    print(df["cluster_group"].value_counts().sort_index())

    m = create_map(df)
    m.save(OUTPUT_HTML)

    print(f"Generated {OUTPUT_HTML}")

    upload_to_s3()

    print(f"Uploaded to s3://{DEPLOY_BUCKET}/{OUTPUT_HTML}")


if __name__ == "__main__":
    main()