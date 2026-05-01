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
      latitude,
      longitude
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

    model_df["house_group"] = kmeans.fit_predict(X_scaled)

    df = df.copy()
    df["house_group"] = None
    df.loc[model_df.index, "house_group"] = model_df["house_group"]

    return df


def house_category_name(group):
    mapping = {
        0: "Affordable Suburbs",
        1: "Family Housing",
        2: "Premium Housing",
        3: "Inner-city High Price",
        4: "Large Land Value Homes",
        5: "Compact Budget Homes",
    }

    if group is None:
        return "Unknown"

    return mapping.get(int(group), "Unknown")


def house_category_description(group):
    mapping = {
        0: "Lower-price suburban homes",
        1: "Typical family-oriented homes",
        2: "Higher-price premium properties",
        3: "Expensive homes in compact or central areas",
        4: "Homes with relatively larger land value",
        5: "Smaller and more budget-friendly homes",
    }

    if group is None:
        return "Unknown"

    return mapping.get(int(group), "Unknown")


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


def house_type_color(group):
    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#000000",
    ]

    if group is None:
        return "#666666"

    return colors[int(group) % len(colors)]


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


def price_cluster_icon_function():
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


def house_cluster_icon_function():
    return """
    function(cluster) {
        var markers = cluster.getAllChildMarkers();
        var counts = {};
        var total = markers.length;

        markers.forEach(function(marker) {
            var g = marker.options.houseGroup;
            if (g !== undefined && g !== null) {
                counts[g] = (counts[g] || 0) + 1;
            }
        });

        var majorityGroup = 0;
        var maxCount = 0;

        Object.keys(counts).forEach(function(g) {
            if (counts[g] > maxCount) {
                maxCount = counts[g];
                majorityGroup = g;
            }
        });

        function getHouseColor(group) {
            var colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#000000"];
            return colors[Number(group) % colors.length];
        }

        var color = getHouseColor(majorityGroup);

        return L.divIcon({
            html:
                '<div style="' +
                'background:' + color + ';' +
                'color:white;' +
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
                '<div>Type ' + majorityGroup + '</div>' +
                '<div style="font-size:10px;">' + total + ' homes</div>' +
                '</div>',
            className: "house-cluster-icon",
            iconSize: [58, 58]
        });
    }
    """


class ClusterAreaLayer(MacroElement):
    def __init__(self, map_name, parent_layer_name, cluster_name, mode):
        super().__init__()
        self._name = "ClusterAreaLayer"
        self.map_name = map_name
        self.parent_layer_name = parent_layer_name
        self.cluster_name = cluster_name
        self.mode = mode

        self._template = Template("""
        {% macro script(this, kwargs) %}

        var areaLayer_{{ this.cluster_name }} = L.layerGroup();

        function getPriceColor_{{ this.cluster_name }}(price) {
            if (price < 400000) return "#1a9850";
            if (price < 500000) return "#66bd63";
            if (price < 600000) return "#a6d96a";
            if (price < 700000) return "#fee08b";
            if (price < 800000) return "#fdae61";
            if (price < 900000) return "#f46d43";
            if (price < 1000000) return "#d73027";
            return "#a50026";
        }

        function getHouseColor_{{ this.cluster_name }}(group) {
            var colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#000000"];
            return colors[Number(group) % colors.length];
        }

        function getClusterInfo_{{ this.cluster_name }}(cluster) {
            var markers = cluster.getAllChildMarkers();

            var priceSum = 0;
            var priceCount = 0;
            var groupCounts = {};

            markers.forEach(function(marker) {
                if (marker.options.price !== undefined && marker.options.price !== null) {
                    priceSum += Number(marker.options.price);
                    priceCount += 1;
                }

                if (marker.options.houseGroup !== undefined && marker.options.houseGroup !== null) {
                    var g = marker.options.houseGroup;
                    groupCounts[g] = (groupCounts[g] || 0) + 1;
                }
            });

            var majorityGroup = 0;
            var maxCount = 0;

            Object.keys(groupCounts).forEach(function(g) {
                if (groupCounts[g] > maxCount) {
                    maxCount = groupCounts[g];
                    majorityGroup = g;
                }
            });

            return {
                avgPrice: priceCount > 0 ? priceSum / priceCount : 0,
                majorityGroup: majorityGroup
            };
        }

        function redrawArea_{{ this.cluster_name }}() {
            areaLayer_{{ this.cluster_name }}.clearLayers();

            if (!{{ this.map_name }}.hasLayer({{ this.parent_layer_name }})) {
                if ({{ this.map_name }}.hasLayer(areaLayer_{{ this.cluster_name }})) {
                    {{ this.map_name }}.removeLayer(areaLayer_{{ this.cluster_name }});
                }
                return;
            }

            if (!{{ this.map_name }}.hasLayer(areaLayer_{{ this.cluster_name }})) {
                areaLayer_{{ this.cluster_name }}.addTo({{ this.map_name }});
            }

            {{ this.cluster_name }}._featureGroup.eachLayer(function(layer) {
                if (layer instanceof L.MarkerCluster) {
                    var info = getClusterInfo_{{ this.cluster_name }}(layer);
                    var bounds = layer.getBounds();

                    var color = "{{ this.mode }}" === "price"
                        ? getPriceColor_{{ this.cluster_name }}(info.avgPrice)
                        : getHouseColor_{{ this.cluster_name }}(info.majorityGroup);

                    var rectangle = L.rectangle(bounds, {
                        color: color,
                        weight: 2,
                        fillColor: color,
                        fillOpacity: 0.13,
                        interactive: false
                    });

                    rectangle.addTo(areaLayer_{{ this.cluster_name }});
                    rectangle.bringToBack();
                }
            });
        }

        {{ this.map_name }}.on("zoomend moveend overlayadd overlayremove", function() {
            redrawArea_{{ this.cluster_name }}();
        });

        {{ this.cluster_name }}.on("animationend", function() {
            redrawArea_{{ this.cluster_name }}();
        });

        setTimeout(redrawArea_{{ this.cluster_name }}, 700);

        {% endmacro %}
        """)


def popup_html(row):
    price = float(row["price"])
    house_type = house_category_name(row["house_group"])
    description = house_category_description(row["house_group"])

    return f"""
    <b>{row["address"]}</b><br>
    Price: ${price:,.0f}<br>
    Bedrooms: {row["bedrooms"]}<br>
    Bathrooms: {row["bathrooms"]}<br>
    Garage: {row["garage"]}<br>
    Floor area: {row["floor_area"]}<br>
    <b>House Type:</b> {house_type}<br>
    <small>{description}</small>
    """


def add_legend(m):
    legend_html = """
    <div style="
        position: fixed;
        bottom: 30px;
        left: 30px;
        z-index: 9999;
        background: white;
        padding: 12px;
        border: 2px solid #999;
        border-radius: 8px;
        font-size: 13px;
        box-shadow: 0 0 8px rgba(0,0,0,0.3);
        max-width: 330px;
    ">
        <b>Map Meaning</b><br><br>

        <b>Price View</b><br>
        Fill colour and area shade = average price<br><br>

        <span style="background:#1a9850;width:14px;height:14px;display:inline-block;"></span> &lt; $400k<br>
        <span style="background:#66bd63;width:14px;height:14px;display:inline-block;"></span> $400k - $500k<br>
        <span style="background:#a6d96a;width:14px;height:14px;display:inline-block;"></span> $500k - $600k<br>
        <span style="background:#fee08b;width:14px;height:14px;display:inline-block;"></span> $600k - $700k<br>
        <span style="background:#fdae61;width:14px;height:14px;display:inline-block;"></span> $700k - $800k<br>
        <span style="background:#f46d43;width:14px;height:14px;display:inline-block;"></span> $800k - $900k<br>
        <span style="background:#d73027;width:14px;height:14px;display:inline-block;"></span> $900k - $1M<br>
        <span style="background:#a50026;width:14px;height:14px;display:inline-block;"></span> $1M+<br><br>

        <b>House Classification View</b><br>
        Fill colour and area shade = house type<br><br>

        <span style="background:#1f77b4;width:14px;height:14px;display:inline-block;"></span> Affordable Suburbs<br>
        <span style="background:#ff7f0e;width:14px;height:14px;display:inline-block;"></span> Family Housing<br>
        <span style="background:#2ca02c;width:14px;height:14px;display:inline-block;"></span> Premium Housing<br>
        <span style="background:#d62728;width:14px;height:14px;display:inline-block;"></span> Inner-city High Price<br>
        <span style="background:#9467bd;width:14px;height:14px;display:inline-block;"></span> Large Land Value Homes<br>
        <span style="background:#000000;width:14px;height:14px;display:inline-block;"></span> Compact Budget Homes<br><br>

        <b>Point Size = Land Area</b>
    </div>
    """

    m.get_root().html.add_child(folium.Element(legend_html))


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

    price_layer = folium.FeatureGroup(
        name="Price View",
        show=True,
    ).add_to(m)

    house_layer = folium.FeatureGroup(
        name="House Classification View",
        show=False,
    ).add_to(m)

    price_cluster = MarkerCluster(
        name="Price Cluster",
        icon_create_function=price_cluster_icon_function(),
        options={
            "showCoverageOnHover": False,
            "removeOutsideVisibleBounds": True,
            "spiderfyOnMaxZoom": True,
            "disableClusteringAtZoom": 17,
            "maxClusterRadius": 80,
        },
    ).add_to(price_layer)

    house_cluster = MarkerCluster(
        name="House Classification Cluster",
        icon_create_function=house_cluster_icon_function(),
        options={
            "showCoverageOnHover": False,
            "removeOutsideVisibleBounds": True,
            "spiderfyOnMaxZoom": True,
            "disableClusteringAtZoom": 17,
            "maxClusterRadius": 80,
        },
    ).add_to(house_layer)

    for _, row in df.iterrows():
        price = float(row["price"])
        radius = land_radius(row["land_area"])
        html = popup_html(row)

        price_marker = folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=radius,
            color="white",
            fill=True,
            fill_color=price_color(price),
            fill_opacity=0.80,
            weight=2,
            popup=folium.Popup(html, max_width=320),
        )
        price_marker.options["price"] = price
        price_marker.options["houseGroup"] = int(row["house_group"])
        price_marker.add_to(price_cluster)

        house_marker = folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=radius,
            color="white",
            fill=True,
            fill_color=house_type_color(row["house_group"]),
            fill_opacity=0.82,
            weight=2,
            popup=folium.Popup(html, max_width=320),
        )
        house_marker.options["price"] = price
        house_marker.options["houseGroup"] = int(row["house_group"])
        house_marker.add_to(house_cluster)

    price_area_layer = ClusterAreaLayer(
        map_name=m.get_name(),
        parent_layer_name=price_layer.get_name(),
        cluster_name=price_cluster.get_name(),
        mode="price",
    )

    house_area_layer = ClusterAreaLayer(
        map_name=m.get_name(),
        parent_layer_name=house_layer.get_name(),
        cluster_name=house_cluster.get_name(),
        mode="house",
    )

    m.get_root().add_child(price_area_layer)
    m.get_root().add_child(house_area_layer)

    add_legend(m)

    folium.LayerControl(collapsed=False).add_to(m)

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

    print("House classification counts:")
    print(df["house_group"].value_counts().sort_index())

    print("House classification summary:")
    print(
        df.groupby("house_group")[
            ["price", "bedrooms", "bathrooms", "garage", "land_area", "floor_area"]
        ].mean()
    )

    m = create_map(df)
    m.save(OUTPUT_HTML)

    print(f"Generated {OUTPUT_HTML}")

    upload_to_s3()

    print(f"Uploaded to s3://{DEPLOY_BUCKET}/{OUTPUT_HTML}")


if __name__ == "__main__":
    main()