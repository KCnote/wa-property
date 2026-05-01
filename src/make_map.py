import boto3
import awswrangler as wr
import folium
from folium.plugins import MarkerCluster
from branca.element import MacroElement
from jinja2 import Template

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import BallTree

import numpy as np

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


def add_local_price_gap_zones(
    df,
    radius_m=500,
    min_price_gap=250000,
    max_pairs=250,
):
    work_df = df.dropna(
        subset=["latitude", "longitude", "price", "house_group"]
    ).copy()

    if work_df.empty:
        df["price_gap_zone"] = False
        return df, []

    coords_rad = np.radians(work_df[["latitude", "longitude"]].to_numpy())
    prices = work_df["price"].to_numpy()

    tree = BallTree(coords_rad, metric="haversine")

    earth_radius_m = 6371000
    radius_rad = radius_m / earth_radius_m

    neighbors = tree.query_radius(coords_rad, r=radius_rad)

    gap_pairs = []
    seen = set()

    for i, neighbor_indices in enumerate(neighbors):
        for j in neighbor_indices:
            if i == j:
                continue

            pair_key = tuple(sorted((i, j)))
            if pair_key in seen:
                continue

            seen.add(pair_key)

            price_gap = abs(float(prices[i]) - float(prices[j]))

            if price_gap >= min_price_gap:
                row_i = work_df.iloc[i]
                row_j = work_df.iloc[j]

                gap_pairs.append(
                    {
                        "i_index": row_i.name,
                        "j_index": row_j.name,
                        "lat1": row_i["latitude"],
                        "lon1": row_i["longitude"],
                        "lat2": row_j["latitude"],
                        "lon2": row_j["longitude"],
                        "address1": row_i["address"],
                        "address2": row_j["address"],
                        "price1": float(row_i["price"]),
                        "price2": float(row_j["price"]),
                        "gap": price_gap,
                    }
                )

    gap_pairs = sorted(gap_pairs, key=lambda x: x["gap"], reverse=True)
    gap_pairs = gap_pairs[:max_pairs]

    df = df.copy()
    df["price_gap_zone"] = False

    for pair in gap_pairs:
        df.loc[pair["i_index"], "price_gap_zone"] = True
        df.loc[pair["j_index"], "price_gap_zone"] = True

    return df, gap_pairs


def house_category_name(group):
    mapping = {
        0: "Affordable Suburbs",
        1: "Family Housing",
        2: "Premium Housing",
        3: "Inner-city High Price",
        4: "Large Land Value Homes",
        5: "Compact Budget Homes",
    }

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

        var color = getColor(avg);
        var avgText = "$" + Math.round(avg / 1000) + "k";

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
            counts[g] = (counts[g] || 0) + 1;
        });

        var majorityGroup = 0;
        var maxCount = 0;

        Object.keys(counts).forEach(function(g) {
            if (counts[g] > maxCount) {
                maxCount = counts[g];
                majorityGroup = g;
            }
        });

        var colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#000000"];
        var color = colors[Number(majorityGroup) % colors.length];

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
                priceSum += Number(marker.options.price);
                priceCount += 1;

                var g = marker.options.houseGroup;
                groupCounts[g] = (groupCounts[g] || 0) + 1;
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


class DynamicInfoPane(MacroElement):
    def __init__(self, map_name, price_layer_name, house_layer_name, gap_layer_name):
        super().__init__()
        self._name = "DynamicInfoPane"
        self.map_name = map_name
        self.price_layer_name = price_layer_name
        self.house_layer_name = house_layer_name
        self.gap_layer_name = gap_layer_name

        self._template = Template("""
        {% macro html(this, kwargs) %}

        <style>
            #info-pane {
                position: fixed;
                top: 80px;
                left: 20px;
                width: 330px;
                max-height: 78vh;
                overflow-y: auto;
                z-index: 9999;
                background: rgba(255, 255, 255, 0.96);
                border: 2px solid #777;
                border-radius: 10px;
                box-shadow: 0 0 12px rgba(0,0,0,0.35);
                padding: 14px;
                font-family: Arial, sans-serif;
                font-size: 13px;
                line-height: 1.35;
            }

            #info-pane h3 {
                margin: 0 0 8px 0;
                font-size: 17px;
            }

            #info-pane h4 {
                margin: 14px 0 6px 0;
                font-size: 14px;
            }

            #info-pane .section {
                display: none;
                border-top: 1px solid #ddd;
                padding-top: 10px;
                margin-top: 10px;
            }

            #info-pane .legend-row {
                display: flex;
                align-items: center;
                gap: 8px;
                margin: 5px 0;
            }

            #info-pane .swatch {
                width: 18px;
                height: 14px;
                border: 1px solid #777;
                flex: 0 0 auto;
            }

            #info-pane .line-swatch {
                width: 26px;
                height: 4px;
                background: red;
                flex: 0 0 auto;
            }

            #info-pane .small-note {
                color: #555;
                font-size: 12px;
                margin-top: 6px;
            }
        </style>

        <div id="info-pane">
            <h3>Map Guide</h3>
            <div class="small-note">
                Toggle layers on the top-right control. This panel updates based on the selected layer.
            </div>

            <div id="price-info" class="section">
                <h4>Price View</h4>
                <p>
                    This view colours each property and clustered area by property price.
                    Darker red means more expensive homes. Green means lower-priced homes.
                </p>

                <div class="legend-row">
                    <span class="swatch" style="background:#1a9850;"></span>
                    <span>&lt; $400k — Lower-price properties</span>
                </div>
                <div class="legend-row">
                    <span class="swatch" style="background:#66bd63;"></span>
                    <span>$400k - $500k</span>
                </div>
                <div class="legend-row">
                    <span class="swatch" style="background:#a6d96a;"></span>
                    <span>$500k - $600k</span>
                </div>
                <div class="legend-row">
                    <span class="swatch" style="background:#fee08b;"></span>
                    <span>$600k - $700k</span>
                </div>
                <div class="legend-row">
                    <span class="swatch" style="background:#fdae61;"></span>
                    <span>$700k - $800k</span>
                </div>
                <div class="legend-row">
                    <span class="swatch" style="background:#f46d43;"></span>
                    <span>$800k - $900k</span>
                </div>
                <div class="legend-row">
                    <span class="swatch" style="background:#d73027;"></span>
                    <span>$900k - $1M</span>
                </div>
                <div class="legend-row">
                    <span class="swatch" style="background:#a50026;"></span>
                    <span>$1M+ — High-price properties</span>
                </div>

                <p class="small-note">
                    Circle size still represents land area. Cluster bubbles show the average price of homes inside that cluster.
                </p>
            </div>

            <div id="house-info" class="section">
                <h4>House Classification View</h4>
                <p>
                    This view groups homes using KMeans based on location, price, bedrooms, bathrooms,
                    garage, land area, and floor area. The colours represent interpreted house types.
                </p>

                <div class="legend-row">
                    <span class="swatch" style="background:#1f77b4;"></span>
                    <span><b>Affordable Suburbs</b> — lower-price suburban homes</span>
                </div>
                <div class="legend-row">
                    <span class="swatch" style="background:#ff7f0e;"></span>
                    <span><b>Family Housing</b> — typical family-oriented homes</span>
                </div>
                <div class="legend-row">
                    <span class="swatch" style="background:#2ca02c;"></span>
                    <span><b>Premium Housing</b> — higher-price premium properties</span>
                </div>
                <div class="legend-row">
                    <span class="swatch" style="background:#d62728;"></span>
                    <span><b>Inner-city High Price</b> — compact or central high-price homes</span>
                </div>
                <div class="legend-row">
                    <span class="swatch" style="background:#9467bd;"></span>
                    <span><b>Large Land Value Homes</b> — homes where land value is relatively important</span>
                </div>
                <div class="legend-row">
                    <span class="swatch" style="background:#000000;"></span>
                    <span><b>Compact Budget Homes</b> — smaller and more budget-friendly homes</span>
                </div>

                <p class="small-note">
                    The group names are interpretation labels. The model only produces numeric clusters;
                    the names are assigned to make the map easier to understand.
                </p>
            </div>

            <div id="gap-info" class="section">
                <h4>Local Price Gap Zones</h4>
                <p>
                    This layer highlights nearby homes with a large price difference.
                    It is useful for finding possible local price boundaries or transition zones.
                </p>

                <div class="legend-row">
                    <span class="line-swatch"></span>
                    <span>Red line = nearby pair with large price gap</span>
                </div>

                <div class="legend-row">
                    <span class="swatch" style="background:red; opacity:0.4; border-radius:50%;"></span>
                    <span>Red circle = approximate midpoint of the price gap zone</span>
                </div>

                <p class="small-note">
                    Current rule: homes within 500m and price difference of at least $250,000.
                    This is spatial anomaly detection, not supervised classification.
                </p>
            </div>
        </div>

        {% endmacro %}

        {% macro script(this, kwargs) %}

        function updateInfoPane() {
            var priceVisible = {{ this.map_name }}.hasLayer({{ this.price_layer_name }});
            var houseVisible = {{ this.map_name }}.hasLayer({{ this.house_layer_name }});
            var gapVisible = {{ this.map_name }}.hasLayer({{ this.gap_layer_name }});

            document.getElementById("price-info").style.display = priceVisible ? "block" : "none";
            document.getElementById("house-info").style.display = houseVisible ? "block" : "none";
            document.getElementById("gap-info").style.display = gapVisible ? "block" : "none";
        }

        {{ this.map_name }}.on("overlayadd overlayremove", function() {
            updateInfoPane();
        });

        setTimeout(updateInfoPane, 500);

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


def create_map(df, gap_pairs):
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

    gap_layer = folium.FeatureGroup(
        name="Local Price Gap Zones",
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

    for pair in gap_pairs:
        line_popup = f"""
        <b>Local Price Gap Zone</b><br>
        Price gap: ${pair["gap"]:,.0f}<br><br>
        <b>Home A</b><br>
        {pair["address1"]}<br>
        ${pair["price1"]:,.0f}<br><br>
        <b>Home B</b><br>
        {pair["address2"]}<br>
        ${pair["price2"]:,.0f}
        """

        folium.PolyLine(
            locations=[
                [pair["lat1"], pair["lon1"]],
                [pair["lat2"], pair["lon2"]],
            ],
            color="red",
            weight=4,
            opacity=0.75,
            popup=folium.Popup(line_popup, max_width=350),
        ).add_to(gap_layer)

        mid_lat = (pair["lat1"] + pair["lat2"]) / 2
        mid_lon = (pair["lon1"] + pair["lon2"]) / 2

        folium.CircleMarker(
            location=[mid_lat, mid_lon],
            radius=9,
            color="red",
            fill=True,
            fill_color="red",
            fill_opacity=0.35,
            weight=2,
            popup=folium.Popup(line_popup, max_width=350),
        ).add_to(gap_layer)

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

    info_pane = DynamicInfoPane(
        map_name=m.get_name(),
        price_layer_name=price_layer.get_name(),
        house_layer_name=house_layer.get_name(),
        gap_layer_name=gap_layer.get_name(),
    )

    m.get_root().add_child(price_area_layer)
    m.get_root().add_child(house_area_layer)
    m.get_root().add_child(info_pane)

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

    df, gap_pairs = add_local_price_gap_zones(
        df,
        radius_m=500,
        min_price_gap=250000,
        max_pairs=250,
    )

    print("House classification counts:")
    print(df["house_group"].value_counts().sort_index())

    print("Local price gap pairs:", len(gap_pairs))

    m = create_map(df, gap_pairs)
    m.save(OUTPUT_HTML)

    print(f"Generated {OUTPUT_HTML}")

    upload_to_s3()

    print(f"Uploaded to s3://{DEPLOY_BUCKET}/{OUTPUT_HTML}")


if __name__ == "__main__":
    main()