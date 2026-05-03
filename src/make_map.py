import boto3
import awswrangler as wr
import folium
from folium.plugins import MarkerCluster
from branca.element import MacroElement
from jinja2 import Template

from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import BallTree
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

import numpy as np
import pandas as pd

DATABASE = "wa_property_db"
TABLE = "wa_property_latest"

ATHENA_OUTPUT = "s3://personal-wa-property-storage-337164669284-ap-southeast-2-an/athena-results/"
DEPLOY_BUCKET = "personal-wa-property-server-337164669284-ap-southeast-2-an"
OUTPUT_HTML = "index.html"

# =========================================================
# Size control settings
# =========================================================
# Main reason Folium gets heavy: every marker/popup is embedded into one HTML file.
# Keep this number small for S3 static hosting.
MAX_MAP_POINTS = 2500
MAX_UNDERVALUED_MARKERS = 150
MAX_GAP_PAIRS = 100

RANDOM_STATE = 42


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
      cbd_dist,
      nearest_stn_dist,
      nearest_sch_dist,
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


def clean_numeric(df):
    df = df.copy()
    numeric_cols = [
        "price", "bedrooms", "bathrooms", "garage", "land_area", "floor_area",
        "cbd_dist", "nearest_stn_dist", "nearest_sch_dist", "latitude", "longitude",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def train_random_forest(df):
    df = df.copy()

    rf_features = [
        "bedrooms",
        "bathrooms",
        "garage",
        "land_area",
        "floor_area",
        "latitude",
        "longitude",
        "cbd_dist",
        "nearest_stn_dist",
        "nearest_sch_dist",
    ]

    available_features = [c for c in rf_features if c in df.columns]
    model_df = df.dropna(subset=available_features + ["price"]).copy()

    if len(model_df) < 100:
        df["predicted_price"] = np.nan
        df["prediction_gap"] = np.nan
        df["prediction_gap_pct"] = np.nan
        return df, {
            "mae": None,
            "r2": None,
            "features": [],
        }

    X = model_df[available_features]
    y = model_df["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    model = RandomForestRegressor(
        n_estimators=120,
        max_depth=14,
        min_samples_leaf=4,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    all_pred = model.predict(X)
    model_df["predicted_price"] = all_pred
    model_df["prediction_gap"] = model_df["predicted_price"] - model_df["price"]
    model_df["prediction_gap_pct"] = model_df["prediction_gap"] / model_df["price"]

    df["predicted_price"] = np.nan
    df["prediction_gap"] = np.nan
    df["prediction_gap_pct"] = np.nan
    df.loc[model_df.index, ["predicted_price", "prediction_gap", "prediction_gap_pct"]] = model_df[
        ["predicted_price", "prediction_gap", "prediction_gap_pct"]
    ]

    importance = sorted(
        zip(available_features, model.feature_importances_),
        key=lambda x: x[1],
        reverse=True,
    )

    metrics = {
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "r2": float(r2_score(y_test, y_pred)),
        "features": importance[:8],
    }

    return df, metrics


def add_kmeans_cluster(df, n_clusters=6):
    features = [
        "latitude", "longitude", "price", "bedrooms", "bathrooms", "garage", "land_area", "floor_area",
    ]
    model_df = df.dropna(subset=features).copy()

    if len(model_df) < n_clusters:
        df = df.copy()
        df["house_group"] = 0
        return df

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(model_df[features])

    # MiniBatchKMeans is faster and lighter than full KMeans for larger datasets.
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=RANDOM_STATE,
        batch_size=512,
        n_init="auto",
    )
    model_df["house_group"] = kmeans.fit_predict(X_scaled)

    df = df.copy()
    df["house_group"] = 0
    df.loc[model_df.index, "house_group"] = model_df["house_group"].astype(int)
    return df


def select_map_points(df):
    """Keep the HTML small while preserving useful points.

    Priority:
    1. Top undervalued candidates from Random Forest
    2. Random sample from remaining data
    """
    df = df.copy()
    if len(df) <= MAX_MAP_POINTS:
        return df

    top_undervalued = (
        df.dropna(subset=["prediction_gap"])
        .sort_values("prediction_gap", ascending=False)
        .head(MAX_UNDERVALUED_MARKERS)
    )

    remaining = df.drop(index=top_undervalued.index, errors="ignore")
    sample_n = max(0, MAX_MAP_POINTS - len(top_undervalued))
    sampled = remaining.sample(n=min(sample_n, len(remaining)), random_state=RANDOM_STATE)

    return pd.concat([top_undervalued, sampled]).sort_index()


def add_local_price_gap_zones(df, radius_m=500, min_price_gap=250000, max_pairs=MAX_GAP_PAIRS):
    work_df = df.dropna(subset=["latitude", "longitude", "price"]).copy()

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
                gap_pairs.append({
                    "i_index": row_i.name,
                    "j_index": row_j.name,
                    "lat1": row_i["latitude"],
                    "lon1": row_i["longitude"],
                    "lat2": row_j["latitude"],
                    "lon2": row_j["longitude"],
                    "address1": row_i.get("address", ""),
                    "address2": row_j.get("address", ""),
                    "price1": float(row_i["price"]),
                    "price2": float(row_j["price"]),
                    "gap": price_gap,
                })

    gap_pairs = sorted(gap_pairs, key=lambda x: x["gap"], reverse=True)[:max_pairs]

    df = df.copy()
    df["price_gap_zone"] = False
    for pair in gap_pairs:
        df.loc[pair["i_index"], "price_gap_zone"] = True
        df.loc[pair["j_index"], "price_gap_zone"] = True

    return df, gap_pairs


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
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#000000"]
    return colors[int(group) % len(colors)]


def deal_color(row):
    gap_pct = row.get("prediction_gap_pct", np.nan)
    if pd.isna(gap_pct):
        return "#999999"
    if gap_pct > 0.10:
        return "#1a9850"  # actual price is meaningfully below model estimate
    if gap_pct < -0.10:
        return "#d73027"  # actual price is meaningfully above model estimate
    return "#2b83ba"


def land_radius(land_area):
    land_area = float(land_area)
    if land_area < 450:
        return 4
    if land_area < 550:
        return 5
    if land_area < 700:
        return 6
    return 7


def popup_html(row, compact=True):
    price = float(row["price"])
    pred = row.get("predicted_price", np.nan)
    gap = row.get("prediction_gap", np.nan)

    pred_line = ""
    if not pd.isna(pred):
        pred_line = f"Predicted: ${pred:,.0f}<br>Gap: ${gap:,.0f}<br>"

    # Keep popup short. Long popup text is a major HTML-size driver.
    return f"""
    <b>{row.get('address', '')}</b><br>
    {row.get('suburb', '')}<br>
    Price: ${price:,.0f}<br>
    {pred_line}
    Bed/Bath/Garage: {row.get('bedrooms', '')}/{row.get('bathrooms', '')}/{row.get('garage', '')}<br>
    Land: {row.get('land_area', '')} sqm
    """


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
                'width:56px;' +
                'height:56px;' +
                'display:flex;' +
                'flex-direction:column;' +
                'align-items:center;' +
                'justify-content:center;' +
                'font-size:12px;' +
                'font-weight:bold;' +
                'box-shadow:0 0 6px rgba(0,0,0,0.45);' +
                '">' +
                '<div>' + avgText + '</div>' +
                '<div style="font-size:10px;">' + count + '</div>' +
                '</div>',
            className: "price-cluster-icon",
            iconSize: [56, 56]
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
                'width:56px;' +
                'height:56px;' +
                'display:flex;' +
                'flex-direction:column;' +
                'align-items:center;' +
                'justify-content:center;' +
                'font-size:12px;' +
                'font-weight:bold;' +
                'box-shadow:0 0 6px rgba(0,0,0,0.45);' +
                '">' +
                '<div>Type ' + majorityGroup + '</div>' +
                '<div style="font-size:10px;">' + total + '</div>' +
                '</div>',
            className: "house-cluster-icon",
            iconSize: [56, 56]
        });
    }
    """


class ClusterAreaLayer(MacroElement):
    """Lightweight coloured rectangle coverage for visible marker clusters.

    This restores the richer colour feel without adding thousands of extra markers.
    """
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

                    L.rectangle(bounds, {
                        color: color,
                        weight: 2,
                        fillColor: color,
                        fillOpacity: 0.18,
                        interactive: false
                    }).addTo(areaLayer_{{ this.cluster_name }}).bringToBack();
                }
            });
        }

        {{ this.map_name }}.on("zoomend moveend overlayadd overlayremove", redrawArea_{{ this.cluster_name }});
        {{ this.cluster_name }}.on("animationend", redrawArea_{{ this.cluster_name }});
        setTimeout(redrawArea_{{ this.cluster_name }}, 700);

        {% endmacro %}
        """)


class LightInfoPane(MacroElement):
    def __init__(self, map_name, price_layer_name, house_layer_name, deal_layer_name, gap_layer_name, metrics, total_rows, map_rows):
        super().__init__()
        self._name = "LightInfoPane"
        self.map_name = map_name
        self.price_layer_name = price_layer_name
        self.house_layer_name = house_layer_name
        self.deal_layer_name = deal_layer_name
        self.gap_layer_name = gap_layer_name
        self.metrics = metrics
        self.total_rows = total_rows
        self.map_rows = map_rows

        if metrics.get("mae") is None:
            mae_text = "N/A"
            r2_text = "N/A"
            feature_html = "<li>Not enough data</li>"
        else:
            mae_text = f"${metrics['mae']:,.0f}"
            r2_text = f"{metrics['r2']:.3f}"
            feature_html = "".join(
                f"<li>{name}: {importance:.3f}</li>"
                for name, importance in metrics.get("features", [])
            )

        self._template = Template(f"""
        {{% macro html(this, kwargs) %}}
        <style>
            #info-pane {{
                position: fixed;
                top: 80px;
                left: 20px;
                width: 315px;
                max-height: 78vh;
                overflow-y: auto;
                z-index: 9999;
                background: rgba(255, 255, 255, 0.96);
                border: 1px solid #777;
                border-radius: 9px;
                box-shadow: 0 0 8px rgba(0,0,0,0.3);
                padding: 12px;
                font-family: Arial, sans-serif;
                font-size: 13px;
                line-height: 1.35;
            }}
            #info-pane h3 {{ margin: 0 0 8px 0; font-size: 16px; }}
            #info-pane h4 {{ margin: 12px 0 5px 0; font-size: 14px; }}
            #info-pane .section {{ display: none; border-top: 1px solid #ddd; padding-top: 8px; margin-top: 8px; }}
            #info-pane .legend-row {{ display:flex; align-items:center; gap:8px; margin:4px 0; }}
            #info-pane .swatch {{ width:16px; height:13px; border:1px solid #777; flex:0 0 auto; }}
            #info-pane .small-note {{ color:#555; font-size:12px; }}
        </style>

        <div id="info-pane">
            <h3>WA Property Map</h3>
            <div class="small-note">
                Loaded rows: {self.total_rows:,}<br>
                Displayed map points: {self.map_rows:,}<br>
                This lightweight version samples points to keep the HTML small.
            </div>

            <div id="price-info" class="section">
                <h4>Price View</h4>
                <p>Properties are coloured by actual sold price.</p>
                <div class="legend-row"><span class="swatch" style="background:#1a9850"></span>&lt; $400k</div>
                <div class="legend-row"><span class="swatch" style="background:#fee08b"></span>$600k - $700k</div>
                <div class="legend-row"><span class="swatch" style="background:#d73027"></span>$900k - $1M</div>
                <div class="legend-row"><span class="swatch" style="background:#a50026"></span>$1M+</div>
            </div>

            <div id="house-info" class="section">
                <h4>House Classification</h4>
                <p>MiniBatchKMeans groups homes by location, price, bedrooms, bathrooms, garage, land area, and floor area.</p>
            </div>

            <div id="deal-info" class="section">
                <h4>Random Forest Valuation</h4>
                <p>Random Forest predicts price from property attributes and location. The layer highlights top undervalued candidates only.</p>
                <b>Model metrics</b><br>
                MAE: {mae_text}<br>
                R²: {r2_text}<br>
                <h4>Top features</h4>
                <ol>{feature_html}</ol>
                <div class="legend-row"><span class="swatch" style="background:#1a9850"></span>Actual price &gt;10% below prediction</div>
                <div class="legend-row"><span class="swatch" style="background:#2b83ba"></span>Near predicted value</div>
                <div class="legend-row"><span class="swatch" style="background:#d73027"></span>Actual price &gt;10% above prediction</div>
            </div>

            <div id="gap-info" class="section">
                <h4>Local Price Gap Zones</h4>
                <p>Only the largest nearby price gaps are shown to reduce file size.</p>
            </div>
        </div>
        {{% endmacro %}}

        {{% macro script(this, kwargs) %}}
        function updateInfoPane() {{
            document.getElementById("price-info").style.display = {self.map_name}.hasLayer({self.price_layer_name}) ? "block" : "none";
            document.getElementById("house-info").style.display = {self.map_name}.hasLayer({self.house_layer_name}) ? "block" : "none";
            document.getElementById("deal-info").style.display = {self.map_name}.hasLayer({self.deal_layer_name}) ? "block" : "none";
            document.getElementById("gap-info").style.display = {self.map_name}.hasLayer({self.gap_layer_name}) ? "block" : "none";
        }}
        {self.map_name}.on("overlayadd overlayremove", updateInfoPane);
        setTimeout(updateInfoPane, 500);
        {{% endmacro %}}
        """)


def create_map(df_map, gap_pairs, metrics, total_rows):
    if df_map.empty:
        raise RuntimeError("No data found from Athena.")

    center_lat = df_map["latitude"].mean()
    center_lon = df_map["longitude"].mean()

    m = folium.Map(location=[center_lat, center_lon], zoom_start=10, tiles="CartoDB positron")

    price_layer = folium.FeatureGroup(name="Price View", show=True).add_to(m)
    house_layer = folium.FeatureGroup(name="House Classification View", show=False).add_to(m)
    deal_layer = folium.FeatureGroup(name="RF Undervalued Candidates", show=False).add_to(m)
    gap_layer = folium.FeatureGroup(name="Local Price Gap Zones", show=False).add_to(m)

    cluster_options = {
        "showCoverageOnHover": False,
        "removeOutsideVisibleBounds": True,
        "spiderfyOnMaxZoom": False,
        "disableClusteringAtZoom": 14,
        "maxClusterRadius": 110,
    }

    price_cluster = MarkerCluster(
        name="Price Cluster",
        icon_create_function=price_cluster_icon_function(),
        options=cluster_options,
    ).add_to(price_layer)
    house_cluster = MarkerCluster(
        name="House Cluster",
        icon_create_function=house_cluster_icon_function(),
        options=cluster_options,
    ).add_to(house_layer)

    for _, row in df_map.iterrows():
        price = float(row["price"])
        radius = land_radius(row["land_area"])
        html = popup_html(row)

        price_marker = folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=radius,
            color="white",
            fill=True,
            fill_color=price_color(price),
            fill_opacity=0.85,
            weight=1,
            popup=folium.Popup(html, max_width=260),
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
            fill_opacity=0.85,
            weight=1,
            popup=folium.Popup(html, max_width=260),
        )
        house_marker.options["price"] = price
        house_marker.options["houseGroup"] = int(row["house_group"])
        house_marker.add_to(house_cluster)

    undervalued = (
        df_map.dropna(subset=["prediction_gap"])
        .sort_values("prediction_gap", ascending=False)
        .head(MAX_UNDERVALUED_MARKERS)
    )

    for _, row in undervalued.iterrows():
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=7,
            color="white",
            fill=True,
            fill_color=deal_color(row),
            fill_opacity=0.9,
            weight=2,
            popup=folium.Popup(popup_html(row), max_width=260),
        ).add_to(deal_layer)

    for pair in gap_pairs:
        line_popup = f"""
        <b>Local Price Gap</b><br>
        Gap: ${pair['gap']:,.0f}<br>
        A: ${pair['price1']:,.0f}<br>
        B: ${pair['price2']:,.0f}
        """
        folium.PolyLine(
            locations=[[pair["lat1"], pair["lon1"]], [pair["lat2"], pair["lon2"]]],
            color="red",
            weight=3,
            opacity=0.65,
            popup=folium.Popup(line_popup, max_width=260),
        ).add_to(gap_layer)

    # Restore rich-looking colour regions without adding more data points.
    m.get_root().add_child(ClusterAreaLayer(
        map_name=m.get_name(),
        parent_layer_name=price_layer.get_name(),
        cluster_name=price_cluster.get_name(),
        mode="price",
    ))
    m.get_root().add_child(ClusterAreaLayer(
        map_name=m.get_name(),
        parent_layer_name=house_layer.get_name(),
        cluster_name=house_cluster.get_name(),
        mode="house",
    ))

    info_pane = LightInfoPane(
        map_name=m.get_name(),
        price_layer_name=price_layer.get_name(),
        house_layer_name=house_layer.get_name(),
        deal_layer_name=deal_layer.get_name(),
        gap_layer_name=gap_layer.get_name(),
        metrics=metrics,
        total_rows=total_rows,
        map_rows=len(df_map),
    )
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
            "CacheControl": "no-cache, no-store, must-revalidate",
        },
    )


def main():
    df = load_data()
    df = clean_numeric(df)
    total_rows = len(df)

    print("Loaded rows:", total_rows)

    df, metrics = train_random_forest(df)
    print("Random Forest metrics:", metrics)

    df = add_kmeans_cluster(df, n_clusters=6)

    df_map = select_map_points(df)
    print("Displayed map rows:", len(df_map))

    df_map, gap_pairs = add_local_price_gap_zones(
        df_map,
        radius_m=500,
        min_price_gap=250000,
        max_pairs=MAX_GAP_PAIRS,
    )
    print("Displayed local price gap pairs:", len(gap_pairs))

    m = create_map(df_map, gap_pairs, metrics, total_rows)
    m.save(OUTPUT_HTML)

    print(f"Generated {OUTPUT_HTML}")
    upload_to_s3()
    print(f"Uploaded to s3://{DEPLOY_BUCKET}/{OUTPUT_HTML}")


if __name__ == "__main__":
    main()
