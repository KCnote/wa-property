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
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, r2_score

import numpy as np
import pandas as pd
import json

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
MAX_MAP_POINTS = 1800
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


def add_pca_features(df):
    """Add PCA scores for visual similarity analysis.

    PCA is used here for exploration, not price prediction. It compresses
    the main property attributes into two scores so similar homes can be
    coloured in a continuous way on the map.
    """
    features = [
        "price", "bedrooms", "bathrooms", "garage", "land_area", "floor_area", "latitude", "longitude",
    ]

    model_df = df.dropna(subset=features).copy()

    df = df.copy()
    df["pca1"] = np.nan
    df["pca2"] = np.nan
    df["pca_score"] = np.nan

    if len(model_df) < 10:
        return df, {"explained_variance": []}

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(model_df[features])

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    model_df["pca1"] = X_pca[:, 0]
    model_df["pca2"] = X_pca[:, 1]
    model_df["pca_score"] = model_df["pca1"] + model_df["pca2"]

    df.loc[model_df.index, ["pca1", "pca2", "pca_score"]] = model_df[["pca1", "pca2", "pca_score"]]

    return df, {
        "explained_variance": [float(v) for v in pca.explained_variance_ratio_],
        "features": features,
    }


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


def pca_color(score):
    if pd.isna(score):
        return "#999999"
    score = float(score)
    if score < -2.0:
        return "#2c7bb6"
    if score < -0.8:
        return "#00a6ca"
    if score < 0.8:
        return "#ffffbf"
    if score < 2.0:
        return "#fdae61"
    return "#d7191c"


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




def format_price_label(price):
    """Convert numeric price into a short map label."""
    price = float(price)
    if price >= 1_000_000:
        return f"${price / 1_000_000:.1f}M"
    return f"${price / 1000:.0f}k"


def price_label_marker(row):
    """Small price label above each property marker.

    Labels are added to separate hidden label layers and are only shown
    when the map is zoomed in enough that clustering is disabled.
    """
    price_text = format_price_label(row["price"])

    return folium.Marker(
        location=[row["latitude"], row["longitude"]],
        icon=folium.DivIcon(
            html=f"""
            <div style="
                font-size: 11px;
                font-weight: bold;
                color: #111;
                background: rgba(255,255,255,0.88);
                border: 1px solid #555;
                border-radius: 4px;
                padding: 1px 4px;
                white-space: nowrap;
                transform: translate(-50%, -28px);
                box-shadow: 0 1px 3px rgba(0,0,0,0.35);
                pointer-events: none;
            ">
                {price_text}
            </div>
            """
        ),
    )


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
                '<div style="font-size:10px;">' + count + '</div>' +
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
                '<div style="font-size:10px;">' + total + '</div>' +
                '</div>',
            className: "house-cluster-icon",
            iconSize: [58, 58]
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
                        weight: 3,
                        fillColor: color,
                        fillOpacity: 0.26,
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
    def __init__(self, map_name, price_layer_name, house_layer_name, pca_layer_name, deal_layer_name, gap_layer_name, metrics, pca_metrics, total_rows, map_rows):
        super().__init__()
        self._name = "LightInfoPane"
        self.map_name = map_name
        self.price_layer_name = price_layer_name
        self.house_layer_name = house_layer_name
        self.pca_layer_name = pca_layer_name
        self.deal_layer_name = deal_layer_name
        self.gap_layer_name = gap_layer_name
        self.metrics = metrics
        self.pca_metrics = pca_metrics
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
                f"<li><b>{name}</b>: {importance:.3f}</li>"
                for name, importance in metrics.get("features", [])
            )

        pca_ev = pca_metrics.get("explained_variance", []) if pca_metrics else []
        if len(pca_ev) >= 2:
            pca_text = f"PC1 explains {pca_ev[0] * 100:.1f}% and PC2 explains {pca_ev[1] * 100:.1f}% of the scaled feature variance."
        else:
            pca_text = "PCA metrics are not available."

        self._template = Template(f"""
        {{% macro html(this, kwargs) %}}
        <style>
            #info-pane {{
                position: fixed;
                top: 80px;
                left: 20px;
                width: 355px;
                max-height: 78vh;
                overflow-y: auto;
                z-index: 9999;
                background: rgba(255, 255, 255, 0.96);
                border: 1px solid #666;
                border-radius: 10px;
                box-shadow: 0 0 10px rgba(0,0,0,0.32);
                padding: 13px;
                font-family: Arial, sans-serif;
                font-size: 13px;
                line-height: 1.35;
            }}
            #info-pane h3 {{ margin: 0 0 8px 0; font-size: 17px; }}
            #info-pane h4 {{ margin: 13px 0 6px 0; font-size: 14px; }}
            #info-pane p {{ margin: 6px 0; }}
            #info-pane .section {{ display: none; border-top: 1px solid #ddd; padding-top: 8px; margin-top: 9px; }}
            #info-pane .legend-row {{ display:flex; align-items:center; gap:8px; margin:5px 0; }}
            #info-pane .swatch {{ width:18px; height:14px; border:1px solid #777; flex:0 0 auto; }}
            #info-pane .circle-swatch {{ width:14px; height:14px; border-radius:50%; border:1px solid #777; flex:0 0 auto; }}
            #info-pane .line-swatch {{ width:28px; height:4px; background:red; flex:0 0 auto; }}
            #info-pane .small-note {{ color:#555; font-size:12px; }}
            #info-pane .metric-box {{ background:#f7f7f7; border:1px solid #ddd; border-radius:7px; padding:7px; margin:6px 0; }}
            #info-pane ol {{ padding-left: 20px; margin: 6px 0; }}
        </style>

        <div id="info-pane">
            <h3>WA Property Map Guide</h3>
            <div class="small-note">
                Loaded rows: {self.total_rows:,}<br>
                Displayed map points: {self.map_rows:,}<br>
                The map uses sampled points to keep the static HTML light, but keeps the original colour interpretation.
            </div>

            <div id="price-info" class="section">
                <h4>Price View</h4>
                <p>
                    This layer shows actual sold prices. Each property point and cluster area is coloured by price band.
                    Cluster bubbles show the average price of properties inside that cluster.
                </p>
                <div class="legend-row"><span class="swatch" style="background:#1a9850"></span><span><b>&lt; $400k</b> — lowest price band / affordable stock</span></div>
                <div class="legend-row"><span class="swatch" style="background:#66bd63"></span><span><b>$400k - $500k</b> — lower-mid price band</span></div>
                <div class="legend-row"><span class="swatch" style="background:#a6d96a"></span><span><b>$500k - $600k</b> — mid price band</span></div>
                <div class="legend-row"><span class="swatch" style="background:#fee08b"></span><span><b>$600k - $700k</b> — typical family-home range</span></div>
                <div class="legend-row"><span class="swatch" style="background:#fdae61"></span><span><b>$700k - $800k</b> — upper-mid price band</span></div>
                <div class="legend-row"><span class="swatch" style="background:#f46d43"></span><span><b>$800k - $900k</b> — expensive properties</span></div>
                <div class="legend-row"><span class="swatch" style="background:#d73027"></span><span><b>$900k - $1M</b> — high-price properties</span></div>
                <div class="legend-row"><span class="swatch" style="background:#a50026"></span><span><b>$1M+</b> — premium/high-end properties</span></div>
                <p class="small-note">
                    Circle size roughly follows land area. Larger circles generally indicate larger land parcels.
                </p>
            </div>

            <div id="house-info" class="section">
                <h4>House Classification View</h4>
                <p>
                    This layer uses MiniBatchKMeans to group similar homes. The model considers location, price,
                    bedrooms, bathrooms, garage, land area, and floor area. The group names below are human-friendly
                    interpretation labels, not labels learned directly by the model.
                </p>
                <div class="legend-row"><span class="swatch" style="background:#1f77b4"></span><span><b>Type 0 — Affordable Suburbs</b><br><span class="small-note">Lower-price suburban homes or budget-friendly areas.</span></span></div>
                <div class="legend-row"><span class="swatch" style="background:#ff7f0e"></span><span><b>Type 1 — Family Housing</b><br><span class="small-note">Typical family-oriented homes with balanced land and building size.</span></span></div>
                <div class="legend-row"><span class="swatch" style="background:#2ca02c"></span><span><b>Type 2 — Premium Housing</b><br><span class="small-note">Higher-price homes with stronger location or property attributes.</span></span></div>
                <div class="legend-row"><span class="swatch" style="background:#d62728"></span><span><b>Type 3 — Inner-city High Price</b><br><span class="small-note">Compact or central homes where location can dominate price.</span></span></div>
                <div class="legend-row"><span class="swatch" style="background:#9467bd"></span><span><b>Type 4 — Large Land Value Homes</b><br><span class="small-note">Homes where land size/value appears relatively important.</span></span></div>
                <div class="legend-row"><span class="swatch" style="background:#000000"></span><span><b>Type 5 — Compact Budget Homes</b><br><span class="small-note">Smaller and more budget-friendly homes.</span></span></div>
                <p class="small-note">
                    Because KMeans is unsupervised, these groups are best used as exploration hints, not final property categories.
                </p>
            </div>

            <div id="pca-info" class="section">
                <h4>PCA Similarity View</h4>
                <p>
                    PCA compresses price, size, room count, and location features into two main components.
                    It does not predict price; it shows overall property similarity.
                </p>
                <div class="metric-box">
                    <b>PCA summary</b><br>
                    {pca_text}
                </div>
                <div class="legend-row"><span class="swatch" style="background:#2c7bb6"></span><span><b>Deep blue</b> — lower PCA score band</span></div>
                <div class="legend-row"><span class="swatch" style="background:#00a6ca"></span><span><b>Blue</b> — lower-mid similarity band</span></div>
                <div class="legend-row"><span class="swatch" style="background:#ffffbf"></span><span><b>Yellow</b> — middle similarity band</span></div>
                <div class="legend-row"><span class="swatch" style="background:#fdae61"></span><span><b>Orange</b> — upper-mid similarity band</span></div>
                <div class="legend-row"><span class="swatch" style="background:#d7191c"></span><span><b>Red</b> — higher PCA score band</span></div>
                <p class="small-note">
                    Use this as an exploratory layer. Nearby homes with similar colours have similar overall feature profiles.
                    Sudden colour changes can reveal different house types or unusual local properties.
                </p>
            </div>

            <div id="deal-info" class="section">
                <h4>Random Forest Valuation</h4>
                <p>
                    Random Forest predicts a reasonable price from property attributes and location.
                    This layer highlights only the strongest undervalued candidates to avoid making the HTML too large.
                </p>
                <div class="metric-box">
                    <b>Model metrics</b><br>
                    MAE: {mae_text}<br>
                    R²: {r2_text}
                </div>
                <h4>Top Features</h4>
                <ol>{feature_html}</ol>
                <div class="legend-row"><span class="circle-swatch" style="background:#1a9850"></span><span><b>Green</b> — actual price is more than 10% below predicted value</span></div>
                <div class="legend-row"><span class="circle-swatch" style="background:#2b83ba"></span><span><b>Blue</b> — actual price is close to predicted value</span></div>
                <div class="legend-row"><span class="circle-swatch" style="background:#d73027"></span><span><b>Red</b> — actual price is more than 10% above predicted value</span></div>
                <p class="small-note">
                    Positive gap = predicted price is higher than actual price. This can suggest possible undervaluation,
                    but it still needs manual checking against condition, renovation, zoning, and listing details.
                </p>
            </div>

            <div id="gap-info" class="section">
                <h4>Local Price Gap Zones</h4>
                <p>
                    This layer highlights nearby homes with large price differences. It is useful for spotting local price boundaries,
                    transition zones, or areas where suburb/street-level factors may strongly affect value.
                </p>
                <div class="legend-row"><span class="line-swatch"></span><span><b>Red line</b> — nearby pair with a large price gap</span></div>
                <p class="small-note">
                    Current rule: within 500m and at least $250k difference. Only top gap pairs are shown to keep the map light.
                </p>
            </div>
        </div>
        {{% endmacro %}}

        {{% macro script(this, kwargs) %}}
        function updateInfoPane() {{
            document.getElementById("price-info").style.display = {self.map_name}.hasLayer({self.price_layer_name}) ? "block" : "none";
            document.getElementById("house-info").style.display = {self.map_name}.hasLayer({self.house_layer_name}) ? "block" : "none";
            document.getElementById("pca-info").style.display = {self.map_name}.hasLayer({self.pca_layer_name}) ? "block" : "none";
            document.getElementById("deal-info").style.display = {self.map_name}.hasLayer({self.deal_layer_name}) ? "block" : "none";
            document.getElementById("gap-info").style.display = {self.map_name}.hasLayer({self.gap_layer_name}) ? "block" : "none";
        }}
        {self.map_name}.on("overlayadd overlayremove", updateInfoPane);
        setTimeout(updateInfoPane, 500);
        {{% endmacro %}}
        """)


class ViewportPriceLabelController(MacroElement):
    """Render price labels only for homes inside the current viewport.

    Important: this is real viewport-based rendering for labels.
    The browser does not create thousands of label DOM nodes at page load.
    Labels are created only when:
      1. zoom >= zoom_threshold
      2. the related view layer is enabled
      3. the property point is inside the current map bounds
    """
    def __init__(self, map_name, zoom_threshold, label_data, watched_layers):
        super().__init__()
        self._name = "ViewportPriceLabelController"
        self.map_name = map_name
        self.zoom_threshold = zoom_threshold
        self.label_data_json = json.dumps(label_data, separators=(",", ":"))
        self.watched_layers = watched_layers

        active_layer_conditions = " || ".join(
            f"{self.map_name}.hasLayer({layer_name})" for layer_name in watched_layers
        )

        self._template = Template(f"""
        {{% macro html(this, kwargs) %}}
        <style>
            .viewport-price-label-clean {{
                background: transparent !important;
                border: none !important;
                box-shadow: none !important;
            }}
            .price-badge-blue {{
                position: relative;
                display: inline-flex;
                align-items: center;
                gap: 4px;
                padding: 3px 7px;
                border-radius: 999px;
                background: linear-gradient(135deg, #0f5bd8 0%, #1e9bff 52%, #72d7ff 100%);
                color: white;
                font-size: 11px;
                font-weight: 800;
                line-height: 1;
                letter-spacing: 0.1px;
                white-space: nowrap;
                border: 1px solid rgba(255,255,255,0.92);
                box-shadow: 0 3px 8px rgba(0,42,120,0.36), 0 0 0 1px rgba(15,91,216,0.28);
                text-shadow: 0 1px 1px rgba(0,0,0,0.32);
                pointer-events: none;
            }}
            .price-badge-blue::after {{
                content: "";
                position: absolute;
                left: 50%;
                bottom: -6px;
                transform: translateX(-50%);
                width: 0;
                height: 0;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 6px solid #1e9bff;
                filter: drop-shadow(0 2px 1px rgba(0,0,0,0.18));
            }}
            .price-badge-dot {{
                width: 6px;
                height: 6px;
                border-radius: 50%;
                background: rgba(255,255,255,0.92);
                box-shadow: 0 0 0 2px rgba(255,255,255,0.24);
                flex: 0 0 auto;
            }}
        </style>
        {{% endmacro %}}

        {{% macro script(this, kwargs) %}}
        var viewportPriceLabels = L.layerGroup();
        var viewportPriceLabelData = {self.label_data_json};
        var viewportPriceLabelTimer = null;
        var viewportPriceLabelMaxVisible = 450;

        function makePriceLabelIcon(text) {{
            return L.divIcon({{
                className: "viewport-price-label-clean",
                html:
                    '<div class="price-badge-blue">' +
                    '<span class="price-badge-dot"></span>' +
                    '<span>' + text + '</span>' +
                    '</div>',
                iconSize: [76, 26],
                iconAnchor: [38, 30]
            }});
        }}

        function redrawViewportPriceLabels() {{
            viewportPriceLabels.clearLayers();

            var activeViewLayer = {active_layer_conditions};
            var shouldShow = {self.map_name}.getZoom() >= {self.zoom_threshold} && activeViewLayer;

            if (!shouldShow) {{
                if ({self.map_name}.hasLayer(viewportPriceLabels)) {{
                    {self.map_name}.removeLayer(viewportPriceLabels);
                }}
                return;
            }}

            if (!{self.map_name}.hasLayer(viewportPriceLabels)) {{
                viewportPriceLabels.addTo({self.map_name});
            }}

            var bounds = {self.map_name}.getBounds().pad(0.08);
            var added = 0;

            for (var i = 0; i < viewportPriceLabelData.length; i++) {{
                var item = viewportPriceLabelData[i];
                var latlng = L.latLng(item[0], item[1]);
                if (bounds.contains(latlng)) {{
                    L.marker(latlng, {{ icon: makePriceLabelIcon(item[2]), interactive: false }})
                        .addTo(viewportPriceLabels);
                    added += 1;
                    if (added >= viewportPriceLabelMaxVisible) break;
                }}
            }}
        }}

        function scheduleViewportPriceLabels() {{
            if (viewportPriceLabelTimer) clearTimeout(viewportPriceLabelTimer);
            viewportPriceLabelTimer = setTimeout(redrawViewportPriceLabels, 120);
        }}

        {self.map_name}.on("zoomend moveend overlayadd overlayremove", scheduleViewportPriceLabels);
        setTimeout(redrawViewportPriceLabels, 700);
        {{% endmacro %}}
        """)


def create_map(df_map, gap_pairs, metrics, pca_metrics, total_rows):
    if df_map.empty:
        raise RuntimeError("No data found from Athena.")

    center_lat = df_map["latitude"].mean()
    center_lon = df_map["longitude"].mean()

    m = folium.Map(location=[center_lat, center_lon], zoom_start=10, tiles="OpenStreetMap")

    price_layer = folium.FeatureGroup(name="Price View", show=True).add_to(m)
    house_layer = folium.FeatureGroup(name="House Classification View", show=False).add_to(m)
    pca_layer = folium.FeatureGroup(name="PCA Similarity View", show=False).add_to(m)
    deal_layer = folium.FeatureGroup(name="RF Undervalued Candidates", show=False).add_to(m)
    gap_layer = folium.FeatureGroup(name="Local Price Gap Zones", show=False).add_to(m)

    LABEL_ZOOM_THRESHOLD = 14

    cluster_options = {
        "showCoverageOnHover": False,
        "removeOutsideVisibleBounds": True,
        "spiderfyOnMaxZoom": False,
        "disableClusteringAtZoom": LABEL_ZOOM_THRESHOLD,
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
    pca_cluster = MarkerCluster(
        name="PCA Similarity Cluster",
        options=cluster_options,
    ).add_to(pca_layer)

    for _, row in df_map.iterrows():
        price = float(row["price"])
        radius = land_radius(row["land_area"])
        html = popup_html(row)

        price_marker = folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=radius + 1,
            color="#222222",
            fill=True,
            fill_color=price_color(price),
            fill_opacity=0.95,
            weight=1,
            popup=folium.Popup(html, max_width=260),
        )
        price_marker.options["price"] = price
        price_marker.options["houseGroup"] = int(row["house_group"])
        price_marker.add_to(price_cluster)

        house_marker = folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=radius + 1,
            color="#222222",
            fill=True,
            fill_color=house_type_color(row["house_group"]),
            fill_opacity=0.95,
            weight=1,
            popup=folium.Popup(html, max_width=260),
        )
        house_marker.options["price"] = price
        house_marker.options["houseGroup"] = int(row["house_group"])
        house_marker.add_to(house_cluster)

        pca_marker = folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=radius + 1,
            color="#222222",
            fill=True,
            fill_color=pca_color(row.get("pca_score", np.nan)),
            fill_opacity=0.95,
            weight=1,
            popup=folium.Popup(html, max_width=260),
        )
        pca_marker.add_to(pca_cluster)

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
        pca_layer_name=pca_layer.get_name(),
        deal_layer_name=deal_layer.get_name(),
        gap_layer_name=gap_layer.get_name(),
        metrics=metrics,
        pca_metrics=pca_metrics,
        total_rows=total_rows,
        map_rows=len(df_map),
    )
    m.get_root().add_child(info_pane)

    label_data = [
        [float(row["latitude"]), float(row["longitude"]), format_price_label(row["price"])]
        for _, row in df_map.dropna(subset=["latitude", "longitude", "price"]).iterrows()
    ]

    m.get_root().add_child(ViewportPriceLabelController(
        map_name=m.get_name(),
        zoom_threshold=LABEL_ZOOM_THRESHOLD,
        label_data=label_data,
        watched_layers=[
            price_layer.get_name(),
            house_layer.get_name(),
            pca_layer.get_name(),
        ],
    ))

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
    df, pca_metrics = add_pca_features(df)
    print("PCA metrics:", pca_metrics)

    df_map = select_map_points(df)
    print("Displayed map rows:", len(df_map))

    df_map, gap_pairs = add_local_price_gap_zones(
        df_map,
        radius_m=500,
        min_price_gap=250000,
        max_pairs=MAX_GAP_PAIRS,
    )
    print("Displayed local price gap pairs:", len(gap_pairs))

    m = create_map(df_map, gap_pairs, metrics, pca_metrics, total_rows)
    m.save(OUTPUT_HTML)

    print(f"Generated {OUTPUT_HTML}")
    upload_to_s3()
    print(f"Uploaded to s3://{DEPLOY_BUCKET}/{OUTPUT_HTML}")


if __name__ == "__main__":
    main()
