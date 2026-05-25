from bridges.synthetic_generator import generate_sensor_data as bridge_sensor
from buildings.synthetic_generator import generate_sensor_data as building_sensor
from dams.synthetic_generator import generate_sensor_data as dam_sensor
from tunnels.synthetic_generator import generate_sensor_data as tunnel_sensor
from highways.synthetic_generator import generate_sensor_data as highway_sensor

from shared.pipeline import (
    load_models,
    predict_bridge_risk,
    predict_building_risk,
    predict_dam_risk,
    predict_tunnel_risk,
    predict_highway_risk
)


# ── Model Paths ────────────────────────────────────────────────────────────────

MODELS = {
    "bridges":   ("models/bridges/isolation_forest.pkl",   "models/bridges/risk_model.pkl"),
    "buildings": ("models/buildings/isolation_forest.pkl", "models/buildings/risk_model.pkl"),
    "dams":      ("models/dams/isolation_forest.pkl",      "models/dams/risk_model.pkl"),
    "tunnels":   ("models/tunnels/isolation_forest.pkl",   "models/tunnels/risk_model.pkl"),
    "highways":  ("models/highways/isolation_forest.pkl",  "models/highways/risk_model.pkl"),
}


# ── Structure Configs (simulates real-world assets) ───────────────────────────

STRUCTURES = {
    "worli_sealink": {
        "type": "bridges",
        "params": {"bridge_id": "worli_sealink", "age_of_bridge": 23, "last_maintenance_days": 180}
    },
    "bandra_worli": {
        "type": "bridges",
        "params": {"bridge_id": "bandra_worli", "age_of_bridge": 15, "last_maintenance_days": 90}
    },
    "csmt_building": {
        "type": "buildings",
        "params": {"building_id": "csmt_building", "building_age_years": 130, "floor_count": 4, "occupancy": 300}
    },
    "koyna_dam": {
        "type": "dams",
        "params": {"dam_id": "koyna_dam", "dam_type": "concrete", "dam_height_m": 103, "dam_age_years": 62}
    },
    "mumbai_metro_tunnel": {
        "type": "tunnels",
        "params": {"tunnel_id": "mumbai_metro_tunnel", "tunnel_type": "metro", "tunnel_length_m": 3200, "tunnel_age_years": 8, "traffic_volume": 1200}
    },
    "mumbai_pune_expressway": {
        "type": "highways",
        "params": {"highway_id": "mumbai_pune_expressway", "highway_type": "expressway", "highway_length_km": 94, "highway_age_years": 24, "traffic_volume": 1800, "heavy_vehicle_count": 300}
    },
}


# ── Sensor generators ─────────────────────────────────────────────────────────

def get_sensor_data(structure_type, params):
    if structure_type == "bridges":
        return bridge_sensor(**params)
    elif structure_type == "buildings":
        return building_sensor(**params)
    elif structure_type == "dams":
        return dam_sensor(**params)
    elif structure_type == "tunnels":
        return tunnel_sensor(**params)
    elif structure_type == "highways":
        return highway_sensor(**params)


# ── Prediction router ─────────────────────────────────────────────────────────

def run_prediction(structure_type, sensor_df, anomaly_model, risk_model):
    if structure_type == "bridges":
        return predict_bridge_risk(sensor_df, anomaly_model, risk_model)
    elif structure_type == "buildings":
        return predict_building_risk(sensor_df, anomaly_model, risk_model)
    elif structure_type == "dams":
        return predict_dam_risk(sensor_df, anomaly_model, risk_model)
    elif structure_type == "tunnels":
        return predict_tunnel_risk(sensor_df, anomaly_model, risk_model)
    elif structure_type == "highways":
        return predict_highway_risk(sensor_df, anomaly_model, risk_model)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():

    print("=" * 60)
    print("   INFRA MIND VISION - ML Health Analysis")
    print("=" * 60)

    loaded_models = {}
    for structure_type, (anomaly_path, risk_path) in MODELS.items():
        loaded_models[structure_type] = load_models(anomaly_path, risk_path)
        print(f"✅ Loaded models: {structure_type}")

    print("\n" + "=" * 60)
    print("   Running Health Analysis for All Structures")
    print("=" * 60)

    for structure_name, config in STRUCTURES.items():

        structure_type = config["type"]
        params = config["params"]

        sensor_df = get_sensor_data(structure_type, params)

        anomaly_model, risk_model = loaded_models[structure_type]

        anomaly_flag, risk_score, risk_level = run_prediction(
            structure_type, sensor_df, anomaly_model, risk_model
        )

        status = "⚠️  ANOMALY DETECTED" if anomaly_flag else "✅ NORMAL"

        print(f"\n🏗️  {structure_name.upper().replace('_', ' ')} ({structure_type})")
        print(f"   Status     : {status}")
        print(f"   Risk Score : {risk_score:.2f}")
        print(f"   Risk Level : {risk_level}")


if __name__ == "__main__":
    main()