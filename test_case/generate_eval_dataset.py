"""
GENERATE EVAL DATASET
================================================================================
Mục đích: DB `drone_fleet.db` hiện chưa tồn tại (chưa ai chạy pipeline thật),
nên script này SINH RA nó bằng cách chạy đúng luồng M1 -> M2 -> M4 thật (import
thẳng code của 3 member, không viết lại logic) ở quy mô nhỏ (8 drone x 1 giờ dữ
liệu, thay vì ~20.000 drone) để có dữ liệu thật cho việc đánh giá khoa học.

ĐIỂM QUAN TRỌNG — vì sao cần file này riêng, không chỉ chạy m1_streaming.py:
    m1_streaming.py bản thật dùng asyncio + 20.000 coroutine, thiết kế cho demo
    production, không tiện chạy trong 1 script đánh giá ngắn. Ở đây ta gọi
    THẲNG các hàm lõi mà M1 cũng gọi (make_bronze_payload, StochasticSimulator)
    theo đúng cách M1 gọi, chỉ bỏ phần asyncio/batching hạ tầng.

CẬP NHẬT (đồng bộ với schema.py/stochastic_simulator.py/m2_analytics_pipeline.py
bản mới): true_lat/true_lon giờ đi thẳng qua Bronze -> Silver -> fact_telemetry
như 1 cột CHÍNH THỨC (Optional, chỉ dùng để đánh giá — xem giải thích trong
schema.py). Không cần bảng phụ eval_gps_ground_truth nữa (bản trước có, đã bỏ)
vì dữ liệu đã nằm sẵn trong fact_telemetry.true_lat/true_lon.

Cách dùng:
    python3 generate_eval_dataset.py --db drone_fleet.db --hours 1 --window-minutes 5
"""

import argparse
import sqlite3

from schema import DDL_STATEMENTS, DroneInfo, make_bronze_payload
from stochastic_simulator import StochasticSimulator, CASE_STUDY_PROFILES

import m2_analytics_pipeline as m2
import m4_ml_engineer as m4


def build_db(db_path: str, hours: float, window_minutes: int, seed_base: int = 42):
    conn = sqlite3.connect(db_path)
    conn.executescript(DDL_STATEMENTS)
    conn.commit()

    n_steps = int(hours * 3600)
    print(f"[GEN] Sinh {len(CASE_STUDY_PROFILES)} drone x {n_steps} giây "
          f"({hours}h) telemetry ...")

    from datetime import datetime, timedelta, timezone
    base_time = datetime(2026, 7, 1, tzinfo=timezone.utc)

    for i, profile in enumerate(CASE_STUDY_PROFILES):
        drone_id = f"DRONE_{i+1:03d}"
        conn.execute(
            """
            INSERT OR REPLACE INTO dim_drones
                (drone_id, model_name, max_battery_capacity_wh, profile_label,
                 payload_kg, wind_zone, battery_health, ambient_temp_c,
                 gps_quality, hardware_reliability, network_zone)
            VALUES (?, 'DJI Matrice 100', 500.0, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                drone_id, profile.label, profile.payload_kg, profile.wind_zone,
                profile.battery_health, profile.ambient_temp_c, profile.gps_quality,
                profile.hardware_reliability, profile.network_zone,
            ),
        )

        sim = StochasticSimulator(drone_id=drone_id, seed=seed_base + i, profile=profile)

        bronze_rows = []
        for t in range(n_steps):
            out = sim.step()
            ts = (base_time + timedelta(seconds=t)).strftime("%Y-%m-%dT%H:%M:%SZ")

            # out đã có true_lat/true_lon (từ stochastic_simulator.py bản mới) —
            # make_bronze_payload() validate qua TelemetryPayload (schema.py bản
            # mới, có field true_lat/true_lon Optional) nên đi thẳng qua, không
            # cần lấy riêng từ sim.true_lat/sim.true_lon như bản trước nữa.
            payload_json = make_bronze_payload(drone_id, out).model_copy(
                update={"timestamp": ts}
            ).model_dump_json()
            bronze_rows.append((payload_json,))

        conn.executemany(
            "INSERT INTO bronze_telemetry (raw_json_payload) VALUES (?)", bronze_rows
        )
        conn.commit()
        print(f"  [{drone_id}] profile={profile.label:20s} -> {n_steps} dòng bronze (kèm ground truth)")

    conn.close()


def run_m2_m4(db_path: str, window_minutes: int):
    print("[GEN] Chạy M2 (Bronze -> Silver -> Gold: *_mean/*_std/*_status) ...")
    conn = sqlite3.connect(db_path)
    m2.init_db(conn)
    m2.WINDOW_MINUTES = window_minutes
    affected = m2.process_bronze_to_silver(conn)
    m2.aggregate_and_upsert_gold(conn, affected)
    conn.close()

    print("[GEN] Chạy M4 (Kalman + EMA baseline; GBM + Linear Quantile baseline) ...")
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA busy_timeout = 5000;")
    df = m4.load_fact_telemetry(conn)
    model_lower, model_upper = m4.train_quantile_models(df)
    model_lower_baseline, model_upper_baseline = m4.train_linear_quantile_baseline(df)
    df_smoothed = m4.apply_kalman_smoothing(df)
    windows = m4.build_window_aggregates(df_smoothed, window_minutes)
    existing = m4.load_existing_gold_windows(conn)
    m4.predict_and_update(
        conn, windows,
        model_lower, model_upper,
        model_lower_baseline, model_upper_baseline,
        existing, dry_run=False,
    )
    conn.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="drone_fleet.db")
    ap.add_argument("--hours", type=float, default=1.0)
    ap.add_argument("--window-minutes", type=int, default=5)
    args = ap.parse_args()

    build_db(args.db, args.hours, args.window_minutes)
    run_m2_m4(args.db, args.window_minutes)
    print(f"[GEN] Xong. DB: {args.db}")


if __name__ == "__main__":
    main()
