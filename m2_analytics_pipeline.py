"""
================================================================================
 M2 — Analytics Engineer  (Bronze -> Silver -> Gold)
================================================================================
Nhiệm vụ M2 (theo đúng cheat-sheet trong schema.py):
    Input  : SELECT raw_json_payload FROM bronze_telemetry
    Output : INSERT vào fact_telemetry (dùng SilverRecord) VÀ
              UPDATE các cột *_mean/*_std/*_status trong fact_gold_summary
              (dùng GoldSummary, CHỈ set field mình sở hữu — không đụng vào
              gps_lat_smooth/gps_lon_smooth/etr_lower_min/etr_upper_min của M4)

Đây là bản viết lại HOÀN TOÀN từ Dronedemo.py (SQL Server + pyodbc), sửa 3 vấn
đề chính:

    1. SQL Server/pyodbc -> SQLite (dùng chung 1 file drone_fleet.db với M1/M4,
       không cần .env/ODBC driver nào cả).
    2. Bỏ đoạn "sửa lỗi" replace("'", '"') / replace("True","true") thừa —
       payload ghi vào Bronze là do TelemetryPayload.model_dump_json() (chuẩn
       JSON, dùng dấu nháy kép, true/false viết thường sẵn), nên
       model_validate_json() parse thẳng được, không cần vá gì thêm.
    3. Xử lý TĂNG DẦN: chỉ đọc các dòng bronze_telemetry MỚI (log_id lớn hơn
       lần chạy trước, lưu trong bảng m2_pipeline_state) thay vì SELECT * mỗi
       lần chạy. Tầng Gold cũng chỉ tính lại đúng những cửa sổ 5 phút bị ảnh
       hưởng bởi dữ liệu mới, không GROUP BY lại toàn bộ fact_telemetry.

QUAN TRỌNG — đồng bộ với M4:
    window_end được tính bằng window_end_of() import từ schema.py — ĐÂY LÀ
    CÔNG THỨC DUY NHẤT, M4 cũng phải import và dùng đúng hàm này (xem
    m4_ml_engineer.py đã sửa). Không tự tính lại bằng SQL hay pandas riêng.

QUAN TRỌNG — quyền sở hữu cột fact_gold_summary:
    UPSERT ở đây CHỈ set các cột: battery_mean/std/status, wind_speed_mean/std,
    motor_temp_mean/std/status, network_status, updated_at.
    KHÔNG bao giờ đụng vào gps_lat_smooth/gps_lon_smooth/etr_lower_min/
    etr_upper_min — đó là 4 cột M4 sở hữu.

Cách dùng:
    python3 m2_analytics_pipeline.py --db drone_fleet.db
    python3 m2_analytics_pipeline.py --db drone_fleet.db --loop --interval-sec 5
================================================================================
"""

import argparse
import logging
import sqlite3
import statistics
import time
from datetime import datetime, timedelta, timezone

from schema import (
    DDL_STATEMENTS,
    GoldSummary,
    HealthStatus,
    SilverRecord,
    TelemetryPayload,
    classify_battery_status,
    classify_motor_temp_status,
    window_end_of,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] M2: %(message)s",
)
log = logging.getLogger("M2-Analytics")

WINDOW_MINUTES = 5

# Thêm 1 bảng nhỏ để lưu "đã xử lý Bronze tới log_id nào" — đây là cơ chế cho
# phép chạy tăng dần, không đọc lại toàn bộ bronze_telemetry mỗi lần chạy.
STATE_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS m2_pipeline_state (
    id                  INTEGER PRIMARY KEY CHECK (id = 1),
    last_bronze_log_id  INTEGER NOT NULL DEFAULT 0
);
"""


# ==============================================================================
# KHỞI TẠO DATABASE
# ==============================================================================

def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(DDL_STATEMENTS)
    conn.executescript(STATE_TABLE_DDL)
    conn.execute(
        "INSERT OR IGNORE INTO m2_pipeline_state (id, last_bronze_log_id) VALUES (1, 0)"
    )
    conn.commit()


def get_last_processed_log_id(conn: sqlite3.Connection) -> int:
    row = conn.execute(
        "SELECT last_bronze_log_id FROM m2_pipeline_state WHERE id = 1"
    ).fetchone()
    return row[0] if row else 0


def set_last_processed_log_id(conn: sqlite3.Connection, log_id: int) -> None:
    conn.execute(
        "UPDATE m2_pipeline_state SET last_bronze_log_id = ? WHERE id = 1",
        (log_id,),
    )


# ==============================================================================
# ƯỚC LƯỢNG remaining_flight_time_min (TARGET cho M4 train)
# ==============================================================================
# GHI CHÚ: đây vẫn là công thức tạm/placeholder (battery / tốc độ tiêu hao giả
# định 2%/phút), giữ nguyên tinh thần bản Dronedemo.py gốc. Khi có đủ lịch sử
# thật, nên thay bằng nội suy dựa trên drain_rate lịch sử của từng drone_id
# (đúng như mô tả trong schema.SilverRecord). Không thuộc phạm vi refactor lần
# này nên không đổi thuật toán, chỉ đổi hạ tầng (SQL Server -> SQLite, batch ->
# incremental).
def estimate_remaining_flight_time_min(payload: TelemetryPayload) -> float:
    return payload.battery_level_pct / 2.0


# ==============================================================================
# TẦNG 1: BRONZE -> SILVER (chỉ xử lý các dòng MỚI)
# ==============================================================================

def process_bronze_to_silver(conn: sqlite3.Connection) -> set:
    """
    Đọc các dòng bronze_telemetry có log_id > lần xử lý trước, validate qua
    TelemetryPayload, ghi vào fact_telemetry.

    Trả về: set các (drone_id, window_end) bị ảnh hưởng bởi dữ liệu mới —
    dùng để tầng Gold chỉ tính lại đúng các cửa sổ này (không GROUP BY toàn bộ
    fact_telemetry).
    """
    last_id = get_last_processed_log_id(conn)

    rows = conn.execute(
        "SELECT log_id, raw_json_payload FROM bronze_telemetry "
        "WHERE log_id > ? ORDER BY log_id",
        (last_id,),
    ).fetchall()

    if not rows:
        log.info("Không có bản ghi Bronze mới (đã xử lý tới log_id=%s).", last_id)
        return set()

    log.info("Đang xử lý %d bản ghi Bronze mới (log_id > %s)...", len(rows), last_id)

    affected_windows = set()
    max_log_id = last_id
    inserted, rejected = 0, 0

    for log_id, raw_json in rows:
        max_log_id = max(max_log_id, log_id)
        try:
            # JSON chuẩn (do model_dump_json() sinh ra) -> parse thẳng, không
            # cần vá lỗi nháy đơn/nháy kép hay True/False như bản cũ.
            payload = TelemetryPayload.model_validate_json(raw_json)

            silver = SilverRecord(
                drone_id=payload.drone_id,
                timestamp=payload.timestamp,
                gps_lat=payload.gps_lat,
                gps_lon=payload.gps_lon,
                altitude_m=payload.altitude_m,
                true_lat=payload.true_lat,
                true_lon=payload.true_lon,
                battery_level_pct=payload.battery_level_pct,
                wind_speed_ms=payload.wind_speed_ms,
                motor_temp_c=payload.motor_temp_c,
                hardware_ok=payload.hardware_ok,
                network_signal_pct=payload.network_signal_pct,
                is_interpolated=False,
                remaining_flight_time_min=estimate_remaining_flight_time_min(payload),
            )

            conn.execute(
                """
                INSERT INTO fact_telemetry (
                    drone_id, timestamp, gps_lat, gps_lon, altitude_m,
                    battery_level_pct, wind_speed_ms, motor_temp_c,
                    hardware_ok, network_signal_pct, is_interpolated,
                    remaining_flight_time_min, true_lat, true_lon
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    silver.drone_id,
                    silver.timestamp,
                    silver.gps_lat,
                    silver.gps_lon,
                    silver.altitude_m,
                    silver.battery_level_pct,
                    silver.wind_speed_ms,
                    silver.motor_temp_c,
                    1 if silver.hardware_ok else 0,
                    silver.network_signal_pct,
                    1 if silver.is_interpolated else 0,
                    silver.remaining_flight_time_min,
                    silver.true_lat,
                    silver.true_lon,
                ),
            )
            inserted += 1

            w_end = window_end_of(silver.timestamp, WINDOW_MINUTES)
            affected_windows.add((silver.drone_id, w_end))

        except Exception as e:
            rejected += 1
            log.error("Bronze log_id=%s KHÔNG hợp lệ, bị từ chối: %s: %s", log_id, type(e).__name__, e)

    # Dù có dòng bị từ chối, vẫn coi là "đã xử lý" (đã đọc + đánh giá) để
    # tránh vòng lặp vô hạn cố xử lý lại 1 dòng lỗi vĩnh viễn.
    set_last_processed_log_id(conn, max_log_id)
    conn.commit()

    log.info("Bronze -> Silver: %d dòng ghi thành công, %d dòng bị từ chối.", inserted, rejected)
    return affected_windows


# ==============================================================================
# TẦNG 2: SILVER -> GOLD — chỉ tính lại các cửa sổ bị ảnh hưởng
# ==============================================================================

def _fetch_window_rows(conn: sqlite3.Connection, drone_id: str, window_end: str):
    """Lấy các dòng fact_telemetry thuộc cửa sổ (window_start, window_end]."""
    window_end_dt = datetime.fromisoformat(window_end.replace("Z", "+00:00"))
    window_start_dt = window_end_dt - timedelta(minutes=WINDOW_MINUTES)
    window_start = window_start_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    return conn.execute(
        """
        SELECT battery_level_pct, wind_speed_ms, motor_temp_c, network_signal_pct
        FROM fact_telemetry
        WHERE drone_id = ?
          AND [timestamp] > ?
          AND [timestamp] <= ?
        """,
        (drone_id, window_start, window_end),
    ).fetchall()


def _safe_stdev(values: list) -> float:
    """statistics.stdev() cần >= 2 điểm dữ liệu; ít hơn thì coi std = 0."""
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values)


def aggregate_and_upsert_gold(conn: sqlite3.Connection, affected_windows: set) -> None:
    """
    Với mỗi (drone_id, window_end) bị ảnh hưởng: gom lại toàn bộ dòng
    fact_telemetry trong cửa sổ đó, tính mean/std, phân loại status, rồi
    UPSERT (INSERT ... ON CONFLICT ... DO UPDATE) vào fact_gold_summary —
    CHỈ set các cột M2 sở hữu.
    """
    if not affected_windows:
        return

    log.info("Đang cập nhật %d cửa sổ Gold bị ảnh hưởng...", len(affected_windows))
    updated = 0

    for drone_id, w_end in sorted(affected_windows):
        rows = _fetch_window_rows(conn, drone_id, w_end)
        if not rows:
            continue

        battery_vals = [r[0] for r in rows]
        wind_vals = [r[1] for r in rows]
        motor_vals = [r[2] for r in rows]
        network_vals = [r[3] for r in rows]

        battery_mean = statistics.mean(battery_vals)
        battery_std = _safe_stdev(battery_vals) or 0.1
        wind_mean = statistics.mean(wind_vals)
        wind_std = _safe_stdev(wind_vals)
        motor_mean = statistics.mean(motor_vals)
        motor_std = _safe_stdev(motor_vals) or 0.1
        network_mean = statistics.mean(network_vals)

        # SỬA: bản trước gọi classify_status(battery_mean, battery_mean, std) —
        # so 1 số với chính nó, z luôn = 0 -> status luôn GREEN bất kể drone
        # đang cảnh báo hay không. Giờ dùng ngưỡng tuyệt đối (xem schema.py).
        battery_status = classify_battery_status(battery_mean)
        motor_status = classify_motor_temp_status(motor_mean)

        if network_mean < 40:
            network_status = HealthStatus.RED
        elif network_mean < 70:
            network_status = HealthStatus.YELLOW
        else:
            network_status = HealthStatus.GREEN

        now_iso = datetime.now(timezone.utc).isoformat()

        # UPSERT thật (SQLite >= 3.24). fact_gold_summary có PRIMARY KEY
        # (drone_id, window_end) nên ON CONFLICT khớp đúng ràng buộc đó.
        # CHỈ set cột M2 sở hữu — 4 cột của M4 (gps_lat_smooth, gps_lon_smooth,
        # etr_lower_min, etr_upper_min) KHÔNG xuất hiện ở đây nên không bao
        # giờ bị ghi đè, dù là lần INSERT đầu hay lần UPDATE sau.
        conn.execute(
            """
            INSERT INTO fact_gold_summary (
                drone_id, window_end,
                battery_mean, battery_std, battery_status,
                wind_speed_mean, wind_speed_std,
                motor_temp_mean, motor_temp_std, motor_temp_status,
                network_status, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (drone_id, window_end) DO UPDATE SET
                battery_mean      = excluded.battery_mean,
                battery_std       = excluded.battery_std,
                battery_status    = excluded.battery_status,
                wind_speed_mean   = excluded.wind_speed_mean,
                wind_speed_std    = excluded.wind_speed_std,
                motor_temp_mean   = excluded.motor_temp_mean,
                motor_temp_std    = excluded.motor_temp_std,
                motor_temp_status = excluded.motor_temp_status,
                network_status    = excluded.network_status,
                updated_at        = excluded.updated_at
            """,
            (
                drone_id, w_end,
                battery_mean, battery_std, battery_status.value,
                wind_mean, wind_std,
                motor_mean, motor_std, motor_status.value,
                network_status.value, now_iso,
            ),
        )
        updated += 1

    conn.commit()
    log.info("Gold: đã UPSERT %d cửa sổ (drone_id, window_end).", updated)


# ==============================================================================
# 1 VÒNG CHẠY PIPELINE ĐẦY ĐỦ
# ==============================================================================

def run_once(conn: sqlite3.Connection) -> None:
    affected_windows = process_bronze_to_silver(conn)
    aggregate_and_upsert_gold(conn, affected_windows)


# ==============================================================================
# MAIN
# ==============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="M2 — Bronze -> Silver -> Gold (SQLite, incremental)")
    parser.add_argument("--db", default="drone_fleet.db", help="Đường dẫn file SQLite (dùng chung với M1/M4)")
    parser.add_argument("--loop", action="store_true", help="Chạy liên tục thay vì 1 lần rồi thoát")
    parser.add_argument("--interval-sec", type=float, default=5.0, help="Chu kỳ lặp khi dùng --loop")
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout = 5000;")  # SỬA: tránh "database is locked"
    init_db(conn)

    try:
        if args.loop:
            log.info("Chạy M2 ở chế độ lặp, chu kỳ %.1fs. Nhấn Ctrl+C để dừng.", args.interval_sec)
            while True:
                run_once(conn)
                time.sleep(args.interval_sec)
        else:
            run_once(conn)
    except KeyboardInterrupt:
        log.info("Nhận Ctrl+C — dừng M2.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
