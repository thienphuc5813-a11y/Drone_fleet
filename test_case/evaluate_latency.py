"""
EVALUATE LATENCY — Độ trễ end-to-end của Dashboard (Section V, bằng chứng #1)
================================================================================
MỤC ĐÍCH (RQ024): dashboard chỉ "giúp vận hành viên ra quyết định ĐỦ NHANH"
nếu con số hiển thị trên màn hình không quá cũ so với thời điểm sự kiện thật
xảy ra trên drone. File này ĐO trực tiếp độ trễ đó bằng dữ liệu đã có sẵn
trong DB — không cần thêm hạ tầng, không cần log thêm gì mới:

    bronze_telemetry.received_at   : M1 ghi (SQLite CURRENT_TIMESTAMP) lúc
                                       1 dòng telemetry thô được đẩy vào hệ
                                       thống — mốc "dữ liệu vừa xảy ra xong".
    fact_gold_summary.updated_at   : M2 ghi lúc UPSERT cửa sổ Gold (rồi M4
                                       ghi ĐÈ lại đúng cột này khi UPDATE 4 cột
                                       của mình) — mốc "dữ liệu sẵn sàng để
                                       app.py SELECT ra hiển thị".

Vì 1 cửa sổ Gold gộp NHIỀU dòng bronze (vd. 5 phút = tối đa 300 dòng/giây),
ta lấy MAX(received_at) trong cửa sổ đó làm mốc bắt đầu — tức là đo latency
tính từ dòng dữ liệu MỚI NHẤT đóng góp vào cửa sổ, đây là latency mà vận hành
viên thực sự cảm nhận (họ quan tâm "con số mới nhất tôi thấy trễ bao lâu",
không phải trung bình toàn bộ dòng trong cửa sổ).

================================================================================
2 THÀNH PHẦN CỦA LATENCY MÀ VẬN HÀNH VIÊN THỰC SỰ CHỊU:

    (a) Pipeline latency  = updated_at - MAX(received_at trong cửa sổ)
        Thời gian M1 -> M2 -> M4 xử lý xong 1 cửa sổ 5 phút.

    (b) Polling latency   = thời gian app.py (M5) CHỜ tới lần st.rerun() kế
        tiếp mới đọc lại DB. app.py dùng time.sleep(3) mỗi vòng nên độ trễ
        polling trung bình = 1.5s (uniform 0-3s), tối đa = 3s. Đây LÀ MỘT
        PHẦN THẬT của latency tổng, không phải chi tiết cài đặt có thể bỏ
        qua — vận hành viên nhìn màn hình, không nhìn thẳng vào DB.

    Latency tổng mà vận hành viên chịu ~ (a) + (b).

Cách dùng:
    python3 evaluate_latency.py --db drone_fleet.db --window-minutes 5
    python3 evaluate_latency.py --db drone_fleet.db --dashboard-poll-sec 3 --out latency_results.csv
================================================================================
"""

import argparse
import sqlite3

import numpy as np
import pandas as pd

from schema import TelemetryPayload, window_end_of


# ==============================================================================
# 1. LOAD DATA
# ==============================================================================

def load_bronze_with_received_at(conn: sqlite3.Connection) -> pd.DataFrame:
    """Đọc bronze_telemetry, parse raw_json_payload để lấy drone_id/timestamp
    (payload) tách biệt với received_at (mốc DB ghi nhận, do M1 tạo ra)."""
    raw = pd.read_sql_query(
        "SELECT log_id, received_at, raw_json_payload FROM bronze_telemetry", conn
    )
    if raw.empty:
        raise RuntimeError(
            "bronze_telemetry rỗng — chưa có dữ liệu nào để đo latency. "
            "Chạy generate_eval_dataset.py hoặc m1_streaming.py trước."
        )

    drone_ids, timestamps = [], []
    for payload_json in raw["raw_json_payload"]:
        payload = TelemetryPayload.model_validate_json(payload_json)
        drone_ids.append(payload.drone_id)
        timestamps.append(payload.timestamp)

    raw["drone_id"] = drone_ids
    raw["event_timestamp"] = timestamps
    # SQLite CURRENT_TIMESTAMP không có timezone -> ép về UTC (đúng với thực
    # tế: cả DB và simulator đều chạy trên cùng máy, cùng chuẩn UTC).
    raw["received_at"] = pd.to_datetime(raw["received_at"], utc=True)
    return raw[["drone_id", "event_timestamp", "received_at"]]


def load_gold_updated_at(conn: sqlite3.Connection) -> pd.DataFrame:
    gold = pd.read_sql_query(
        "SELECT drone_id, window_end, updated_at FROM fact_gold_summary "
        "WHERE etr_lower_min IS NOT NULL",  # chỉ tính cửa sổ đã hoàn tất cả M2 lẫn M4
        conn,
    )
    if gold.empty:
        raise RuntimeError(
            "fact_gold_summary chưa có cửa sổ nào hoàn tất (M2 + M4 đều đã ghi). "
            "Chạy m2_analytics_pipeline.py rồi m4_ml_engineer.py trước."
        )
    gold["updated_at"] = pd.to_datetime(gold["updated_at"], utc=True)
    return gold


def load_dim_drones(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query("SELECT drone_id, profile_label FROM dim_drones", conn)


# ==============================================================================
# 2. TÍNH PIPELINE LATENCY THEO TỪNG CỬA SỔ GOLD
# ==============================================================================

def compute_pipeline_latency(
    bronze: pd.DataFrame, gold: pd.DataFrame, dim_drones: pd.DataFrame, window_minutes: int
) -> pd.DataFrame:
    bronze = bronze.copy()
    bronze["window_end"] = bronze["event_timestamp"].apply(
        lambda t: window_end_of(t, window_minutes)
    )

    # Mốc "dữ liệu mới nhất" của mỗi cửa sổ = MAX(received_at) trong cửa sổ đó
    last_received = (
        bronze.groupby(["drone_id", "window_end"])["received_at"]
        .max()
        .reset_index()
        .rename(columns={"received_at": "last_bronze_received_at"})
    )

    merged = gold.merge(last_received, on=["drone_id", "window_end"], how="inner")
    merged = merged.merge(dim_drones, on="drone_id", how="left")

    merged["pipeline_latency_sec"] = (
        merged["updated_at"] - merged["last_bronze_received_at"]
    ).dt.total_seconds()

    # Latency âm (Gold ghi TRƯỚC khi dòng bronze cuối cùng "đến") không có ý
    # nghĩa vật lý ở đây — xảy ra khi generate_eval_dataset.py ghi bronze dồn
    # dập rồi mới chạy M2/M4 sau (batch mode), khác với luồng M1 streaming
    # thật (near real-time, ghi từng dòng). Giữ lại nhưng gắn cờ rõ ràng thay
    # vì âm thầm loại bỏ, để không tạo cảm giác số liệu "đẹp" hơn thực tế.
    merged["batch_generated_flag"] = merged["pipeline_latency_sec"] < 0
    return merged


# ==============================================================================
# 3. BÁO CÁO
# ==============================================================================

def summarize(df: pd.DataFrame, dashboard_poll_sec: float) -> pd.DataFrame:
    """Tổng hợp theo profile_label + OVERALL. Cộng thêm polling latency
    trung bình (poll/2, vì uniform 0..poll) và tối đa (poll) vào latency
    tổng mà vận hành viên thực sự chịu."""
    rows = []
    for label, g in df.groupby("profile_label"):
        rows.append(_row(label, g, dashboard_poll_sec))
    rows.append(_row("OVERALL", df, dashboard_poll_sec))
    return pd.DataFrame(rows)


def _row(label: str, g: pd.DataFrame, dashboard_poll_sec: float) -> dict:
    valid = g.loc[~g["batch_generated_flag"], "pipeline_latency_sec"]
    n_negative = int(g["batch_generated_flag"].sum())
    if valid.empty:
        pipeline_mean = pipeline_p50 = pipeline_p95 = float("nan")
    else:
        pipeline_mean = float(valid.mean())
        pipeline_p50 = float(valid.median())
        pipeline_p95 = float(np.percentile(valid, 95))

    return {
        "profile_label": label,
        "n_windows": len(g),
        "n_negative_batch_artifact": n_negative,
        "pipeline_latency_mean_sec": pipeline_mean,
        "pipeline_latency_p50_sec": pipeline_p50,
        "pipeline_latency_p95_sec": pipeline_p95,
        "dashboard_poll_mean_sec": dashboard_poll_sec / 2.0,
        "dashboard_poll_max_sec": dashboard_poll_sec,
        "total_user_perceived_mean_sec": (
            pipeline_mean + dashboard_poll_sec / 2.0 if not np.isnan(pipeline_mean) else float("nan")
        ),
        "total_user_perceived_p95_sec": (
            pipeline_p95 + dashboard_poll_sec if not np.isnan(pipeline_p95) else float("nan")
        ),
    }


# ==============================================================================
# 4. MAIN
# ==============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Đo latency end-to-end (Bronze -> Gold + polling dashboard) từ drone_fleet.db"
    )
    ap.add_argument("--db", default="drone_fleet.db")
    ap.add_argument("--window-minutes", type=int, default=5)
    ap.add_argument(
        "--dashboard-poll-sec", type=float, default=3.0,
        help="Chu kỳ time.sleep() trong app.py (M5) — mặc định 3s, khớp code hiện tại",
    )
    ap.add_argument("--out", default="latency_results.csv")
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    bronze = load_bronze_with_received_at(conn)
    gold = load_gold_updated_at(conn)
    dim_drones = load_dim_drones(conn)
    conn.close()

    per_window = compute_pipeline_latency(bronze, gold, dim_drones, args.window_minutes)
    result = summarize(per_window, args.dashboard_poll_sec)

    print("=" * 100)
    print("DASHBOARD END-TO-END LATENCY — Bronze (received_at) -> Gold (updated_at) + Polling")
    print("=" * 100)
    print(result.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

    n_neg = int(per_window["batch_generated_flag"].sum())
    if n_neg > 0:
        print(
            f"\n[LƯU Ý] {n_neg}/{len(per_window)} cửa sổ có pipeline_latency_sec < 0 — "
            "đây là DẤU HIỆU dữ liệu được sinh bằng generate_eval_dataset.py (ghi bronze "
            "hàng loạt RỒI MỚI chạy M2/M4), khác với luồng M1 streaming thật chạy liên tục. "
            "Để có số liệu latency đại diện cho vận hành thật, nên đo lại trên 1 phiên chạy "
            "m1_streaming.py + m2_analytics_pipeline.py --loop + m4_ml_engineer.py đồng thời, "
            "không dùng generate_eval_dataset.py cho mục đích này."
        )

    per_window.to_csv(args.out.replace(".csv", "_per_window.csv"), index=False)
    result.to_csv(args.out, index=False)
    print(f"\n[INFO] Đã ghi {args.out} và {args.out.replace('.csv', '_per_window.csv')}")


if __name__ == "__main__":
    main()
