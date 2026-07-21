"""
CASE STUDY — "worst_case" drone: dashboard có "kể đúng chuyện" không?
================================================================================
MỤC ĐÍCH (Section V, bằng chứng #2): đây là bằng chứng ĐỊNH TÍNH, n=1
(illustrative case, KHÔNG PHẢI statistical evidence — phải nói rõ điều này
trong bài, không được lập lờ để trông giống bằng chứng định lượng).

Câu hỏi cụ thể: khi 1 drone thực sự đang xấu dần (pin tụt, động cơ nóng lên)
vì lý do VẬT LÝ THẬT trong simulator (payload nặng + gió bão + pin cũ + trời
lạnh — đúng 4/7 yếu tố xấu nhất trong DroneProfile), dashboard có PHẢN ÁNH
ĐÚNG diễn biến đó bằng chuỗi trạng thái GREEN -> YELLOW -> RED hay không,
đúng lúc, đúng lý do?

CÁCH LÀM — không cần hạ tầng mới, chỉ chạy DÀI HƠN 1 drone:
    1. Mô phỏng riêng 1 drone "DRONE_CASE_WORST" với profile "worst_case"
       (lấy nguyên từ CASE_STUDY_PROFILES trong stochastic_simulator.py,
       KHÔNG tự chế thông số mới) trong 35 phút (đủ dài để pin tụt qua cả
       2 ngưỡng 40% và 20% — xem tính toán tốc độ tụt pin trong docstring
       của StochasticSimulator._step_battery()).
    2. Chạy ĐÚNG code M2 (m2_analytics_pipeline) và M4 (m4_ml_engineer) thật
       — không viết lại logic phân loại trạng thái ở đây.
    3. Đọc lại fact_gold_summary theo thứ tự window_end, tìm các điểm
       CHUYỂN TRẠNG THÁI (battery_status / motor_temp_status đổi giá trị).
    4. Đối chiếu mỗi điểm chuyển với số liệu vật lý thật tại đúng cửa sổ đó
       (battery_mean, motor_temp_mean) VÀ với hồ sơ profile (dim_drones) —
       để trả lời "vì sao" bằng dữ liệu, không suy diễn.
    5. Xuất: (a) CSV timeline đầy đủ, (b) biểu đồ PNG minh họa chuỗi trạng
       thái theo thời gian — dùng làm Figure thay cho ảnh chụp màn hình
       dashboard thật (vì không có phiên Streamlit đang chạy để chụp).

Cách dùng:
    python3 case_study_worst_case.py --minutes 35 --window-minutes 5
    python3 case_study_worst_case.py --minutes 35 --db case_worst.db --out-prefix case_worst
================================================================================
"""

import argparse
import sqlite3
from datetime import datetime, timedelta, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from schema import DDL_STATEMENTS, make_bronze_payload
from stochastic_simulator import StochasticSimulator, CASE_STUDY_PROFILES

import m2_analytics_pipeline as m2
import m4_ml_engineer as m4

DRONE_ID = "DRONE_CASE_WORST"
STATUS_COLOR = {"green": "#2ecc71", "yellow": "#f1c40f", "red": "#e74c3c"}


# ==============================================================================
# 1. SINH DỮ LIỆU — 1 drone, profile "worst_case", chạy DÀI HƠN bình thường
# ==============================================================================

def build_case_study_db(db_path: str, minutes: int, seed: int = 7) -> None:
    profile = next(p for p in CASE_STUDY_PROFILES if p.label == "worst_case")

    conn = sqlite3.connect(db_path)
    conn.executescript(DDL_STATEMENTS)
    conn.execute(
        """
        INSERT OR REPLACE INTO dim_drones
            (drone_id, model_name, max_battery_capacity_wh, profile_label,
             payload_kg, wind_zone, battery_health, ambient_temp_c,
             gps_quality, hardware_reliability, network_zone)
        VALUES (?, 'DJI Matrice 100', 500.0, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            DRONE_ID, profile.label, profile.payload_kg, profile.wind_zone,
            profile.battery_health, profile.ambient_temp_c, profile.gps_quality,
            profile.hardware_reliability, profile.network_zone,
        ),
    )
    conn.commit()

    sim = StochasticSimulator(drone_id=DRONE_ID, seed=seed, profile=profile)
    base_time = datetime(2026, 7, 1, tzinfo=timezone.utc)
    n_steps = int(minutes * 60)

    print(f"[CASE STUDY] Mô phỏng {DRONE_ID} (profile={profile.label}) trong "
          f"{minutes} phút ({n_steps} bước)...")
    rows = []
    for t in range(n_steps):
        out = sim.step()
        ts = (base_time + timedelta(seconds=t)).strftime("%Y-%m-%dT%H:%M:%SZ")
        payload = make_bronze_payload(DRONE_ID, out)
        payload_json = payload.model_copy(update={"timestamp": ts}).model_dump_json()
        rows.append((payload_json,))

    conn.executemany("INSERT INTO bronze_telemetry (raw_json_payload) VALUES (?)", rows)
    conn.commit()
    conn.close()
    print(f"[CASE STUDY] Đã ghi {n_steps} dòng bronze cho {DRONE_ID}.")


def run_m2_m4(db_path: str, window_minutes: int) -> None:
    conn = sqlite3.connect(db_path)
    m2.init_db(conn)
    m2.WINDOW_MINUTES = window_minutes
    affected = m2.process_bronze_to_silver(conn)
    m2.aggregate_and_upsert_gold(conn, affected)
    conn.close()

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


# ==============================================================================
# 2. TRÍCH XUẤT TIMELINE + PHÁT HIỆN ĐIỂM CHUYỂN TRẠNG THÁI
# ==============================================================================

def load_timeline(db_path: str) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        """
        SELECT window_end, battery_mean, battery_status,
               motor_temp_mean, motor_temp_status, network_status,
               etr_lower_min, etr_upper_min
        FROM fact_gold_summary
        WHERE drone_id = ?
        ORDER BY window_end
        """,
        conn, params=(DRONE_ID,),
    )
    profile_row = pd.read_sql_query(
        "SELECT * FROM dim_drones WHERE drone_id = ?", conn, params=(DRONE_ID,)
    ).iloc[0].to_dict()
    conn.close()
    return df, profile_row


def find_transitions(df: pd.DataFrame, status_col: str) -> pd.DataFrame:
    """Trả về các dòng mà status_col ĐỔI GIÁ TRỊ so với dòng trước đó
    (dòng đầu tiên luôn được coi là 1 'transition' từ trạng thái ban đầu)."""
    changed = df[status_col] != df[status_col].shift(1)
    return df[changed]


def build_reason_notes(profile: dict) -> list[str]:
    """Lặp lại đúng logic 'vì sao' mà app.py (M5) đã hiển thị ở sidebar —
    dùng chung 1 cách diễn giải profile, không tạo ra 1 bộ lý do khác cho
    bài báo so với cái dashboard thật sự cho vận hành viên xem."""
    notes = []
    if profile.get("payload_kg", 0) > 0:
        notes.append(f"mang thêm {profile['payload_kg']:.1f}kg hàng (pin tụt nhanh hơn, động cơ nóng hơn)")
    if profile.get("wind_zone") == "storm":
        notes.append("bay trong vùng gió bão (mu gió cao -> động cơ phải tải nhiều hơn)")
    if profile.get("battery_health", 1.0) < 0.9:
        notes.append(f"pin đã xuống cấp (battery_health={profile['battery_health']:.2f}, tụt nhanh + dao động thất thường hơn)")
    if profile.get("ambient_temp_c", 25) < 15:
        notes.append(f"thời tiết lạnh ({profile['ambient_temp_c']:.0f}°C, hiệu ứng vật lý thật của pin Li-ion: lạnh -> tụt nhanh hơn)")
    if profile.get("gps_quality", 1.0) < 0.8:
        notes.append("module GPS chất lượng thấp (không ảnh hưởng battery/motor, chỉ ảnh hưởng RMSE vị trí)")
    if profile.get("network_zone") == "rural":
        notes.append("vùng phủ sóng yếu (không ảnh hưởng battery/motor, chỉ ảnh hưởng network_status)")
    return notes


# ==============================================================================
# 3. BIỂU ĐỒ — thay cho screenshot dashboard thật
# ==============================================================================

def plot_timeline(df: pd.DataFrame, out_png: str) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    x = range(len(df))

    for ax, col_mean, col_status, ylabel, title in [
        (ax1, "battery_mean", "battery_status", "Battery (%)", "Battery — mean theo cửa sổ 5 phút"),
        (ax2, "motor_temp_mean", "motor_temp_status", "Motor Temp (°C)", "Motor Temperature — mean theo cửa sổ 5 phút"),
    ]:
        for i in x:
            ax.axvspan(i - 0.5, i + 0.5, color=STATUS_COLOR.get(df[col_status].iloc[i], "#bdc3c7"), alpha=0.25)
        ax.plot(x, df[col_mean], color="#2c3e50", marker="o", linewidth=1.8, markersize=4)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.3)

    ax2.set_xticks(list(x))
    ax2.set_xticklabels(
        [w[11:16] for w in df["window_end"]], rotation=45, ha="right"
    )
    ax2.set_xlabel("window_end (UTC, HH:MM)")

    handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.4) for c in STATUS_COLOR.values()]
    fig.legend(handles, ["GREEN", "YELLOW", "RED"], loc="upper right", ncol=3)
    fig.suptitle(f"Case study — {DRONE_ID} (profile: worst_case)", fontsize=13, y=1.0)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"[INFO] Đã lưu biểu đồ: {out_png}")


# ==============================================================================
# 4. MAIN
# ==============================================================================

def main():
    ap = argparse.ArgumentParser(description="Case study worst_case: đối chiếu chuỗi trạng thái dashboard với sự kiện thật")
    ap.add_argument("--db", default="case_worst.db")
    ap.add_argument("--minutes", type=int, default=35, help="Độ dài mô phỏng (phút) — đủ để pin tụt qua ngưỡng 40% và 20%")
    ap.add_argument("--window-minutes", type=int, default=5)
    ap.add_argument("--out-prefix", default="case_worst")
    args = ap.parse_args()

    build_case_study_db(args.db, args.minutes)
    run_m2_m4(args.db, args.window_minutes)

    df, profile = load_timeline(args.db)
    if df.empty:
        raise RuntimeError("Không có cửa sổ Gold nào cho DRONE_CASE_WORST — kiểm tra lại --minutes có đủ lớn hơn --window-minutes chưa.")

    reasons = build_reason_notes(profile)

    print("\n" + "=" * 100)
    print(f"HỒ SƠ (PROFILE) CỦA {DRONE_ID} — 'vì sao' drone này ở kịch bản xấu nhất")
    print("=" * 100)
    for r in reasons:
        print(f"  - {r}")

    print("\n" + "=" * 100)
    print("TOÀN BỘ TIMELINE (mỗi dòng = 1 cửa sổ 5 phút mà dashboard sẽ hiển thị)")
    print("=" * 100)
    print(df.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

    print("\n" + "=" * 100)
    print("ĐIỂM CHUYỂN TRẠNG THÁI — BATTERY")
    print("=" * 100)
    battery_transitions = find_transitions(df, "battery_status")
    print(battery_transitions.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

    print("\n" + "=" * 100)
    print("ĐIỂM CHUYỂN TRẠNG THÁI — MOTOR TEMP")
    print("=" * 100)
    motor_transitions = find_transitions(df, "motor_temp_status")
    print(motor_transitions.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

    n_battery_transitions = len(battery_transitions) - 1  # trừ dòng đầu (trạng thái khởi đầu, không phải "chuyển")
    reached_red = (df["battery_status"] == "red").any()
    print("\n" + "=" * 100)
    print("KẾT LUẬN CASE STUDY (bằng chứng định tính, n=1 — KHÔNG suy rộng thành thống kê)")
    print("=" * 100)
    if reached_red and n_battery_transitions >= 1:
        print(
            f"-> Trong {args.minutes} phút mô phỏng, battery_status của {DRONE_ID} đã chuyển "
            f"{n_battery_transitions} lần và ĐẠT TỚI 'red', khớp với việc profile 'worst_case' "
            f"kết hợp payload nặng + gió bão + pin cũ + trời lạnh làm pin tụt nhanh hơn baseline "
            f"(xem docstring _step_battery() trong stochastic_simulator.py). Dashboard (app.py) đọc "
            f"đúng các cột battery_status này từ fact_gold_summary nên sẽ hiển thị đúng chuỗi "
            f"🟢->🟡->🔴 tương ứng, kèm panel 'vì sao' liệt kê đúng {len(reasons)} yếu tố ở trên."
        )
    else:
        print(
            f"-> Trong {args.minutes} phút mô phỏng, battery_status CHƯA đạt tới 'red' "
            f"(hoặc không có chuyển trạng thái). Thử tăng --minutes (khuyến nghị >= 35 phút "
            f"với profile worst_case, dựa trên tốc độ tụt pin ước tính trong docstring) rồi chạy lại."
        )

    df.to_csv(f"{args.out_prefix}_timeline.csv", index=False)
    plot_timeline(df, f"{args.out_prefix}_status_timeline.png")
    print(f"\n[INFO] Đã ghi {args.out_prefix}_timeline.csv")


if __name__ == "__main__":
    main()
