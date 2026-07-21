"""
================================================================================
 evaluate_metrics.py — ĐÁNH GIÁ KHOA HỌC cho pipeline Drone Fleet Monitoring
================================================================================
File này KHÔNG thuộc pipeline sản xuất (M1/M2/M4/M5) — nó CHỈ ĐỌC dữ liệu đã
có trong drone_fleet.db để tính các chỉ số định lượng, phục vụ phần
"Results/Discussion" của một bài báo khoa học thay vì chỉ mô tả hệ thống.

Trả lời 2 câu hỏi nghiên cứu:

    RQ1 (định vị GPS):
        Kalman filter (mô hình CHÍNH) có làm giảm sai số định vị so với GPS
        thô và so với baseline EMA hay không, giảm bao nhiêu mét, có ý nghĩa
        thống kê không (paired t-test)? Có khác nhau giữa các kịch bản
        DroneProfile không (vd GPS kém ở "poor_gps_rural")?

        Đo bằng: RMSE / MAE (đơn vị: MÉT, không phải độ) so với true_lat/
        true_lon — vị trí THẬT do StochasticSimulator sinh ra (xem
        stochastic_simulator.py + schema.py, field true_lat/true_lon).

    RQ2 (ước lượng ETR — Estimated Time Remaining):
        Khoảng tin cậy [etr_lower, etr_upper] do Gradient Boosting Quantile
        Regression (mô hình CHÍNH) sinh ra có "well-calibrated" hay không —
        tức là giá trị remaining_flight_time_min THẬT có rơi vào khoảng đó
        đúng ~80% số lần (do dùng quantile 0.10/0.90) hay không? So với
        baseline Linear Quantile Regression thì sao (pinball loss)?

CÁCH DÙNG:
    python3 evaluate_metrics.py --db drone_fleet.db
    python3 evaluate_metrics.py --db drone_fleet.db --report report.md

LƯU Ý QUAN TRỌNG (đọc trước khi trích số liệu vào bài báo):
    1. true_lat/true_lon CHỈ tồn tại trong fact_telemetry nếu dữ liệu được
       sinh SAU KHI schema.py/stochastic_simulator.py/m2_analytics_pipeline.py
       đã được cập nhật (bản này). Dữ liệu cũ (trước bản cập nhật) sẽ có
       true_lat/true_lon = NULL -> script sẽ CẢNH BÁO và dừng phần RQ1, đề
       nghị chạy lại `rm -f drone_fleet.db && ./run_all.sh` để sinh dữ liệu
       mới có ground truth.
    2. remaining_flight_time_min hiện được M2 tính bằng công thức PLACEHOLDER
       xác định (battery_level_pct / 2.0 — xem estimate_remaining_flight_time_min()
       trong m2_analytics_pipeline.py), KHÔNG có nhiễu ngẫu nhiên. Vì vậy
       khoảng tin cậy ETR ở giai đoạn này gần như suy biến (lower ≈ upper) và
       coverage sẽ gần 100% một cách "dễ dàng" — đây là hạn chế của DỮ LIỆU,
       không phải lỗi tính toán. Script vẫn tính đủ số liệu, nhưng in rõ cảnh
       báo này ra để không bị hiểu nhầm là mô hình "hoàn hảo". Muốn có số liệu
       ETR thật sự có ý nghĩa, cần thay công thức ở M2 bằng ước lượng có nhiễu
       (vd nội suy từ lịch sử drain rate thật, như mô tả gốc trong schema.py).
================================================================================
"""

import argparse
import sqlite3
import sys
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy import stats

from schema import window_end_of

# Tái sử dụng NGUYÊN VẸN thuật toán làm mượt của M4 (Kalman + EMA) — không
# viết lại thuật toán ở đây, để đảm bảo số liệu đánh giá phản ánh ĐÚNG những
# gì pipeline thật đang chạy, không phải một bản cài đặt khác đi.
from m4_ml_engineer import (
    apply_kalman_smoothing,
    FEATURE_COLS,
    LOWER_Q,
    UPPER_Q,
)

METERS_PER_DEGREE_LAT = 111_320.0  # xấp xỉ, đủ chính xác cho khoảng cách nhỏ (vài km)


# ==============================================================================
# TIỆN ÍCH: khoảng cách haversine xấp xỉ phẳng (đủ tốt cho phạm vi bay < vài km)
# ==============================================================================
def positional_error_m(lat_pred, lon_pred, lat_true, lon_true) -> np.ndarray:
    lat_true_rad = np.radians(lat_true)
    dlat_m = (lat_pred - lat_true) * METERS_PER_DEGREE_LAT
    dlon_m = (lon_pred - lon_true) * METERS_PER_DEGREE_LAT * np.cos(lat_true_rad)
    return np.sqrt(dlat_m**2 + dlon_m**2)


def rmse(err_m: np.ndarray) -> float:
    return float(np.sqrt(np.mean(err_m**2)))


def mae(err_m: np.ndarray) -> float:
    return float(np.mean(np.abs(err_m)))


# ==============================================================================
# RQ1 — ĐỘ CHÍNH XÁC ĐỊNH VỊ: Raw vs EMA (baseline) vs Kalman (mô hình chính)
# ==============================================================================
def evaluate_gps_accuracy(conn: sqlite3.Connection) -> dict | None:
    df = pd.read_sql_query(
        "SELECT drone_id, timestamp, gps_lat, gps_lon, true_lat, true_lon "
        "FROM fact_telemetry ORDER BY drone_id, timestamp",
        conn,
    )
    if df.empty:
        print("[RQ1] fact_telemetry rỗng — chưa có dữ liệu để đánh giá.")
        return None

    n_missing_truth = df["true_lat"].isna().sum()
    if n_missing_truth == len(df):
        print(
            "[RQ1] ⚠️  KHÔNG có ground truth (true_lat/true_lon đều NULL).\n"
            "       Dữ liệu này được sinh TRƯỚC KHI cập nhật schema/simulator.\n"
            "       Hãy chạy: rm -f drone_fleet.db drone_fleet.db-* && ./run_all.sh\n"
            "       để sinh dữ liệu MỚI có ground truth rồi chạy lại script này."
        )
        return None
    elif n_missing_truth > 0:
        print(
            f"[RQ1] ⚠️  Có {n_missing_truth}/{len(df)} dòng thiếu ground truth "
            f"(dữ liệu cũ lẫn với dữ liệu mới) — các dòng này sẽ bị loại khỏi RQ1."
        )
        df = df.dropna(subset=["true_lat", "true_lon"]).reset_index(drop=True)

    # Chạy lại CHÍNH XÁC thuật toán mà M4 dùng trong pipeline thật, để RMSE
    # phản ánh đúng hệ thống đang chạy (không phải 1 cài đặt song song khác).
    df_smoothed = apply_kalman_smoothing(df)

    err_raw = positional_error_m(df_smoothed["gps_lat"], df_smoothed["gps_lon"],
                                  df_smoothed["true_lat"], df_smoothed["true_lon"])
    err_ema = positional_error_m(df_smoothed["gps_lat_ema"], df_smoothed["gps_lon_ema"],
                                  df_smoothed["true_lat"], df_smoothed["true_lon"])
    err_kalman = positional_error_m(df_smoothed["gps_lat_smooth"], df_smoothed["gps_lon_smooth"],
                                     df_smoothed["true_lat"], df_smoothed["true_lon"])

    overall = pd.DataFrame({
        "method": ["Raw GPS (chưa lọc)", "EMA (baseline)", "Kalman filter (mô hình chính)"],
        "RMSE_m": [rmse(err_raw), rmse(err_ema), rmse(err_kalman)],
        "MAE_m":  [mae(err_raw), mae(err_ema), mae(err_kalman)],
        "n_points": [len(err_raw)] * 3,
    })

    # Paired t-test trên SAI SỐ BÌNH PHƯƠNG (cùng 1 tập điểm, 3 phương pháp đo
    # trên CÙNG dữ liệu -> so sánh có ghép cặp là đúng thiết kế, mạnh hơn
    # t-test độc lập).
    t_kalman_vs_raw = stats.ttest_rel(err_kalman**2, err_raw**2)
    t_kalman_vs_ema = stats.ttest_rel(err_kalman**2, err_ema**2)
    t_ema_vs_raw = stats.ttest_rel(err_ema**2, err_raw**2)

    significance = pd.DataFrame({
        "so_sanh": [
            "Kalman vs Raw GPS",
            "Kalman vs EMA (baseline)",
            "EMA (baseline) vs Raw GPS",
        ],
        "t_statistic": [t_kalman_vs_raw.statistic, t_kalman_vs_ema.statistic, t_ema_vs_raw.statistic],
        "p_value": [t_kalman_vs_raw.pvalue, t_kalman_vs_ema.pvalue, t_ema_vs_raw.pvalue],
        "co_y_nghia_thong_ke_p<0.05": [
            t_kalman_vs_raw.pvalue < 0.05,
            t_kalman_vs_ema.pvalue < 0.05,
            t_ema_vs_raw.pvalue < 0.05,
        ],
    })

    # Breakdown theo profile_label (dim_drones) — trả lời "GPS kém đi ở kịch
    # bản nào rõ rệt nhất" (vd poor_gps_rural, worst_case).
    profiles = pd.read_sql_query("SELECT drone_id, profile_label FROM dim_drones", conn)
    df_smoothed = df_smoothed.merge(profiles, on="drone_id", how="left")
    df_smoothed["err_raw_m"] = err_raw.values
    df_smoothed["err_kalman_m"] = err_kalman.values
    df_smoothed["err_ema_m"] = err_ema.values

    by_profile = (
        df_smoothed.groupby("profile_label")
        .agg(
            RMSE_raw_m=("err_raw_m", lambda s: rmse(s.values)),
            RMSE_kalman_m=("err_kalman_m", lambda s: rmse(s.values)),
            RMSE_ema_m=("err_ema_m", lambda s: rmse(s.values)),
            n_points=("err_raw_m", "size"),
        )
        .reset_index()
        .sort_values("RMSE_kalman_m", ascending=False)
    )

    return {"overall": overall, "significance": significance, "by_profile": by_profile}


# ==============================================================================
# RQ2 — ĐỘ TIN CẬY (CALIBRATION) CỦA ETR: GBM (chính) vs Linear (baseline)
# ==============================================================================
def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
    diff = y_true - y_pred
    return float(np.mean(np.maximum(quantile * diff, (quantile - 1) * diff)))


def evaluate_etr_calibration(conn: sqlite3.Connection, window_minutes: int = 5) -> dict | None:
    gold = pd.read_sql_query(
        "SELECT drone_id, window_end, etr_lower_min, etr_upper_min, "
        "etr_lower_baseline, etr_upper_baseline FROM fact_gold_summary",
        conn,
    )
    gold = gold.dropna(subset=["etr_lower_min", "etr_upper_min",
                                "etr_lower_baseline", "etr_upper_baseline"])
    if gold.empty:
        print("[RQ2] fact_gold_summary chưa có dự đoán ETR nào (chạy m4_ml_engineer.py trước).")
        return None

    telemetry = pd.read_sql_query(
        "SELECT drone_id, timestamp, remaining_flight_time_min FROM fact_telemetry "
        "WHERE remaining_flight_time_min IS NOT NULL ORDER BY drone_id, timestamp",
        conn,
    )
    telemetry["window_end"] = telemetry["timestamp"].apply(lambda t: window_end_of(t, window_minutes))
    # Giá trị "thật" đại diện cho cửa sổ = giá trị đo GẦN NHẤT với cuối cửa sổ
    # (khớp đúng cách M4 lấy gps_lat_smooth = "last" trong cửa sổ).
    y_actual = (
        telemetry.groupby(["drone_id", "window_end"])["remaining_flight_time_min"]
        .last()
        .reset_index()
        .rename(columns={"remaining_flight_time_min": "y_actual"})
    )

    merged = gold.merge(y_actual, on=["drone_id", "window_end"], how="inner")
    if merged.empty:
        print("[RQ2] Không khớp được cửa sổ Gold với giá trị thực tế trong fact_telemetry.")
        return None

    y = merged["y_actual"].values

    def _coverage(lower, upper):
        return float(np.mean((y >= lower) & (y <= upper)))

    def _width(lower, upper):
        return float(np.mean(upper - lower))

    nominal_coverage = UPPER_Q - LOWER_Q  # 0.90 - 0.10 = 0.80 -> kỳ vọng ~80%

    summary = pd.DataFrame({
        "model": ["Gradient Boosting Quantile Regression (chính)",
                  "Linear Quantile Regression (baseline)"],
        "coverage_thuc_te": [
            _coverage(merged["etr_lower_min"], merged["etr_upper_min"]),
            _coverage(merged["etr_lower_baseline"], merged["etr_upper_baseline"]),
        ],
        "coverage_ky_vong": [nominal_coverage, nominal_coverage],
        "do_rong_TB_phut": [
            _width(merged["etr_lower_min"], merged["etr_upper_min"]),
            _width(merged["etr_lower_baseline"], merged["etr_upper_baseline"]),
        ],
        "pinball_loss_lower(q=0.10)": [
            pinball_loss(y, merged["etr_lower_min"].values, LOWER_Q),
            pinball_loss(y, merged["etr_lower_baseline"].values, LOWER_Q),
        ],
        "pinball_loss_upper(q=0.90)": [
            pinball_loss(y, merged["etr_upper_min"].values, UPPER_Q),
            pinball_loss(y, merged["etr_upper_baseline"].values, UPPER_Q),
        ],
    })

    # Cảnh báo về placeholder target (xem docstring đầu file)
    is_deterministic = np.allclose(merged["etr_lower_min"], merged["etr_upper_min"], atol=0.5)
    if is_deterministic:
        print(
            "[RQ2] ⚠️  Khoảng tin cậy gần như suy biến (lower ≈ upper) ở phần lớn "
            "cửa sổ — remaining_flight_time_min hiện là hàm XÁC ĐỊNH của "
            "battery_level_pct (công thức placeholder trong M2), không có nhiễu "
            "thật. Coverage/pinball loss ở bảng dưới vẫn ĐÚNG về mặt tính toán, "
            "nhưng KHÔNG phản ánh đúng độ khó dự đoán thực tế — cần thay công "
            "thức ước lượng target trong m2_analytics_pipeline.py trước khi dùng "
            "số liệu này làm kết luận chính của bài báo."
        )

    return {"summary": summary, "n_windows": len(merged), "is_deterministic_target": is_deterministic}


# ==============================================================================
# IN KẾT QUẢ + XUẤT BÁO CÁO MARKDOWN
# ==============================================================================
def _df_to_md(df: pd.DataFrame, float_fmt: str = "{:.4f}") -> str:
    df_fmt = df.copy()
    for c in df_fmt.select_dtypes(include=[float]).columns:
        df_fmt[c] = df_fmt[c].map(lambda v: float_fmt.format(v))
    return df_fmt.to_markdown(index=False)


def print_and_build_report(gps_result: dict | None, etr_result: dict | None) -> str:
    lines = [
        "# Báo cáo đánh giá định lượng — Drone Fleet Monitoring",
        f"_Sinh lúc: {datetime.now(timezone.utc).isoformat()}_",
        "",
    ]

    print("\n" + "=" * 80)
    print("RQ1 — ĐỘ CHÍNH XÁC ĐỊNH VỊ GPS (RMSE/MAE, đơn vị: mét)")
    print("=" * 80)
    lines += ["## RQ1 — Độ chính xác định vị GPS (RMSE/MAE, đơn vị: mét)", ""]
    if gps_result is None:
        print("(Bỏ qua — xem cảnh báo phía trên)")
        lines += ["_Bỏ qua — chưa có ground truth true_lat/true_lon trong dữ liệu hiện tại._", ""]
    else:
        print("\n-- Tổng thể --")
        print(gps_result["overall"].to_string(index=False))
        lines += ["### Tổng thể", "", _df_to_md(gps_result["overall"]), ""]

        print("\n-- Kiểm định ý nghĩa thống kê (paired t-test trên sai số bình phương) --")
        print(gps_result["significance"].to_string(index=False))
        lines += ["### Kiểm định ý nghĩa thống kê (paired t-test)", "",
                  _df_to_md(gps_result["significance"], "{:.6g}"), ""]

        print("\n-- Theo từng kịch bản (DroneProfile) --")
        print(gps_result["by_profile"].to_string(index=False))
        lines += ["### Theo từng kịch bản (DroneProfile)", "", _df_to_md(gps_result["by_profile"]), ""]

    print("\n" + "=" * 80)
    print("RQ2 — ĐỘ TIN CẬY (CALIBRATION) CỦA KHOẢNG DỰ ĐOÁN ETR")
    print("=" * 80)
    lines += ["## RQ2 — Độ tin cậy (calibration) của khoảng dự đoán ETR", ""]
    if etr_result is None:
        print("(Bỏ qua — xem cảnh báo phía trên)")
        lines += ["_Bỏ qua — chưa đủ dữ liệu Gold để đánh giá._", ""]
    else:
        print(f"\nSố cửa sổ được đánh giá: {etr_result['n_windows']}")
        print(etr_result["summary"].to_string(index=False))
        lines += [
            f"Số cửa sổ được đánh giá: **{etr_result['n_windows']}**", "",
            _df_to_md(etr_result["summary"]), "",
        ]
        if etr_result["is_deterministic_target"]:
            lines += [
                "> ⚠️ **Lưu ý:** khoảng tin cậy gần như suy biến vì "
                "`remaining_flight_time_min` hiện là hàm xác định của "
                "`battery_level_pct` (placeholder trong M2, chưa có nhiễu thật). "
                "Cần thay công thức này trước khi dùng số liệu ETR làm kết luận "
                "chính thức.", "",
            ]

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Đánh giá khoa học: RMSE định vị GPS + coverage/pinball loss ETR"
    )
    parser.add_argument("--db", default="drone_fleet.db", help="Đường dẫn file SQLite")
    parser.add_argument("--window-minutes", type=int, default=5, help="Phải KHỚP với M2/M4")
    parser.add_argument("--report", default="evaluation_report.md",
                         help="Đường dẫn file Markdown xuất báo cáo (đặt rỗng '' để tắt)")
    args = parser.parse_args()

    warnings.filterwarnings("ignore", category=FutureWarning)

    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    try:
        gps_result = evaluate_gps_accuracy(conn)
        etr_result = evaluate_etr_calibration(conn, args.window_minutes)
    finally:
        conn.close()

    report_md = print_and_build_report(gps_result, etr_result)

    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            f.write(report_md)
        print(f"\n[INFO] Đã ghi báo cáo Markdown: {args.report}")

    if gps_result is None and etr_result is None:
        sys.exit(1)


if __name__ == "__main__":
    main()
