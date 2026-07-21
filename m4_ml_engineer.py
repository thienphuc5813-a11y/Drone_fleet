"""
================================================================================
 M4 (ML ENGINEER) — Kalman Smoothing (GPS) + Quantile Regression (ETR)
================================================================================
Theo đúng phân công trong schema.py:

    Input  : SELECT * FROM fact_telemetry
              (dùng remaining_flight_time_min làm target)
              import KalmanFilter2D của M3 (file kalman_filter.py)
    Output : UPDATE fact_gold_summary
              SET gps_lat_smooth=, gps_lon_smooth=, etr_lower_min=, etr_upper_min=
              (mô hình CHÍNH — KHÔNG đụng cột *_mean/*_std/*_status của M2)

BỔ SUNG (phục vụ đánh giá khoa học — so sánh với baseline):
    Ngoài mô hình CHÍNH, M4 giờ CŨNG chạy song song 2 baseline đơn giản hơn
    trên CÙNG 1 input, để evaluate_metrics.py có thể so sánh định lượng:
        - GPS  : EMA (Exponential Moving Average)  đối chứng cho Kalman filter
                  -> ghi vào gps_lat_ema / gps_lon_ema
        - ETR  : Linear Quantile Regression         đối chứng cho Gradient
                  Boosting Quantile Regression
                  -> ghi vào etr_lower_baseline / etr_upper_baseline
    Baseline KHÔNG thay thế mô hình chính, chỉ để có căn cứ trả lời câu hỏi
    "mô hình phức tạp hơn có thực sự tốt hơn mô hình đơn giản hay không, tốt
    hơn bao nhiêu, có ý nghĩa thống kê không" — đúng tinh thần 1 bài nghiên
    cứu khoa học (có đối chứng), không chỉ 1 hệ thống demo.

QUAN TRỌNG — điều cần đồng bộ với M2 (ĐÃ SỬA):
    Trước đây M4 tự tính window_end bằng pandas `.floor()` riêng, còn M2 tính
    bằng SQL riêng — 2 công thức này có nguy cơ lệch nhau (floor vs ceil, lệch
    múi giờ) khiến M4 UPDATE 0 dòng dù code chạy không lỗi.
    Giờ CẢ M2 VÀ M4 ĐỀU import và gọi CHUNG 1 hàm duy nhất: `window_end_of()`
    từ schema.py. Không tự định nghĩa lại hàm này ở đây nữa.

Cách dùng:
    python3 m4_ml_engineer.py --db drone_fleet.db
    python3 m4_ml_engineer.py --db drone_fleet.db --window-minutes 5 --dry-run
================================================================================
"""

import argparse
import sqlite3
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import QuantileRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# window_end_of() giờ import CHUNG từ schema.py — đây là công thức DUY NHẤT
# mà M2 và M4 cùng dùng để gộp cửa sổ 5 phút, không tự tính riêng nữa (xem
# giải thích ở docstring đầu file).
from schema import window_end_of


# ==============================================================================
# 1. KALMAN FILTER — import từ M3, có fallback để M4 không bị block
# ==============================================================================
try:
    from kalman_filter import KalmanFilter2D  # file thật của M3
    print("[INFO] Đã import KalmanFilter2D thật từ kalman_filter.py (M3)")
except ImportError:
    print(
        "[WARN] Chưa thấy kalman_filter.py của M3 trong thư mục hiện tại "
        "-> dùng fallback EMA (Exponential Moving Average) tạm thời.\n"
        "       Khi M3 xong, chỉ cần bỏ file kalman_filter.py vào cùng thư mục, "
        "       import thật sẽ tự động được dùng — KHÔNG cần sửa gì ở dưới."
    )

    class KalmanFilter2D:
        """
        FALLBACK TẠM THỜI (EMA đơn giản) — thay bằng bản thật của M3 khi có.
        Interface bắt buộc phải khớp: update(raw_lat, raw_lon) -> (lat, lon)
        """

        def __init__(self, alpha: float = 0.3):
            self.alpha = alpha
            self._lat = None
            self._lon = None

        def update(self, raw_lat: float, raw_lon: float):
            if self._lat is None:
                self._lat, self._lon = raw_lat, raw_lon
            else:
                self._lat = self.alpha * raw_lat + (1 - self.alpha) * self._lat
                self._lon = self.alpha * raw_lon + (1 - self.alpha) * self._lon
            return self._lat, self._lon


# ==============================================================================
# 1b. EMA SMOOTHER — BASELINE ĐỐI CHỨNG cho Kalman filter (LUÔN chạy, khác
#     với class KalmanFilter2D fallback ở trên — cái đó chỉ dùng TẠM khi
#     thiếu file của M3, còn EMASmoother ở đây là 1 baseline CÓ CHỦ ĐÍCH,
#     chạy song song với Kalman filter thật để so sánh, không phải để thay thế
#     khi thiếu file. Lý do chọn EMA làm baseline: đơn giản, 1 tham số (alpha),
#     là baseline chuẩn phổ biến khi đánh giá các thuật toán sensor fusion.
# ==============================================================================
class EMASmoother:
    """Baseline làm mượt GPS bằng Exponential Moving Average.
    Interface khớp KalmanFilter2D: update(raw_lat, raw_lon) -> (lat, lon)."""

    def __init__(self, alpha: float = 0.3):
        self.alpha = alpha
        self._lat = None
        self._lon = None

    def update(self, raw_lat: float, raw_lon: float):
        if self._lat is None:
            self._lat, self._lon = raw_lat, raw_lon
        else:
            self._lat = self.alpha * raw_lat + (1 - self.alpha) * self._lat
            self._lon = self.alpha * raw_lon + (1 - self.alpha) * self._lon
        return self._lat, self._lon


# ==============================================================================
# 2. CẤU HÌNH FEATURE / TARGET cho Quantile Regression (ETR)
# ==============================================================================
FEATURE_COLS = [
    "battery_level_pct",
    "wind_speed_ms",
    "motor_temp_c",
    "altitude_m",
    "network_signal_pct",
]
TARGET_COL = "remaining_flight_time_min"
LOWER_Q = 0.10   # cận dưới ETR ~ 10th percentile
UPPER_Q = 0.90   # cận trên ETR ~ 90th percentile
MIN_TRAIN_ROWS = 30


# ==============================================================================
# 3. LOAD DATA
# ==============================================================================
def load_fact_telemetry(conn: sqlite3.Connection) -> pd.DataFrame:
    df = pd.read_sql_query("SELECT * FROM fact_telemetry", conn)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, format="mixed")
    df = df.sort_values(["drone_id", "timestamp"]).reset_index(drop=True)
    return df


def load_existing_gold_windows(conn: sqlite3.Connection) -> set:
    """Lấy set các (drone_id, window_end) mà M2 đã INSERT sẵn.
    M4 CHỈ được UPDATE các dòng đã tồn tại này, không tự INSERT."""
    rows = conn.execute("SELECT drone_id, window_end FROM fact_gold_summary").fetchall()
    return set(rows)


# ==============================================================================
# 4. TRAIN QUANTILE REGRESSION cho remaining_flight_time_min
# ==============================================================================
def train_quantile_models(df: pd.DataFrame):
    train_df = df.dropna(subset=[TARGET_COL])
    if len(train_df) < MIN_TRAIN_ROWS:
        raise ValueError(
            f"Không đủ dữ liệu để train (chỉ có {len(train_df)} dòng có "
            f"remaining_flight_time_min, cần >= {MIN_TRAIN_ROWS}). "
            f"Chờ M2 tính thêm target hoặc chạy generate_mock_telemetry_row() nhiều hơn."
        )

    X = train_df[FEATURE_COLS]
    y = train_df[TARGET_COL]

    model_lower = GradientBoostingRegressor(
        loss="quantile", alpha=LOWER_Q, n_estimators=200, max_depth=3, random_state=42
    )
    model_upper = GradientBoostingRegressor(
        loss="quantile", alpha=UPPER_Q, n_estimators=200, max_depth=3, random_state=42
    )
    model_lower.fit(X, y)
    model_upper.fit(X, y)
    print(f"[INFO] Train xong Quantile Regression trên {len(train_df)} dòng.")
    return model_lower, model_upper


# ==============================================================================
# 4b. TRAIN LINEAR QUANTILE REGRESSION — BASELINE ĐỐI CHỨNG cho GBM ở trên.
#     Cùng feature, cùng target, cùng train_df -- chỉ khác thuật toán (tuyến
#     tính, không có tương tác/phi tuyến). Nếu GBM không tốt hơn baseline này
#     một cách có ý nghĩa thống kê thì độ phức tạp thêm của GBM chưa chắc đáng
#     giá -- đây chính là câu hỏi mà evaluate_metrics.py sẽ trả lời bằng số.
# ==============================================================================
def train_linear_quantile_baseline(df: pd.DataFrame):
    train_df = df.dropna(subset=[TARGET_COL])
    if len(train_df) < MIN_TRAIN_ROWS:
        raise ValueError(
            f"Không đủ dữ liệu để train baseline (chỉ có {len(train_df)} dòng, "
            f"cần >= {MIN_TRAIN_ROWS})."
        )

    X = train_df[FEATURE_COLS]
    y = train_df[TARGET_COL]

    # StandardScaler cần thiết vì QuantileRegressor dùng regularization (mặc
    # định alpha=1.0 kiểu Lasso) -- không scale thì feature có scale lớn (vd
    # network_signal_pct ~0-100) sẽ bị phạt khác feature scale nhỏ, không công
    # bằng giữa các feature.
    model_lower = make_pipeline(
        StandardScaler(), QuantileRegressor(quantile=LOWER_Q, alpha=0.0, solver="highs")
    )
    model_upper = make_pipeline(
        StandardScaler(), QuantileRegressor(quantile=UPPER_Q, alpha=0.0, solver="highs")
    )
    model_lower.fit(X, y)
    model_upper.fit(X, y)
    print(f"[INFO] Train xong Linear Quantile Regression (baseline) trên {len(train_df)} dòng.")
    return model_lower, model_upper


# ==============================================================================
# 5. KALMAN SMOOTHING + EMA (BASELINE) — chạy tuần tự theo từng drone
# ==============================================================================
def apply_kalman_smoothing(df: pd.DataFrame) -> pd.DataFrame:
    """Thêm 4 cột gps_lat_smooth/gps_lon_smooth (Kalman, mô hình CHÍNH) và
    gps_lat_ema/gps_lon_ema (EMA, BASELINE đối chứng) vào df, tính tuần tự
    theo từng drone_id, CÙNG 1 input gps_lat/gps_lon thô, để so sánh công
    bằng (Kalman filter và EMA cần chạy liên tục, không được xáo trộn thứ tự)."""
    smoothed_lat = np.empty(len(df))
    smoothed_lon = np.empty(len(df))
    ema_lat = np.empty(len(df))
    ema_lon = np.empty(len(df))

    for drone_id, group in df.groupby("drone_id", sort=False):
        kf = KalmanFilter2D()
        ema = EMASmoother()
        idx = group.index
        for i in idx:
            lat, lon = kf.update(df.at[i, "gps_lat"], df.at[i, "gps_lon"])
            e_lat, e_lon = ema.update(df.at[i, "gps_lat"], df.at[i, "gps_lon"])
            loc = df.index.get_loc(i)
            smoothed_lat[loc] = lat
            smoothed_lon[loc] = lon
            ema_lat[loc] = e_lat
            ema_lon[loc] = e_lon

    df = df.copy()
    df["gps_lat_smooth"] = smoothed_lat
    df["gps_lon_smooth"] = smoothed_lon
    df["gps_lat_ema"] = ema_lat
    df["gps_lon_ema"] = ema_lon
    return df


# ==============================================================================
# 6. GỘP CỬA SỔ 5 PHÚT — dùng window_end_of() CHUNG import từ schema.py
# ==============================================================================
# (Đã bỏ hàm window_end_of() tự định nghĩa riêng bằng pandas .floor() ở đây.
# Giờ dùng thẳng schema.window_end_of() — hàm này nhận datetime/str và trả về
# str ISO 8601 'YYYY-MM-DDTHH:MM:SSZ', đúng CHUNG công thức với M2.)


def build_window_aggregates(df: pd.DataFrame, window_minutes: int) -> pd.DataFrame:
    df = df.copy()
    # window_end giờ là STRING (cùng định dạng với cột window_end trong
    # fact_gold_summary), không còn là pd.Timestamp như bản cũ — nhờ vậy
    # so khớp UPDATE ở predict_and_update() không cần convert lại nữa.
    df["window_end"] = df["timestamp"].apply(lambda t: window_end_of(t, window_minutes))

    agg = (
        df.groupby(["drone_id", "window_end"])
        .agg(
            gps_lat_smooth=("gps_lat_smooth", "last"),   # vị trí tại cuối cửa sổ
            gps_lon_smooth=("gps_lon_smooth", "last"),
            gps_lat_ema=("gps_lat_ema", "last"),          # baseline, cùng mốc "last"
            gps_lon_ema=("gps_lon_ema", "last"),
            battery_level_pct=("battery_level_pct", "mean"),
            wind_speed_ms=("wind_speed_ms", "mean"),
            motor_temp_c=("motor_temp_c", "mean"),
            altitude_m=("altitude_m", "mean"),
            network_signal_pct=("network_signal_pct", "mean"),
        )
        .reset_index()
    )
    return agg


# ==============================================================================
# 7. PREDICT ETR + UPDATE fact_gold_summary
# ==============================================================================
def predict_and_update(
    conn: sqlite3.Connection,
    windows: pd.DataFrame,
    model_lower,
    model_upper,
    model_lower_baseline,
    model_upper_baseline,
    existing_gold_windows: set,
    dry_run: bool = False,
) -> None:
    X = windows[FEATURE_COLS]
    windows = windows.copy()
    windows["etr_lower_min"] = model_lower.predict(X)
    windows["etr_upper_min"] = model_upper.predict(X)
    windows["etr_lower_baseline"] = model_lower_baseline.predict(X)
    windows["etr_upper_baseline"] = model_upper_baseline.predict(X)

    # đảm bảo lower <= upper cho CẢ 2 mô hình (Quantile Regression không có
    # ràng buộc này mặc định, dù là GBM hay Linear)
    lo = np.minimum(windows["etr_lower_min"], windows["etr_upper_min"])
    hi = np.maximum(windows["etr_lower_min"], windows["etr_upper_min"])
    windows["etr_lower_min"], windows["etr_upper_min"] = lo, hi

    lo_b = np.minimum(windows["etr_lower_baseline"], windows["etr_upper_baseline"])
    hi_b = np.maximum(windows["etr_lower_baseline"], windows["etr_upper_baseline"])
    windows["etr_lower_baseline"], windows["etr_upper_baseline"] = lo_b, hi_b

    updated, skipped_no_gold_row = 0, 0
    now_iso = datetime.now(timezone.utc).isoformat()

    for _, row in windows.iterrows():
        # row["window_end"] đã là string ISO 8601 chuẩn (do window_end_of()
        # trả về), không cần .isoformat().replace(...) như bản cũ nữa.
        window_end_str = row["window_end"]
        key = (row["drone_id"], window_end_str)

        if key not in existing_gold_windows:
            # M4 không được tự INSERT — chỉ M2 mới tạo dòng trong fact_gold_summary.
            # Nếu M2 chưa kịp ghi dòng cho cửa sổ này thì bỏ qua, chạy lại sau.
            skipped_no_gold_row += 1
            continue

        if dry_run:
            print(
                f"[DRY-RUN] {key} -> gps=({row['gps_lat_smooth']:.6f}, "
                f"{row['gps_lon_smooth']:.6f}) etr_main=[{row['etr_lower_min']:.2f}, "
                f"{row['etr_upper_min']:.2f}] etr_baseline=[{row['etr_lower_baseline']:.2f}, "
                f"{row['etr_upper_baseline']:.2f}]"
            )
            continue

        conn.execute(
            """
            UPDATE fact_gold_summary
            SET gps_lat_smooth     = ?,
                gps_lon_smooth     = ?,
                etr_lower_min      = ?,
                etr_upper_min      = ?,
                gps_lat_ema        = ?,
                gps_lon_ema        = ?,
                etr_lower_baseline = ?,
                etr_upper_baseline = ?,
                updated_at         = ?
            WHERE drone_id = ? AND window_end = ?
            """,
            (
                float(row["gps_lat_smooth"]),
                float(row["gps_lon_smooth"]),
                float(row["etr_lower_min"]),
                float(row["etr_upper_min"]),
                float(row["gps_lat_ema"]),
                float(row["gps_lon_ema"]),
                float(row["etr_lower_baseline"]),
                float(row["etr_upper_baseline"]),
                now_iso,
                row["drone_id"],
                window_end_str,
            ),
        )
        updated += 1

    if not dry_run:
        conn.commit()

    print(f"[INFO] Đã UPDATE {updated} dòng trong fact_gold_summary (mô hình chính + baseline).")
    if skipped_no_gold_row:
        print(
            f"[WARN] Bỏ qua {skipped_no_gold_row} cửa sổ vì M2 chưa INSERT dòng "
            f"tương ứng trong fact_gold_summary (chạy lại sau khi M2 xong)."
        )


# ==============================================================================
# 8. MAIN
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="M4 — Kalman smoothing + ETR quantile regression")
    parser.add_argument("--db", default="drone_fleet.db", help="Đường dẫn file SQLite")
    parser.add_argument("--window-minutes", type=int, default=5, help="Độ dài cửa sổ tổng hợp (phút)")
    parser.add_argument("--dry-run", action="store_true", help="Chỉ in kết quả, không UPDATE vào DB")
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    conn.execute("PRAGMA busy_timeout = 5000;")  # SỬA: tránh "database is locked"
    try:
        print(f"[INFO] Đang đọc fact_telemetry từ {args.db} ...")
        df = load_fact_telemetry(conn)
        print(f"[INFO] Đọc được {len(df)} dòng.")

        print("[INFO] Đang train Quantile Regression (ETR) — mô hình chính (GBM) ...")
        model_lower, model_upper = train_quantile_models(df)

        print("[INFO] Đang train Linear Quantile Regression (ETR) — baseline đối chứng ...")
        model_lower_baseline, model_upper_baseline = train_linear_quantile_baseline(df)

        print("[INFO] Đang chạy Kalman smoothing (chính) + EMA (baseline) cho GPS ...")
        df_smoothed = apply_kalman_smoothing(df)

        print(f"[INFO] Đang gộp cửa sổ {args.window_minutes} phút ...")
        windows = build_window_aggregates(df_smoothed, args.window_minutes)
        print(f"[INFO] Có {len(windows)} cửa sổ (drone_id, window_end).")

        existing_gold_windows = load_existing_gold_windows(conn)
        print(f"[INFO] fact_gold_summary hiện có {len(existing_gold_windows)} dòng do M2 tạo.")

        predict_and_update(
            conn, windows,
            model_lower, model_upper,
            model_lower_baseline, model_upper_baseline,
            existing_gold_windows, dry_run=args.dry_run,
        )
    finally:
        conn.close()


if __name__ == "__main__":
    main()
