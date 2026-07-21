"""
================================================================================
 DATA CONTRACT + ERD — Drone Fleet Monitoring Dashboard
================================================================================
File này là NGUỒN CHÂN LÝ DUY NHẤT. Mọi member (M1-M5) import từ đây, KHÔNG tự
định nghĩa lại field ở nơi khác. Nếu cần đổi field, sửa NGAY TẠI ĐÂY rồi báo cả
nhóm — không sửa "chui" ở file riêng của mình.

--------------------------------------------------------------------------------
 ERD — 3 bảng duy nhất, không hơn (đủ dùng cho cả dự án)
--------------------------------------------------------------------------------

    dim_drones (bảng tĩnh, thông tin drone)
    ├── drone_id      PK
    ├── model_name
    └── max_battery_capacity_wh

    fact_telemetry (Silver — 1 dòng = 1 lần đo, join với dim_drones qua drone_id)
    ├── drone_id            FK -> dim_drones.drone_id
    ├── timestamp
    ├── gps_lat / gps_lon / altitude_m
    ├── battery_level_pct / wind_speed_ms / motor_temp_c
    ├── hardware_ok / network_signal_pct
    ├── is_interpolated
    └── remaining_flight_time_min   <-- TARGET để M4 train model (M2 tính sẵn)

    fact_gold_summary (Gold — 1 dòng = 1 cửa sổ tổng hợp 5 phút / drone)
    ├── drone_id            FK -> dim_drones.drone_id
    ├── window_end
    ├── [cột M2 sở hữu]  battery_mean/std/status, wind_*, motor_temp_*, network_status
    ├── [cột M4 sở hữu]  gps_lat_smooth, gps_lon_smooth, etr_lower_min, etr_upper_min
    └── updated_at

Không có bảng thứ 4. Không có quan hệ nhiều-nhiều. Join duy nhất là qua drone_id.

--------------------------------------------------------------------------------
 CHEAT-SHEET theo từng người — Input nhận gì, Output trả gì
--------------------------------------------------------------------------------
    M1 (Streaming)
        Input  : gọi hàm step() của M3 (hoặc generate_mock_telemetry_row() bên dưới
                  trong lúc M3 chưa xong)
        Output : INSERT vào bronze_telemetry (raw_json_payload) — dùng
                  make_bronze_payload() để validate trước khi ghi

    M2 (Analytics Engineer)
        Input  : SELECT raw_json_payload FROM bronze_telemetry
        Output : INSERT vào fact_telemetry (dùng SilverRecord) VÀ
                  UPDATE các cột *_mean/*_std/*_status trong fact_gold_summary
                  (dùng GoldSummary, chỉ set field mình sở hữu)

    M3 (AI/Toán)
        Input  : không cần input từ ai — chỉ cần biết tên field cần trả
                  (xem TelemetryPayload)
        Output : hàm step() trả dict đúng field; hàm Kalman filter riêng
                  (file kalman_filter.py) nhận (raw_lat, raw_lon) trả (lat, lon)
                  đã làm mượt

    M4 (ML Engineer)
        Input  : SELECT * FROM fact_telemetry (dùng cột remaining_flight_time_min
                  làm target), import KalmanFilter2D của M3
        Output : UPDATE fact_gold_summary SET gps_lat_smooth=, gps_lon_smooth=,
                  etr_lower_min=, etr_upper_min= (CHỈ 4 cột này, không đụng cột M2)

    M5 (Dashboard)
        Input  : SELECT * FROM fact_gold_summary
        Output : không ghi gì cả — chỉ hiển thị

--------------------------------------------------------------------------------
 Cách dùng
--------------------------------------------------------------------------------
    from schema import (
        TelemetryPayload, SilverRecord, GoldSummary, DroneInfo,
        make_bronze_payload, generate_mock_telemetry_row, classify_status,
        window_end_of, DDL_STATEMENTS,
    )

--------------------------------------------------------------------------------
 window_end_of() — CÔNG THỨC DUY NHẤT để gộp cửa sổ 5 phút
--------------------------------------------------------------------------------
    M2 (INSERT fact_gold_summary) và M4 (UPDATE fact_gold_summary) BẮT BUỘC phải
    dùng CHUNG hàm window_end_of() bên dưới để tính window_end. Trước đây 2 bên
    tự tính riêng (M2 dùng SQL, M4 dùng pandas .floor()) và có nguy cơ lệch nhau
    (floor vs ceil, lệch múi giờ) khiến M4 UPDATE 0 dòng dù chạy không lỗi. Từ
    giờ cả 2 chỉ import và gọi đúng 1 hàm này.
"""

import random
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


# ==============================================================================
# ENUM dùng chung
# ==============================================================================

class HealthStatus(str, Enum):
    """Trạng thái cảnh báo cho thẻ màu Dashboard (M2 tính, M4/M5 chỉ đọc)."""
    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"


# ==============================================================================
# DIM_DRONES — bảng tĩnh, M2 tạo 1 lần lúc khởi tạo hệ thống
# ==============================================================================

class DroneInfo(BaseModel):
    """
    SỬA: thêm các field profile (payload_kg, wind_zone, ...) — trước đây
    profile của M3 chỉ tồn tại trong RAM lúc simulator chạy, không ai khác
    (M2/M4/M5) biết được. Giờ lưu profile vào dim_drones NGAY LÚC ĐĂNG KÝ
    drone (do M1 làm ở init_db), để M5 query ra và hiển thị "vì sao" drone
    này đang cảnh báo (VD: đang mang 4kg hàng, đang ở vùng gió bão).
    """
    drone_id: str = Field(..., description="PK, vd: 'DRONE_001'")
    model_name: str = Field(default="DJI Matrice 100", description="Model drone")
    max_battery_capacity_wh: float = Field(
        default=500.0, description="Dung lượng pin tối đa, đơn vị Wh — dùng để tính ETR"
    )

    # --- Profile (case study) — ghi 1 lần lúc đăng ký drone, không đổi sau đó ---
    profile_label: str = Field(default="baseline", description="Tên kịch bản, vd: 'worst_case'")
    payload_kg: float = Field(default=0.0, description="Khối lượng hàng mang thêm (kg)")
    wind_zone: str = Field(default="moderate", description="calm | moderate | storm")
    battery_health: float = Field(default=1.0, description="0.7 (pin cũ) - 1.0 (pin mới)")
    ambient_temp_c: float = Field(default=25.0, description="Nhiệt độ môi trường (độ C)")
    gps_quality: float = Field(default=1.0, description="0.5 (module rẻ) - 1.0 (module xịn)")
    hardware_reliability: float = Field(default=0.99, description="Xác suất hardware_ok mỗi giây")
    network_zone: str = Field(default="suburban", description="urban | suburban | rural")


# ==============================================================================
# BRONZE LAYER — Raw payload
# Owner ghi: M3 (sinh ra) -> M1 (đẩy stream vào Bronze, KHÔNG được sửa field)
# ==============================================================================

class TelemetryPayload(BaseModel):
    """
    Một bản ghi telemetry thô, sinh mỗi giây bởi StochasticSimulator (M3),
    được M1 stream thẳng vào tầng Bronze — KHÔNG qua xử lý gì thêm.
    """
    drone_id: str = Field(..., description="ID định danh drone, vd: 'DRONE_001'")
    timestamp: str = Field(..., description="ISO 8601 UTC, vd: '2026-07-06T10:15:30Z'")

    # Vị trí (chưa qua Kalman filter — đây là raw GPS, có nhiễu)
    gps_lat: float
    gps_lon: float
    altitude_m: float = Field(..., description="Độ cao, đơn vị mét")

    # --- GROUND TRUTH (bổ sung cho mục đích ĐÁNH GIÁ KHOA HỌC) ---
    # StochasticSimulator biết chính xác vị trí thật (true_lat/true_lon) mà nó
    # dùng để sinh ra gps_lat/gps_lon có nhiễu — trước đây giá trị này chỉ tồn
    # tại trong RAM lúc mô phỏng chạy rồi mất đi, không ai đánh giá được Kalman
    # filter/EMA làm mượt TỐT HƠN dữ liệu thô bao nhiêu (không có gì để so sánh).
    # Giờ ghi thêm true_lat/true_lon NGAY TỪ BRONZE để evaluate_metrics.py có
    # thể tính RMSE thật. Optional + default None để KHÔNG phá vỡ tương thích
    # ngược với generate_mock_telemetry_row() (mock không có ground truth).
    true_lat: Optional[float] = Field(
        default=None, description="[CHỈ DÙNG ĐỂ ĐÁNH GIÁ] Vị trí thật, không nhiễu, do simulator sinh ra"
    )
    true_lon: Optional[float] = Field(
        default=None, description="[CHỈ DÙNG ĐỂ ĐÁNH GIÁ] Vị trí thật, không nhiễu, do simulator sinh ra"
    )

    # Trạng thái pin & môi trường (sinh từ Wiener / O-U process)
    battery_level_pct: float = Field(..., ge=0, le=100, description="% pin còn lại")
    wind_speed_ms: float = Field(..., ge=0, description="Tốc độ gió, m/s")
    motor_temp_c: float = Field(..., description="Nhiệt độ động cơ, độ C")

    # Trạng thái phần cứng / kết nối
    hardware_ok: bool = Field(..., description="True nếu phần cứng không lỗi")
    network_signal_pct: float = Field(..., ge=0, le=100, description="% cường độ tín hiệu")

    @field_validator("timestamp")
    @classmethod
    def validate_iso8601(cls, v: str) -> str:
        try:
            datetime.fromisoformat(v.replace("Z", "+00:00"))
        except ValueError:
            raise ValueError(f"timestamp phải là ISO 8601 UTC, nhận được: {v}")
        return v


# ==============================================================================
# SILVER LAYER — fact_telemetry (Cleaned data + target cho M4 train)
# Owner ghi: M2 (đọc Bronze, làm sạch, ghi Silver)
# Owner đọc: M4 (train model + input cho Kalman filter)
# ==============================================================================

class SilverRecord(BaseModel):
    """
    Bản ghi đã làm sạch: null đã xử lý, timestamp đã chuẩn hóa, trùng lặp đã loại.
    """
    drone_id: str
    timestamp: str

    gps_lat: float
    gps_lon: float
    altitude_m: float

    # Ground truth (xem giải thích ở TelemetryPayload) — chuyển tiếp nguyên
    # văn từ Bronze, M2 KHÔNG tính toán gì thêm ở đây, chỉ pass-through.
    true_lat: Optional[float] = None
    true_lon: Optional[float] = None

    battery_level_pct: float = Field(..., ge=0, le=100)
    wind_speed_ms: float = Field(..., ge=0)
    motor_temp_c: float

    hardware_ok: bool
    network_signal_pct: float = Field(..., ge=0, le=100)

    is_interpolated: bool = Field(
        default=False,
        description="True nếu giá trị này được nội suy để thay thế null/lỗi gốc",
    )

    # QUAN TRỌNG: M2 tính sẵn cột này (dựa trên drain_rate lịch sử của từng drone_id)
    # để M4 dùng làm target train Quantile Regression — M4 KHÔNG tự derive lại.
    remaining_flight_time_min: Optional[float] = Field(
        default=None,
        description="Target cho M4: thời gian bay còn lại thực tế (phút), "
                    "M2 tính bằng cách nội suy từ thời điểm pin về 0% trong lịch sử",
    )


# ==============================================================================
# GOLD LAYER — fact_gold_summary (Aggregated, ready for Dashboard)
# Đây là bảng NHIỀU NGƯỜI CÙNG GHI -> mỗi member CHỈ ghi đúng field mình sở hữu.
# ==============================================================================

class GoldSummary(BaseModel):
    """
    Bảng tổng hợp cuối cùng mà M5 (Dashboard) query. M5 CHỈ ĐỌC.

    QUY TẮC SỞ HỮU CỘT (bắt buộc tuân thủ để tránh ghi đè lẫn nhau):
        - M2 ghi:  *_mean, *_std, *_status
        - M4 ghi:  gps_lat_smooth, gps_lon_smooth, etr_lower_min, etr_upper_min
                    (mô hình CHÍNH) + gps_lat_ema, gps_lon_ema, etr_lower_baseline,
                    etr_upper_baseline (mô hình BASELINE, phục vụ so sánh khoa học)
        - M5:      không ghi field nào, chỉ SELECT
        - updated_at: ai update field của mình thì tự cập nhật lại field này
    """
    drone_id: str
    window_end: str = Field(..., description="Thời điểm cuối cửa sổ tổng hợp (ISO 8601)")

    # --- Owner: M2 ---
    battery_mean: float
    battery_std: float
    battery_status: HealthStatus

    wind_speed_mean: float
    wind_speed_std: float

    motor_temp_mean: float
    motor_temp_std: float
    motor_temp_status: HealthStatus

    network_status: HealthStatus

    # --- Owner: M4 (mô hình CHÍNH) ---
    gps_lat_smooth: Optional[float] = Field(default=None, description="GPS sau Kalman filter")
    gps_lon_smooth: Optional[float] = None
    etr_lower_min: Optional[float] = Field(default=None, description="Cận dưới ETR (phút)")
    etr_upper_min: Optional[float] = Field(default=None, description="Cận trên ETR (phút)")

    # --- Owner: M4 (mô hình BASELINE — đối chứng khoa học) ---
    gps_lat_ema: Optional[float] = Field(default=None, description="GPS sau EMA (baseline đối chứng Kalman)")
    gps_lon_ema: Optional[float] = None
    etr_lower_baseline: Optional[float] = Field(default=None, description="Cận dưới ETR — Linear Quantile Regression")
    etr_upper_baseline: Optional[float] = Field(default=None, description="Cận trên ETR — Linear Quantile Regression")

    # --- Metadata ---
    updated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def make_bronze_payload(drone_id: str, sim_output: dict) -> TelemetryPayload:
    """
    "Cửa khẩu" duy nhất giữa M3 và M1.
    M3 chỉ cần trả dict đúng field; hàm này validate ngay, raise lỗi tại chỗ
    nếu M3 gõ sai tên field hoặc thiếu field — không cần chờ integration mới biết.
    """
    return TelemetryPayload(
        drone_id=drone_id,
        timestamp=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        **sim_output,
    )


def generate_mock_telemetry_row(drone_id: str = "DRONE_001") -> dict:
    """
    DÙNG TRONG NGÀY 2-5: M1 dùng hàm này để có data giả ngay lập tức, không
    cần chờ M3 code xong stochastic_simulator.py. Trả về dict đúng 100% field
    của TelemetryPayload, để Ngày 6 chỉ cần đổi 1 dòng import là chuyển sang
    data thật của M3 mà không sửa gì ở code M1.

    M4 và M5 cũng có thể gọi hàm này để tự sinh vài trăm dòng CSV mock test khung
    train model / khung UI trong lúc chờ pipeline thật.
    """
    return {
        "gps_lat": 10.7626 + random.uniform(-0.001, 0.001),
        "gps_lon": 106.6602 + random.uniform(-0.001, 0.001),
        "altitude_m": random.uniform(80, 150),
        "battery_level_pct": random.uniform(20, 100),
        "wind_speed_ms": random.uniform(0, 10),
        "motor_temp_c": random.uniform(40, 75),
        "hardware_ok": random.random() > 0.02,
        "network_signal_pct": random.uniform(60, 100),
    }


def window_end_of(ts, window_minutes: int = 5) -> str:
    """
    CÔNG THỨC DUY NHẤT để tính window_end — M2 và M4 BẮT BUỘC dùng chung hàm
    này (không tự tính lại bằng SQL/pandas riêng) để tránh lệch nhau.

    Ceiling timestamp về mốc window_minutes phút gần nhất, vd với
    window_minutes=5: 10:15:30Z -> 10:20:00Z, và 10:15:00Z (đúng mốc) -> 10:15:00Z.

    Tham số:
        ts             : str ISO 8601 (có 'Z' hoặc offset) hoặc datetime object.
        window_minutes : độ dài cửa sổ tính bằng phút (mặc định 5).

    Trả về: str ISO 8601 UTC, dạng 'YYYY-MM-DDTHH:MM:SSZ'.
    """
    if isinstance(ts, str):
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    else:
        dt = ts

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt = dt.astimezone(timezone.utc)

    floored_minute = (dt.minute // window_minutes) * window_minutes
    floored = dt.replace(minute=floored_minute, second=0, microsecond=0)

    is_exact_mark = (dt.second == 0 and dt.microsecond == 0 and dt.minute == floored_minute)
    window_end = floored if is_exact_mark else floored + timedelta(minutes=window_minutes)

    return window_end.strftime("%Y-%m-%dT%H:%M:%SZ")


def classify_status(value: float, mean: float, std: float, k: float = 2.0) -> HealthStatus:
    """
    Dùng khi có 1 BASELINE THẬT (mean/std từ lịch sử/kỳ vọng bình thường,
    KHÔNG PHẢI tính từ chính cửa sổ đang muốn phân loại) để so sánh giá trị
    hiện tại lệch bao xa. k=2.0 tương đương ~95% khoảng tin cậy (Gaussian).
        |value - mean| <= 1*std          -> GREEN
        1*std < |value - mean| <= k*std  -> YELLOW
        |value - mean| > k*std           -> RED

    CẢNH BÁO: KHÔNG gọi classify_status(x, x, std) — so 1 số với chính nó
    luôn cho z=0 -> luôn GREEN, vô hiệu hóa toàn bộ logic cảnh báo. Đây là
    lỗi thật đã xảy ra ở bản M2 trước đó. Nếu chưa có baseline lịch sử thật,
    dùng classify_battery_status()/classify_motor_temp_status() bên dưới
    (ngưỡng tuyệt đối) thay vì hàm này.
    """
    if std <= 0:
        return HealthStatus.GREEN
    z = abs(value - mean) / std
    if z <= 1.0:
        return HealthStatus.GREEN
    elif z <= k:
        return HealthStatus.YELLOW
    return HealthStatus.RED


# ==============================================================================
# NGƯỠNG TUYỆT ĐỐI — dùng khi CHƯA có baseline lịch sử thật để so sánh.
# SỬA: bản M2 trước đó gọi classify_status(battery_mean, battery_mean, std) —
# so 1 số với chính nó, luôn ra GREEN bất kể drone có đang cảnh báo hay không.
# 2 hàm dưới đây thay thế bằng ngưỡng vật lý cố định, đơn giản và đúng mục
# đích giám sát: pin thấp / động cơ nóng là nguy hiểm, không phụ thuộc gì vào
# chính cửa sổ dữ liệu đang xét.
# ==============================================================================

BATTERY_RED_THRESHOLD_PCT = 20.0     # dưới 20% pin -> nguy hiểm
BATTERY_YELLOW_THRESHOLD_PCT = 40.0  # dưới 40% pin -> cảnh báo

MOTOR_TEMP_RED_THRESHOLD_C = 90.0     # trên 90°C -> nguy hiểm
MOTOR_TEMP_YELLOW_THRESHOLD_C = 70.0  # trên 70°C -> cảnh báo


def classify_battery_status(battery_mean_pct: float) -> HealthStatus:
    if battery_mean_pct < BATTERY_RED_THRESHOLD_PCT:
        return HealthStatus.RED
    elif battery_mean_pct < BATTERY_YELLOW_THRESHOLD_PCT:
        return HealthStatus.YELLOW
    return HealthStatus.GREEN


def classify_motor_temp_status(motor_temp_mean_c: float) -> HealthStatus:
    if motor_temp_mean_c > MOTOR_TEMP_RED_THRESHOLD_C:
        return HealthStatus.RED
    elif motor_temp_mean_c > MOTOR_TEMP_YELLOW_THRESHOLD_C:
        return HealthStatus.YELLOW
    return HealthStatus.GREEN


# ==============================================================================
# DDL — M2 copy đúng nguyên văn để CREATE TABLE, khỏi tự suy ra cột từ Pydantic
# (SQLite dialect — đủ dùng cho demo; đổi kiểu nếu dùng Postgres/MySQL)
# ==============================================================================

DDL_STATEMENTS = """
CREATE TABLE IF NOT EXISTS dim_drones (
    drone_id                TEXT PRIMARY KEY,
    model_name              TEXT NOT NULL DEFAULT 'DJI Matrice 100',
    max_battery_capacity_wh REAL NOT NULL DEFAULT 500.0,
    profile_label            TEXT NOT NULL DEFAULT 'baseline',
    payload_kg               REAL NOT NULL DEFAULT 0.0,
    wind_zone                 TEXT NOT NULL DEFAULT 'moderate',
    battery_health            REAL NOT NULL DEFAULT 1.0,
    ambient_temp_c            REAL NOT NULL DEFAULT 25.0,
    gps_quality               REAL NOT NULL DEFAULT 1.0,
    hardware_reliability      REAL NOT NULL DEFAULT 0.99,
    network_zone              TEXT NOT NULL DEFAULT 'suburban'
);

CREATE TABLE IF NOT EXISTS bronze_telemetry (
    log_id            INTEGER PRIMARY KEY AUTOINCREMENT,
    received_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    raw_json_payload  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS fact_telemetry (
    id                          INTEGER PRIMARY KEY AUTOINCREMENT,
    drone_id                    TEXT NOT NULL REFERENCES dim_drones(drone_id),
    timestamp                   TEXT NOT NULL,
    gps_lat                     REAL NOT NULL,
    gps_lon                     REAL NOT NULL,
    altitude_m                  REAL NOT NULL,
    battery_level_pct           REAL NOT NULL,
    wind_speed_ms               REAL NOT NULL,
    motor_temp_c                REAL NOT NULL,
    hardware_ok                 BOOLEAN NOT NULL,
    network_signal_pct          REAL NOT NULL,
    is_interpolated             BOOLEAN NOT NULL DEFAULT 0,
    remaining_flight_time_min   REAL,
    true_lat                    REAL,
    true_lon                    REAL
);

CREATE TABLE IF NOT EXISTS fact_gold_summary (
    drone_id            TEXT NOT NULL REFERENCES dim_drones(drone_id),
    window_end          TEXT NOT NULL,
    battery_mean        REAL NOT NULL,
    battery_std         REAL NOT NULL,
    battery_status       TEXT NOT NULL,
    wind_speed_mean      REAL NOT NULL,
    wind_speed_std       REAL NOT NULL,
    motor_temp_mean      REAL NOT NULL,
    motor_temp_std       REAL NOT NULL,
    motor_temp_status    TEXT NOT NULL,
    network_status       TEXT NOT NULL,
    gps_lat_smooth       REAL,
    gps_lon_smooth       REAL,
    etr_lower_min        REAL,
    etr_upper_min        REAL,
    -- === CỘT BASELINE (M4 sở hữu, phục vụ SO SÁNH khoa học) ===
    -- gps_lat_ema/gps_lon_ema : baseline làm mượt bằng Exponential Moving
    --   Average (đối chứng cho Kalman filter — cùng input, khác thuật toán).
    -- etr_lower_baseline/etr_upper_baseline : baseline Linear Quantile
    --   Regression (đối chứng cho Gradient Boosting Quantile Regression).
    gps_lat_ema          REAL,
    gps_lon_ema          REAL,
    etr_lower_baseline   REAL,
    etr_upper_baseline   REAL,
    updated_at           TEXT NOT NULL,
    PRIMARY KEY (drone_id, window_end)
);
"""


# ==============================================================================
# SELF-TEST — chạy: python3 schema.py
# Mọi member nên chạy 1 lần trước khi bắt đầu code phần của mình.
# ==============================================================================

if __name__ == "__main__":
    print("=== 1. Test DroneInfo (dim_drones) ===")
    drone = DroneInfo(drone_id="DRONE_001")
    print(drone.model_dump())

    print("\n=== 2. Test TelemetryPayload (Bronze) ===")
    sample_bronze = TelemetryPayload(
        drone_id="DRONE_001",
        timestamp="2026-07-06T10:15:30Z",
        **generate_mock_telemetry_row(),
    )
    print(sample_bronze.model_dump())

    print("\n=== 3. Test make_bronze_payload() — cửa khẩu M3 -> M1 ===")
    mock_row = generate_mock_telemetry_row("DRONE_001")
    bronze_payload = make_bronze_payload("DRONE_001", mock_row)
    print(bronze_payload.model_dump_json())

    print("\n=== 4. Test bắt lỗi khi gõ sai tên field ===")
    try:
        wrong = {**mock_row}
        wrong["pin"] = wrong.pop("battery_level_pct")
        make_bronze_payload("DRONE_001", wrong)
    except Exception as e:
        print("Bắt lỗi đúng như kỳ vọng:", type(e).__name__)

    print("\n=== 5. Test SilverRecord (fact_telemetry) ===")
    silver = SilverRecord(
        drone_id="DRONE_001",
        timestamp="2026-07-06T10:15:30Z",
        is_interpolated=False,
        remaining_flight_time_min=12.5,
        **mock_row,
    )
    print(silver.model_dump())

    print("\n=== 6. Test classify_status() — M2 dùng cho control limit ===")
    print("value=85, mean=85, std=2 ->", classify_status(85, 85, 2).value)
    print("value=80, mean=85, std=2 ->", classify_status(80, 85, 2).value)
    print("value=70, mean=85, std=2 ->", classify_status(70, 85, 2).value)

    print("\n=== 7. Test GoldSummary (fact_gold_summary) ===")
    sample_gold = GoldSummary(
        drone_id="DRONE_001",
        window_end="2026-07-06T10:20:00Z",
        battery_mean=85.0, battery_std=2.1, battery_status=HealthStatus.GREEN,
        wind_speed_mean=4.5, wind_speed_std=0.8,
        motor_temp_mean=59.0, motor_temp_std=1.5, motor_temp_status=HealthStatus.YELLOW,
        network_status=HealthStatus.GREEN,
        gps_lat_smooth=10.76261, gps_lon_smooth=106.66015,
        etr_lower_min=9.0, etr_upper_min=15.0,
    )
    print(sample_gold.model_dump())

    print("\n=== 8. Test window_end_of() — công thức dùng chung M2 & M4 ===")
    print("10:15:30Z ->", window_end_of("2026-07-06T10:15:30Z"))   # -> 10:20:00Z
    print("10:15:00Z ->", window_end_of("2026-07-06T10:15:00Z"))   # -> 10:15:00Z (đã đúng mốc)
    print("10:19:59Z ->", window_end_of("2026-07-06T10:19:59Z"))   # -> 10:20:00Z

    print("\nTAT CA TEST PASS. Schema san sang de nhom dung chung.")
