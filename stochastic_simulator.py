"""
MEMBER 3 — StochasticSimulator (bản ADVANCED — có profile theo từng drone)
================================================================================
So với bản trước, mỗi drone giờ có 1 "hồ sơ" (DroneProfile) riêng — nghĩa là
2 drone không chỉ khác nhau vì random seed, mà khác nhau CÓ HỆ THỐNG vì đang
ở trong 1 kịch bản (case study) khác nhau: mang hàng nặng, bay vào vùng gió
bão, pin đã xuống cấp, GPS module rẻ tiền, v.v.

7 YẾU TỐ trong DroneProfile và tác động vật lý của từng cái:

    payload_kg          : Khối lượng hàng mang thêm (kg)
                            -> pin tụt NHANH hơn, động cơ NÓNG hơn (phải tải nặng)
    wind_zone            : Vùng bay ("calm" / "moderate" / "storm")
                            -> quyết định mu, sigma của O-U process cho gió
    battery_health        : Độ "khỏe" của pin (0.7 - 1.0, pin cũ thì thấp)
                            -> pin cũ tụt nhanh hơn VÀ dao động thất thường hơn
    ambient_temp_c        : Nhiệt độ môi trường (độ C, mô phỏng thời tiết)
                            -> trời lạnh làm pin Lithium tụt nhanh hơn (hiệu ứng
                               thật của pin Li-ion); trời nóng làm động cơ nóng hơn
    gps_quality           : Chất lượng module GPS (0.5 - 1.0)
                            -> module rẻ tiền = nhiễu cảm biến GPS lớn hơn
    hardware_reliability   : Độ tin cậy phần cứng (0.9 - 0.999)
                            -> xác suất hardware_ok=True mỗi giây
    network_zone          : Vùng phủ sóng ("urban" / "suburban" / "rural")
                            -> quyết định mu, sigma của tín hiệu mạng

Bản DEFAULT (không truyền profile) giữ NGUYÊN 100% hành vi của bản cũ — đảm
bảo code cũ (m1_streaming.py bản trước, test_simulator.py) vẫn chạy đúng y hệt
nếu chưa truyền profile.
"""

import math
import random
from dataclasses import dataclass, field, asdict


# ==============================================================================
# DRONE PROFILE — "hồ sơ" case study cho từng drone
# ==============================================================================

@dataclass
class DroneProfile:
    payload_kg: float = 0.0
    wind_zone: str = "moderate"          # "calm" | "moderate" | "storm"
    battery_health: float = 1.0          # 0.7 (pin cũ) -> 1.0 (pin mới)
    ambient_temp_c: float = 25.0         # nhiệt độ môi trường
    gps_quality: float = 1.0             # 0.5 (module rẻ) -> 1.0 (module xịn)
    hardware_reliability: float = 0.99   # xác suất hardware_ok mỗi giây
    network_zone: str = "suburban"        # "urban" | "suburban" | "rural"
    label: str = "baseline"              # tên kịch bản, chỉ để log/debug

    def as_dict(self) -> dict:
        return asdict(self)


# Tham số (mu, sigma) cho O-U process của gió, theo từng vùng bay.
# "moderate" giữ NGUYÊN giá trị cũ (mu=4.0, sigma=0.5) để không đổi hành vi
# mặc định khi chưa cấu hình profile.
WIND_ZONE_PARAMS = {
    "calm":     (2.0, 0.3),
    "moderate": (4.0, 0.5),
    "storm":    (9.0, 1.5),
}

# Tham số (mu, sigma) cho tín hiệu mạng, theo vùng phủ sóng.
# "suburban" giữ NGUYÊN giá trị cũ (mu=92.0, sigma=3.0).
NETWORK_ZONE_PARAMS = {
    "urban":    (97.0, 1.5),
    "suburban": (92.0, 3.0),
    "rural":    (75.0, 6.0),
}


# ==============================================================================
# MỘT SỐ KỊCH BẢN (CASE STUDY) MẪU — dùng để gán cho fleet trong m1_streaming.py
# ==============================================================================

CASE_STUDY_PROFILES: list[DroneProfile] = [
    DroneProfile(label="baseline"),
    DroneProfile(label="heavy_payload", payload_kg=4.0),
    DroneProfile(label="storm_zone", wind_zone="storm"),
    DroneProfile(label="heavy_in_storm", payload_kg=4.0, wind_zone="storm"),
    DroneProfile(label="aging_battery", battery_health=0.75),
    DroneProfile(label="cold_weather", ambient_temp_c=5.0),
    DroneProfile(label="poor_gps_rural", gps_quality=0.6, network_zone="rural"),
    DroneProfile(label="worst_case", payload_kg=3.0, wind_zone="storm",
                 battery_health=0.8, ambient_temp_c=8.0, gps_quality=0.7,
                 hardware_reliability=0.95, network_zone="rural"),
]


class StochasticSimulator:
    def __init__(self, drone_id: str = "DRONE_001", seed: int | None = None,
                 profile: DroneProfile | None = None):
        self.drone_id = drone_id
        self.profile = profile or DroneProfile()  # mặc định = hành vi bản cũ
        if seed is not None:
            random.seed(seed)

        # --- Trạng thái nội bộ (state) của các quá trình ngẫu nhiên ---
        self.battery = 100.0
        wind_mu, _ = WIND_ZONE_PARAMS[self.profile.wind_zone]
        self.wind = wind_mu             # khởi tạo đúng bằng mu của vùng bay
        self.motor_temp = 45.0
        network_mu, _ = NETWORK_ZONE_PARAMS[self.profile.network_zone]
        self.network_signal = network_mu

        self.true_lat = 10.9447
        self.true_lon = 106.8243
        self.true_altitude = 100.0
        self.heading = random.uniform(0, 2 * math.pi)

    # ------------------------------------------------------------------
    # Wiener process: battery giảm dần + nhiễu tích lũy (drift âm)
    # CHỊU ẢNH HƯỞNG của: payload_kg, battery_health, ambient_temp_c
    # ------------------------------------------------------------------
    def _step_battery(self) -> float:
        base_drift = 0.03
        base_sigma = 0.015

        # payload_kg: mỗi kg hàng mang thêm làm tăng tiêu hao ~15%
        payload_factor = 1.0 + self.profile.payload_kg * 0.15

        # battery_health: pin cũ (health thấp) vừa tụt nhanh hơn, vừa dao
        # động thất thường hơn (nghịch đảo của health)
        health_factor = 1.0 / max(self.profile.battery_health, 0.1)

        # ambient_temp_c: pin Li-ion tụt nhanh hơn khi trời lạnh (hiệu ứng
        # vật lý thật) -- dưới 15 độ C, mỗi độ lạnh thêm làm tăng tiêu hao 2%
        cold_penalty = max(0.0, 15.0 - self.profile.ambient_temp_c) * 0.02
        ambient_factor = 1.0 + cold_penalty

        drift = base_drift * payload_factor * health_factor * ambient_factor
        sigma = base_sigma * health_factor

        d_battery = -drift + sigma * random.gauss(0, 1)
        self.battery = max(0.0, min(100.0, self.battery + d_battery))
        return self.battery

    # ------------------------------------------------------------------
    # Ornstein-Uhlenbeck: dX = theta * (mu - X) * dt + sigma * dW
    # ------------------------------------------------------------------
    def _step_ou(self, x: float, mu: float, theta: float, sigma: float,
                 lower: float = 0.0, upper: float | None = None) -> float:
        dt = 1.0
        dx = theta * (mu - x) * dt + sigma * random.gauss(0, 1)
        new_x = x + dx
        new_x = max(lower, new_x)
        if upper is not None:
            new_x = min(upper, new_x)
        return new_x

    # CHỊU ẢNH HƯỞNG của: wind_zone
    def _step_wind(self) -> float:
        mu, sigma = WIND_ZONE_PARAMS[self.profile.wind_zone]
        self.wind = self._step_ou(self.wind, mu=mu, theta=0.3, sigma=sigma, lower=0.0)
        return self.wind

    # CHỊU ẢNH HƯỞNG của: wind (gián tiếp qua wind_zone), payload_kg, ambient_temp_c
    def _step_motor_temp(self) -> float:
        base_mu = 45.0 + 1.5 * self.wind
        # Mang nặng hơn -> động cơ phải tải nhiều hơn -> nóng hơn
        payload_heat = self.profile.payload_kg * 2.0
        # Trời nóng hơn baseline (25C) -> động cơ tản nhiệt kém hơn -> nóng hơn
        ambient_offset = (self.profile.ambient_temp_c - 25.0) * 0.3

        target_mu = base_mu + payload_heat + ambient_offset
        self.motor_temp = self._step_ou(
            self.motor_temp, mu=target_mu, theta=0.2, sigma=0.8, lower=20.0, upper=130.0
        )
        return self.motor_temp

    # CHỊU ẢNH HƯỞNG của: network_zone
    def _step_network(self) -> float:
        mu, sigma = NETWORK_ZONE_PARAMS[self.profile.network_zone]
        self.network_signal = self._step_ou(
            self.network_signal, mu=mu, theta=0.25, sigma=sigma, lower=0.0, upper=100.0
        )
        return self.network_signal

    # ------------------------------------------------------------------
    # GPS: random walk có quán tính hướng bay + Gaussian noise
    # CHỊU ẢNH HƯỞNG của: gps_quality (module rẻ = nhiễu lớn hơn)
    # ------------------------------------------------------------------
    def _step_gps(self) -> tuple[float, float, float]:
        self.heading += random.gauss(0, 0.15)
        step_size = 0.00005

        self.true_lat += step_size * math.cos(self.heading)
        self.true_lon += step_size * math.sin(self.heading)
        self.true_altitude = max(0.0, self.true_altitude + random.gauss(0, 0.3))

        base_sensor_noise_std = 0.00003
        # gps_quality thấp -> nhiễu cảm biến lớn hơn (chia cho quality < 1)
        sensor_noise_std = base_sensor_noise_std / max(self.profile.gps_quality, 0.1)

        noisy_lat = self.true_lat + random.gauss(0, sensor_noise_std)
        noisy_lon = self.true_lon + random.gauss(0, sensor_noise_std)
        return noisy_lat, noisy_lon, self.true_altitude

    # ------------------------------------------------------------------
    # Hardware: lỗi hiếm gặp, xác suất theo hardware_reliability của profile
    # ------------------------------------------------------------------
    def _step_hardware_ok(self) -> bool:
        return random.random() < self.profile.hardware_reliability

    # ------------------------------------------------------------------
    # PUBLIC API — hàm duy nhất mà M1 cần gọi
    # ------------------------------------------------------------------
    def step(self) -> dict:
        """
        Trả về đúng 8 field mà schema.TelemetryPayload yêu cầu (KHÔNG bao gồm
        drone_id/timestamp — 2 field này do make_bronze_payload() tự thêm vào),
        CỘNG THÊM true_lat/true_lon (ground truth, chỉ để ĐÁNH GIÁ khoa học —
        không phải input hợp lệ cho bất kỳ thuật toán làm mượt/dự đoán nào,
        M2/M4 KHÔNG được dùng 2 field này để tính toán, chỉ evaluate_metrics.py
        mới được đọc).
        Profile không xuất hiện trực tiếp trong output -- nó chỉ ẢNH HƯỞNG tới
        các con số telemetry (đúng tinh thần: dữ liệu tự "kể chuyện", không cần
        lộ tên kịch bản ra Bronze layer).
        """
        lat, lon, alt = self._step_gps()
        return {
            "gps_lat": lat,
            "gps_lon": lon,
            "altitude_m": alt,
            "true_lat": self.true_lat,
            "true_lon": self.true_lon,
            "battery_level_pct": self._step_battery(),
            "wind_speed_ms": self._step_wind(),
            "motor_temp_c": self._step_motor_temp(),
            "hardware_ok": self._step_hardware_ok(),
            "network_signal_pct": self._step_network(),
        }


if __name__ == "__main__":
    print("=== So sánh baseline vs worst_case sau 300 bước (300 giây) ===")
    baseline = StochasticSimulator(seed=42, profile=DroneProfile(label="baseline"))
    worst = StochasticSimulator(seed=42, profile=next(
        p for p in CASE_STUDY_PROFILES if p.label == "worst_case"
    ))

    for _ in range(300):
        b_out = baseline.step()
        w_out = worst.step()

    print(f"baseline   : battery={b_out['battery_level_pct']:.1f}%  "
          f"motor_temp={b_out['motor_temp_c']:.1f}C  wind={b_out['wind_speed_ms']:.1f}m/s")
    print(f"worst_case : battery={w_out['battery_level_pct']:.1f}%  "
          f"motor_temp={w_out['motor_temp_c']:.1f}C  wind={w_out['wind_speed_ms']:.1f}m/s")
    print()
    print("-> worst_case phải có battery THẤP HƠN, motor_temp CAO HƠN, "
          "wind CAO HƠN baseline một cách rõ rệt (không chỉ do random).")
