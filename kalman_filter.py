"""
MEMBER 3 — Kalman Filter (Sensor Fusion cho GPS)
================================================================================
Làm mượt tọa độ GPS thô (nhiễu do sai số cảm biến) thành đường bay mượt hơn.
Đây là thuật toán KINH ĐIỂN dùng thực tế trong drone/robotics cho sensor fusion.

Mô hình: constant-velocity 2D Kalman Filter.
    State vector x = [lat, lon, v_lat, v_lon]  (vị trí + vận tốc)
    - Predict: dự đoán vị trí tiếp theo dựa trên vận tốc hiện tại
    - Update : kết hợp dự đoán với giá trị GPS đo được (có nhiễu) theo trọng số
               Kalman Gain (tự động cân bằng giữa "tin dự đoán" và "tin đo đạc")

Được M4 import và gọi trong inference_service.py (Giai đoạn 3, bước 4) để làm
mượt gps_lat/gps_lon trước khi ghi vào cột gps_lat_smooth/gps_lon_smooth của
Gold layer.
"""

import numpy as np


class KalmanFilter2D:
    def __init__(self, dt: float = 1.0,
                 process_noise: float = 1e-5,
                 measurement_noise: float = 1e-4):
        """
        dt                 : bước thời gian giữa 2 lần đo (giây)
        process_noise      : độ tin cậy vào mô hình chuyển động (nhỏ = tin mô hình)
        measurement_noise  : độ nhiễu của cảm biến GPS (nhỏ = tin đo đạc)
        """
        self.dt = dt

        # State: [lat, lon, v_lat, v_lon]
        self.x = np.zeros((4, 1))
        self.initialized = False

        # State transition matrix (mô hình constant velocity)
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])

        # Measurement matrix — ta chỉ đo được lat/lon, không đo trực tiếp vận tốc
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ])

        # Ma trận hiệp phương sai nhiễu quá trình / đo đạc
        self.Q = process_noise * np.eye(4)
        self.R = measurement_noise * np.eye(2)

        # Ma trận hiệp phương sai của ước lượng — khởi tạo lớn vì chưa biết gì
        self.P = np.eye(4) * 1.0

    def update(self, raw_lat: float, raw_lon: float) -> tuple[float, float]:
        """
        Nhận 1 điểm GPS thô mỗi giây, trả về (lat, lon) đã làm mượt.
        Lần gọi đầu tiên: khởi tạo state = điểm đo luôn (chưa có gì để lọc).
        """
        z = np.array([[raw_lat], [raw_lon]])

        if not self.initialized:
            self.x[0, 0] = raw_lat
            self.x[1, 0] = raw_lon
            self.initialized = True
            return raw_lat, raw_lon

        # ---- Predict ----
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q

        # ---- Update (kết hợp với giá trị đo được) ----
        y = z - self.H @ x_pred                      # innovation (sai số dự đoán)
        S = self.H @ P_pred @ self.H.T + self.R        # innovation covariance
        K = P_pred @ self.H.T @ np.linalg.inv(S)       # Kalman Gain

        self.x = x_pred + K @ y
        self.P = (np.eye(4) - K @ self.H) @ P_pred

        return float(self.x[0, 0]), float(self.x[1, 0])


if __name__ == "__main__":
    import random
    random.seed(0)

    kf = KalmanFilter2D()
    true_lat, true_lon = 10.9447, 106.8243

    print("So sanh GPS tho (nhieu) vs GPS sau Kalman filter:")
    for i in range(10):
        true_lat += 0.00005
        true_lon += 0.00003
        raw_lat = true_lat + random.gauss(0, 0.00005)   # nhiễu cảm biến
        raw_lon = true_lon + random.gauss(0, 0.00005)

        smooth_lat, smooth_lon = kf.update(raw_lat, raw_lon)
        print(f"  step {i}: raw=({raw_lat:.6f}, {raw_lon:.6f})  "
              f"smooth=({smooth_lat:.6f}, {smooth_lon:.6f})  "
              f"true=({true_lat:.6f}, {true_lon:.6f})")
