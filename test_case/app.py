import sqlite3
import time

import pandas as pd
import streamlit as st

# ==============================================================================
# 0. CẤU HÌNH — DB_PATH khớp đúng tên file mà M1/M2/M4 dùng (drone_fleet.db,
#    KHÔNG phải "DroneFleetDB.db" của bản cũ)
# ==============================================================================
DB_PATH = "drone_fleet.db"

# ==============================================================================
# 1. CẤU HÌNH NGÔN NGỮ (DICTIONARY TRANSLATION)
# ==============================================================================
CURRENT_LANG = "EN"  # Đổi thành "VI" nếu muốn giao diện tiếng Việt

LANG_MAP = {
    "VI": {
        "title": "🛸 Drone Fleet Monitoring Dashboard — M5 Workspace",
        "sidebar_header": "Cấu hình hệ thống",
        "select_drone": "Chọn Drone ID",
        "battery_label": "Năng lượng Pin",
        "motor_temp_label": "Nhiệt độ động cơ",
        "network_label": "Tín hiệu mạng",
        "etr_label": "Dự báo thời gian bay còn lại (ETR)",
        "etr_help": "Dải tin cậy được tính toán bởi mô hình Quantile Regression của M4",
        "map_title": "📍 Bản đồ định vị Drone thực tế",
        "status_stable": "Ổn định",
        "status_normal": "Bình thường",
        "status_warning": "Nhiệt độ cao",
        "status_danger": "Cảnh báo yếu!",
        "minutes": "phút",
        "no_drones_title": "⚠️ Chưa có drone nào trong hệ thống",
        "no_drones_body": (
            "Bảng `dim_drones` đang trống hoặc database `{db}` chưa tồn tại. "
            "Hãy chạy M1 (streaming) trước để khởi tạo database và đăng ký drone."
        ),
        "no_gold_title": "⏳ Chưa có dữ liệu Gold cho drone này",
        "no_gold_body": (
            "M2/M4 chưa kịp tính xong cửa sổ Gold đầu tiên cho **{drone}**. "
            "Trang sẽ tự làm mới sau vài giây — không hiển thị dữ liệu giả."
        ),
        "last_window": "Cửa sổ dữ liệu gần nhất",
    },
    "EN": {
        "title": "🛸 Drone Fleet Monitoring Dashboard — M5 Workspace",
        "sidebar_header": "System Configuration",
        "select_drone": "Select Drone ID",
        "battery_label": "Battery Level",
        "motor_temp_label": "Motor Temperature",
        "network_label": "Network Signal",
        "etr_label": "Estimated Time Remaining (ETR)",
        "etr_help": "Confidence interval calculated by Member 4's Quantile Regression model",
        "map_title": "📍 Real-time Drone Telemetry Map",
        "status_stable": "Stable",
        "status_normal": "Normal",
        "status_warning": "High Temp",
        "status_danger": "Low Battery Alert!",
        "minutes": "mins",
        "no_drones_title": "⚠️ No drones registered yet",
        "no_drones_body": (
            "`dim_drones` is empty or `{db}` does not exist yet. "
            "Run M1 (streaming) first to initialize the database and register drones."
        ),
        "no_gold_title": "⏳ No Gold data yet for this drone",
        "no_gold_body": (
            "M2/M4 haven't finished computing the first Gold window for **{drone}** yet. "
            "This page auto-refreshes in a few seconds — no placeholder data is shown."
        ),
        "last_window": "Latest data window",
    },
}

text = LANG_MAP[CURRENT_LANG]

# ==============================================================================
# 2. KHỞI TẠO CONFIG TRANG STREAMLIT
# ==============================================================================
st.set_page_config(page_title="Drone Fleet Monitoring Dashboard", layout="wide")
st.title(text["title"])
st.sidebar.header(text["sidebar_header"])

# ==============================================================================
# 3. DANH SÁCH DRONE — LẤY ĐỘNG TỪ dim_drones, KHÔNG HARDCODE
#    (bản cũ hardcode ["DRONE_001", "DRONE_002"], bỏ sót DRONE_003)
# ==============================================================================
def get_drone_ids() -> list:
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("PRAGMA busy_timeout = 5000;")  # SỬA: tránh "database is locked"
        df = pd.read_sql_query("SELECT drone_id FROM dim_drones ORDER BY drone_id", conn)
        conn.close()
        return df["drone_id"].tolist()
    except Exception:
        # DB chưa tồn tại / bảng chưa tồn tại -> coi như chưa có drone nào,
        # KHÔNG fallback sang dữ liệu giả.
        return []


# SỬA: thêm hàm lấy profile (case study) của drone -- để dashboard trả lời
# được câu hỏi "VÌ SAO drone này đang cảnh báo", không chỉ hiển thị con số.
def get_drone_profile(drone_id: str) -> dict | None:
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("PRAGMA busy_timeout = 5000;")
        df = pd.read_sql_query(
            "SELECT * FROM dim_drones WHERE drone_id = ?", conn, params=(drone_id,)
        )
        conn.close()
    except Exception:
        return None
    if df.empty:
        return None
    return df.iloc[0].to_dict()


drone_ids = get_drone_ids()

if not drone_ids:
    st.warning(f"### {text['no_drones_title']}")
    st.write(text["no_drones_body"].format(db=DB_PATH))
    time.sleep(3)
    st.rerun()

drone_selected = st.sidebar.selectbox(text["select_drone"], drone_ids)

# ==============================================================================
# 3b. HIỂN THỊ PROFILE (CASE STUDY) — trả lời câu hỏi "VÌ SAO"
#     Trước đây payload/wind_zone/... chỉ tồn tại trong RAM lúc M3 chạy,
#     không ai xem lại được. Giờ M1 đã ghi profile vào dim_drones lúc đăng
#     ký fleet, nên M5 chỉ cần SELECT ra và hiển thị -- không tính toán gì
#     thêm (đúng vai trò M5: chỉ đọc).
# ==============================================================================
profile = get_drone_profile(drone_selected)
if profile:
    zone_note = []
    if profile.get("payload_kg", 0) > 0:
        zone_note.append(f"🏋️ Đang mang {profile['payload_kg']:.1f}kg hàng")
    if profile.get("wind_zone") == "storm":
        zone_note.append("🌪️ Đang bay trong vùng gió bão")
    elif profile.get("wind_zone") == "calm":
        zone_note.append("🍃 Vùng gió yên tĩnh")
    if profile.get("battery_health", 1.0) < 0.9:
        zone_note.append(f"🔋 Pin đã xuống cấp (health={profile['battery_health']:.2f})")
    if profile.get("ambient_temp_c", 25) < 15:
        zone_note.append(f"❄️ Thời tiết lạnh ({profile['ambient_temp_c']:.0f}°C)")
    if profile.get("gps_quality", 1.0) < 0.8:
        zone_note.append("📡 Module GPS chất lượng thấp")
    if profile.get("network_zone") == "rural":
        zone_note.append("📶 Vùng phủ sóng yếu (rural)")

    profile_label = profile.get("profile_label", "baseline")
    if zone_note:
        st.sidebar.info(f"**Kịch bản: `{profile_label}`**\n\n" + "\n".join(f"- {n}" for n in zone_note))
    else:
        st.sidebar.success(f"**Kịch bản: `{profile_label}`** (điều kiện tiêu chuẩn)")

# ==============================================================================
# 4. HÀM ĐỌC DỮ LIỆU TỪ TẦNG GOLD (M2 + M4 ghi, M5 CHỈ ĐỌC)
#    Giữ nguyên cách đọc ĐÚNG của bản cũ: lấy đúng 1 dòng mới nhất theo
#    window_end. KHÔNG còn fallback generate_mock_telemetry_row() — nếu
#    chưa có dữ liệu thì trả về None và UI tự xử lý, không giả vờ có số liệu.
# ==============================================================================
def get_gold_data(drone_id: str):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA busy_timeout = 5000;")  # SỬA: tránh "database is locked"
    try:
        query = """
            SELECT * FROM fact_gold_summary
            WHERE drone_id = ?
            ORDER BY window_end DESC LIMIT 1
        """
        df = pd.read_sql_query(query, conn, params=(drone_id,))
    finally:
        conn.close()

    if df.empty:
        return None
    return df.iloc[0].to_dict()


data = get_gold_data(drone_selected)

if data is None:
    st.info(f"### {text['no_gold_title']}")
    st.write(text["no_gold_body"].format(drone=drone_selected))
    time.sleep(3)
    st.rerun()

st.caption(f"{text['last_window']}: {data.get('window_end', '—')}")

# ==============================================================================
# 5. HIỂN THỊ METRICS
# ==============================================================================
# SỬA: thêm cột thứ 4 cho network_status — M2 tính đủ 3 tín hiệu cảnh báo
# (battery_status, motor_temp_status, network_status) vào fact_gold_summary
# nhưng bản trước chỉ hiển thị 2/3, network_status bị bỏ sót khỏi khu vực
# chính (phát hiện từ heuristic evaluation, vi phạm nguyên tắc "Consistency
# and standards": 2 tín hiệu cùng cấp lại hiển thị khác nhau).
# LƯU Ý: M2 KHÔNG lưu network_mean (chỉ lưu network_status), nên cột này chỉ
# hiển thị được icon + nhãn trạng thái, không có số % kèm theo như 2 cột kia.
col1, col2, col3, col4 = st.columns(4)

# SỬA: bản cũ chỉ check "== green" rồi coi mọi giá trị khác (kể cả "yellow")
# là đỏ nguy hiểm -> mất hết ý nghĩa của control limit 3 mức mà M2 tính bằng
# classify_status() (schema.py). Giờ map đủ 3 mức green/yellow/red.
STATUS_ICON_LABEL = {
    "green":  ("🟢", text["status_stable"]),
    "yellow": ("🟡", text["status_warning"]),
    "red":    ("🔴", text["status_danger"]),
}

with col1:
    icon, label = STATUS_ICON_LABEL.get(
        data.get("battery_status"), ("⚪", "—")
    )
    st.metric(
        label=f"{text['battery_label']} ({icon} {label})",
        value=f"{data.get('battery_mean', 0.0):.1f} %",
    )

with col2:
    icon, label = STATUS_ICON_LABEL.get(
        data.get("motor_temp_status"), ("⚪", "—")
    )
    st.metric(
        label=f"{text['motor_temp_label']} ({icon} {label})",
        value=f"{data.get('motor_temp_mean', 0.0):.1f} °C",
    )

with col3:
    etr_lower = data.get("etr_lower_min")
    etr_upper = data.get("etr_upper_min")
    if etr_lower is not None and etr_upper is not None:
        st.metric(
            label=text["etr_label"],
            value=f"{round(float(etr_lower), 1)} - {round(float(etr_upper), 1)} {text['minutes']}",
            help=text["etr_help"],
        )
    else:
        # M4 chưa kịp UPDATE 4 cột của mình cho cửa sổ này -> nói rõ là
        # đang chờ, không tự bịa công thức thay cho model của M4.
        st.metric(label=text["etr_label"], value="—", help=text["etr_help"])

with col4:
    net_status = data.get("network_status")
    icon, label = STATUS_ICON_LABEL.get(net_status, ("⚪", "—"))
    # Không có network_mean được M2 lưu lại (chỉ có status), nên value chỉ
    # là icon + nhãn, không kèm số % như battery/motor_temp.
    st.metric(label=text["network_label"], value=f"{icon} {label}")

st.markdown("---")

# ==============================================================================
# 6. BẢN ĐỒ — chỉ vẽ khi M4 đã có gps_lat_smooth/gps_lon_smooth, không tự
#    chế toạ độ mặc định như bản cũ (10.7626, 106.6602) khi thiếu dữ liệu.
# ==============================================================================
st.subheader(text["map_title"])

lat_smooth = data.get("gps_lat_smooth")
lon_smooth = data.get("gps_lon_smooth")

if lat_smooth is not None and lon_smooth is not None:
    map_data = pd.DataFrame([{"lat": lat_smooth, "lon": lon_smooth}])
    st.map(map_data)
else:
    st.info(text["no_gold_body"].format(drone=drone_selected))

# ==============================================================================
# 7. POLLING — near real-time, giữ nguyên cơ chế đúng của bản cũ
# ==============================================================================
time.sleep(3)
st.rerun()
