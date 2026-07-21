"""
================================================================================
 M1 — Data / Streaming Engineer  (bản scale cho fleet lớn, ~20.000 drone)
================================================================================
Nhiệm vụ M1 (theo đúng cheat-sheet trong schema.py):
    Input  : gọi hàm step() của M3 (hoặc generate_mock_telemetry_row() trong
              lúc M3 chưa xong)
    Output : INSERT vào bronze_telemetry (raw_json_payload) — dùng
              make_bronze_payload() để validate trước khi ghi

--------------------------------------------------------------------------------
 VÌ SAO ĐỔI KIẾN TRÚC SO VỚI BẢN CŨ (3 drone, threading.Thread)
--------------------------------------------------------------------------------
Bản cũ: mỗi drone = 1 threading.Thread riêng. Với 3 drone thì ổn, nhưng với
~20.000 drone thì KHÔNG chạy được — tạo 20.000 OS thread thật sẽ ngốn hàng
GB RAM (mỗi thread mặc định tốn ~1-8MB stack), tranh chấp GIL liên tục, và
hầu hết hệ điều hành sẽ từ chối tạo thêm thread hoặc máy bị treo.

Bản này:
    1. FLEET_CONFIG được SINH BẰNG CODE (generate_fleet_config()), không gõ
       tay từng dòng — đổi NUM_DRONES là xong.
    2. Producer đổi từ "1 thread/drone" sang "1 event loop asyncio DUY NHẤT
       chạy hàng chục nghìn coroutine". asyncio xử lý được rất nhiều task
       đồng thời (mỗi coroutine chỉ ngủ/thức theo chu kỳ riêng, gần như
       không tốn thread thật), phù hợp bài toán này hơn threading nhiều.
    3. Ghi Bronze theo BATCH thay vì ghi từng dòng: producer gom bản ghi vào
       1 buffer trong bộ nhớ, cứ mỗi FLUSH_INTERVAL_SEC giây thì đẩy cả batch
       sang thread Consumer-Writer, Consumer dùng executemany() + 1 lần
       commit cho cả batch. Việc này giảm số lần commit SQLite từ ~20.000
       lần/giây xuống chỉ còn vài lần/giây — SQLite (kể cả WAL) sẽ nghẽn nếu
       commit từng dòng ở quy mô này.

KHÔNG tự định nghĩa field ở đây — mọi field, mọi validate đều lấy từ schema.py.
"""

import asyncio
import json
import logging
import os
import queue
import signal
import sqlite3
import threading
import time
from dataclasses import dataclass

from schema import (
    DDL_STATEMENTS,
    DroneInfo,
    make_bronze_payload,
    generate_mock_telemetry_row,
)

# ==============================================================================
# CẤU HÌNH CHUNG
# ==============================================================================

DB_PATH = "drone_fleet.db"

# Cứ mỗi FLUSH_INTERVAL_SEC giây, buffer trong bộ nhớ được đẩy sang Consumer
# để ghi 1 batch (executemany + 1 commit), thay vì ghi từng dòng.
FLUSH_INTERVAL_SEC = 0.5

# Cờ bật/tắt nguồn dữ liệu:
#   True  -> dùng generate_mock_telemetry_row() (chỉ dùng khi CHƯA có
#             stochastic_simulator.py, vd lúc code song song ở Ngày 2-5)
#   False -> dùng StochasticSimulator.step() thật của M3 (CÓ profile)
#
# SỬA: trước đây hardcode True — đúng lúc M3 chưa code xong. Giờ M3 ĐÃ XONG
# (stochastic_simulator.py có đầy đủ DroneProfile), nên mặc định đổi thành
# False để chạy dữ liệu thật. Vẫn cho phép bật lại mock qua biến môi trường
# nếu cần test nhanh không cần simulator (vd DRONE_USE_MOCK_DATA=true).
USE_MOCK_DATA = os.environ.get("DRONE_USE_MOCK_DATA", "false").lower() == "true"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(threadName)s: %(message)s",
)
log = logging.getLogger("M1-Streaming")


# ==============================================================================
# SINH FLEET_CONFIG BẰNG CODE — KHÔNG ĐIỀN TAY TỪNG DRONE
# ==============================================================================

def generate_fleet_config(
    num_drones: int = 100,
    base_interval_sec: float = 1.0,
    weak_signal_interval_sec: float = 2.0,
    weak_signal_every: int = 20,
) -> list[tuple[str, float]]:
    """
    Sinh danh sách (drone_id, interval_sec) cho num_drones drone.

    - drone_id đánh số 0-đệm 5 chữ số: DRONE_00001 .. DRONE_000100 (dễ sort,
      dễ đoán khi query/debug).
    - Cứ mỗi `weak_signal_every` drone thì có 1 drone "tín hiệu yếu" gửi thưa
      hơn (interval = weak_signal_interval_sec) — giữ đúng tinh thần bản cũ
      (DRONE_003 gửi thưa vì tín hiệu yếu), chỉ khác là áp dụng theo tỉ lệ
      thay vì liệt kê từng ID.

    Muốn đổi quy mô fleet: chỉ cần đổi NUM_DRONES bên dưới, không cần sửa gì
    ở producer/consumer.
    """
    if weak_signal_every <= 0:
        weak_signal_every = num_drones + 1  # tắt hẳn nhánh tín hiệu yếu

    fleet = []
    for i in range(1, num_drones + 1):
        drone_id = f"DRONE_{i:05d}"
        interval = (
            weak_signal_interval_sec
            if i % weak_signal_every == 0
            else base_interval_sec
        )
        fleet.append((drone_id, interval))
    return fleet


# Đổi số lượng drone ở đây — KHÔNG sửa gì khác trong file này.
# SỬA: cho phép override bằng biến môi trường DRONE_FLEET_NUM_DRONES, để
# script demo (run_all.sh) chạy fleet nhỏ trên 1 máy mà không cần sửa hardcode
# ở đây (dùng thật ngoài terminal của M1 vẫn mặc định 5.000 drone như cũ).
NUM_DRONES = int(os.environ.get("DRONE_FLEET_NUM_DRONES", 5_000))

# SỬA: thêm hệ số tăng tốc thời gian cho DEMO. Mỗi step() vẫn đại diện đúng
# 1 "giây mô phỏng" (không đổi công thức vật lý trong stochastic_simulator.py),
# nhưng khoảng nghỉ THẬT giữa 2 lần gọi step() được rút ngắn -- nghĩa là
# nhiều giây mô phỏng trôi qua hơn trong 1 giây thực tế. Mặc định = 1.0
# (đúng tốc độ thật, 1 giây thực = 1 giây mô phỏng) khi chạy thật ngoài
# terminal; demo có thể tăng lên (vd 10x) để thấy pin tụt xuống mức
# yellow/red trong vài phút thay vì phải đợi hơn 1 giờ.
TIME_ACCELERATION = float(os.environ.get("DRONE_TIME_ACCELERATION", 1.0))
BASE_INTERVAL_SEC = 1.0 / max(TIME_ACCELERATION, 0.001)
FLEET_CONFIG = generate_fleet_config(NUM_DRONES, base_interval_sec=BASE_INTERVAL_SEC)


# ==============================================================================
# NGUỒN DỮ LIỆU — cửa khẩu duy nhất để chuyển từ mock sang dữ liệu thật của M3
# ==============================================================================

# ==============================================================================
# GÁN PROFILE (CASE STUDY) CHO TỪNG DRONE
# ==============================================================================
# Mỗi drone được gán 1 profile trong CASE_STUDY_PROFILES (import từ M3) theo
# kiểu round-robin dựa trên số thứ tự trong drone_id -- đảm bảo:
#   1. Deterministic: chạy lại vẫn ra đúng drone đó nhận đúng profile đó
#      (không phụ thuộc random), dễ demo/giải thích lại.
#   2. Đa dạng: với fleet nhỏ (vd 8-10 drone) sẽ phủ đủ mọi kịch bản; với
#      fleet lớn (5.000 drone) các kịch bản lặp lại tuần hoàn.
def get_profile_for_drone(drone_id: str):
    """
    Trích số thứ tự từ drone_id (vd 'DRONE_00007' -> 7), rồi chọn profile
    theo (số thứ tự - 1) % số lượng profile có sẵn.
    """
    from stochastic_simulator import CASE_STUDY_PROFILES
    try:
        idx = int(drone_id.split("_")[-1]) - 1
    except (ValueError, IndexError):
        idx = 0
    return CASE_STUDY_PROFILES[idx % len(CASE_STUDY_PROFILES)]


def get_sim_output_maker():
    """
    Trả về 1 "maker" function: nhận vào drone_id, trả về hàm sim_output_fn()
    RIÊNG cho drone đó (không tham số, gọi 1 lần ra 1 dòng dữ liệu thô).

    ĐÂY LÀ NƠI DUY NHẤT cần sửa khi tích hợp dữ liệu thật của M3 — không sửa
    gì ở producer_task() hay main().

    QUAN TRỌNG: mỗi drone_id được cấp 1 simulator instance RIÊNG (khi dùng dữ
    liệu thật của M3) — vì StochasticSimulator giữ state nội bộ (battery,
    heading, wind, motor_temp...) theo kiểu random-walk/OU-process, dùng
    chung 1 instance cho nhiều drone sẽ khiến dữ liệu bị TRỘN LẪN.
    """
    if USE_MOCK_DATA:
        def maker(drone_id: str):
            # generate_mock_telemetry_row() không giữ state (thuần random mỗi
            # lần gọi) nên không bắt buộc phải tách riêng theo drone, nhưng
            # vẫn truyền drone_id vào cho nhất quán interface với nhánh thật.
            return lambda: generate_mock_telemetry_row(drone_id)
        return maker

    # --- Nhánh dùng dữ liệu thật của M3 ---
    try:
        from stochastic_simulator import StochasticSimulator
    except ImportError as e:
        raise RuntimeError(
            "USE_MOCK_DATA=False nhưng chưa tìm thấy stochastic_simulator.py "
            "của M3. Đặt lại USE_MOCK_DATA=True để chạy tạm với mock, hoặc "
            "chờ M3 nộp file."
        ) from e

    def maker(drone_id: str):
        profile = get_profile_for_drone(drone_id)
        log.info(f"{drone_id} -> profile: {profile.label} "
                 f"(payload={profile.payload_kg}kg, wind_zone={profile.wind_zone}, "
                 f"battery_health={profile.battery_health}, "
                 f"ambient={profile.ambient_temp_c}C)")
        simulator = StochasticSimulator(drone_id=drone_id, profile=profile)
        return simulator.step

    return maker


# ==============================================================================
# PRODUCER — 1 EVENT LOOP asyncio DUY NHẤT chạy hàng chục nghìn coroutine
#            (thay cho hàng chục nghìn OS thread của kiến trúc cũ)
# ==============================================================================

class AsyncBuffer:
    """
    Buffer trong bộ nhớ, gom bản ghi từ hàng chục nghìn coroutine producer.
    An toàn vì mọi thao tác append/swap đều chạy trong CÙNG 1 thread (event
    loop asyncio), không có race-condition giữa các coroutine (asyncio chỉ
    chuyển context tại điểm `await`, list.append() không await nên không bị
    xen ngang).
    """

    def __init__(self) -> None:
        self._records: list[tuple[str, dict]] = []

    def add(self, drone_id: str, sim_output: dict) -> None:
        self._records.append((drone_id, sim_output))

    def swap(self) -> list[tuple[str, dict]]:
        """Lấy toàn bộ batch hiện có và reset buffer về rỗng."""
        batch, self._records = self._records, []
        return batch


async def producer_task(
    drone_id: str,
    interval_sec: float,
    sim_output_fn,
    buffer: AsyncBuffer,
    stop_event: asyncio.Event,
) -> None:
    """1 coroutine nhẹ cho 1 drone — thay cho 1 thread/drone của bản cũ."""
    while not stop_event.is_set():
        try:
            sim_output = sim_output_fn()
            # Producer CHỈ sinh dữ liệu thô, KHÔNG tự validate ở đây — validate
            # là việc của Consumer (giữ producer đơn giản, nhanh).
            buffer.add(drone_id, sim_output)
        except Exception:
            log.exception(f"Producer {drone_id} lỗi khi sinh dữ liệu, bỏ qua vòng này")

        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval_sec)
        except asyncio.TimeoutError:
            pass  # hết interval, lặp tiếp — đây là nhánh chạy bình thường


async def flush_task(
    buffer: AsyncBuffer,
    out_queue: "queue.Queue",
    stop_event: asyncio.Event,
    flush_interval_sec: float,
) -> None:
    """
    Định kỳ gom buffer thành 1 batch rồi đẩy sang queue thread-safe cho
    Consumer-Writer. Chạy tới khi stop_event set VÀ buffer đã rỗng hẳn (đảm
    bảo không rơi mất dữ liệu ở batch cuối cùng).
    """
    while True:
        await asyncio.sleep(flush_interval_sec)
        batch = buffer.swap()
        if batch:
            out_queue.put(batch)
        if stop_event.is_set() and not batch:
            break
    # Vét nốt batch cuối (nếu có dữ liệu sinh ra đúng lúc dừng)
    final_batch = buffer.swap()
    if final_batch:
        out_queue.put(final_batch)


async def run_producers(
    fleet_config: list,
    sim_output_maker,
    out_queue: "queue.Queue",
    stop_event: asyncio.Event,
    flush_interval_sec: float = FLUSH_INTERVAL_SEC,
) -> None:
    """Khởi tạo toàn bộ coroutine producer + 1 coroutine flush, chạy tới khi dừng."""
    buffer = AsyncBuffer()

    tasks = [
        asyncio.create_task(
            producer_task(drone_id, interval, sim_output_maker(drone_id), buffer, stop_event)
        )
        for drone_id, interval in fleet_config
    ]
    tasks.append(asyncio.create_task(flush_task(buffer, out_queue, stop_event, flush_interval_sec)))

    log.info(f"Đã khởi tạo {len(fleet_config)} producer (asyncio) + 1 flush task.")
    await asyncio.gather(*tasks)


def producer_loop_thread(
    fleet_config: list,
    sim_output_maker,
    out_queue: "queue.Queue",
    stop_event: threading.Event,
) -> None:
    """
    Chạy TOÀN BỘ event loop asyncio (mọi producer) trong 1 thread riêng, tách
    biệt với thread Consumer-Writer bên dưới — vẫn giữ mô hình
    Producer-Consumer quen thuộc, chỉ khác là "Producer" giờ là 1 thread duy
    nhất chứa hàng chục nghìn coroutine, không phải hàng chục nghìn thread.
    """
    async_stop = asyncio.Event()

    async def bridge_stop():
        # threading.Event không await được -> poll nhẹ để bắc cầu sang
        # asyncio.Event mà không tốn CPU đáng kể.
        while not stop_event.is_set():
            await asyncio.sleep(0.1)
        async_stop.set()

    async def main_async():
        await asyncio.gather(
            run_producers(fleet_config, sim_output_maker, out_queue, async_stop),
            bridge_stop(),
        )

    asyncio.run(main_async())


# ==============================================================================
# CONSUMER — 1 thread duy nhất, validate + ghi Bronze THEO BATCH
# ==============================================================================

@dataclass
class ConsumerStats:
    written: int = 0
    rejected: int = 0


def consumer_loop(
    in_queue: "queue.Queue",
    db_path: str,
    stop_event: threading.Event,
    stats: ConsumerStats,
) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")  # cho phép đọc song song lúc đang ghi
    # SỬA: SQLite mặc định busy_timeout=0 -> khi M2/M4 đang ghi cùng lúc, lần
    # commit của M1 sẽ raise "database is locked" NGAY LẬP TỨC thay vì đợi.
    # Đặt busy_timeout=5000 (ms) để SQLite tự đợi/retry tối đa 5s trước khi
    # mới thật sự báo lỗi -- cần thiết khi 3 tiến trình (M1/M2/M4) cùng ghi
    # chung 1 file .db.
    conn.execute("PRAGMA busy_timeout = 5000;")
    log.info("Consumer (writer) bắt đầu, đã kết nối database")

    while not (stop_event.is_set() and in_queue.empty()):
        try:
            batch = in_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        valid_rows = []
        for drone_id, sim_output in batch:
            try:
                # ĐÚNG THEO SCHEMA: validate qua make_bronze_payload() trước
                # khi ghi. Nếu M3 lỡ gõ sai tên field, lỗi bị bắt NGAY TẠI
                # ĐÂY, không phải chờ M2/M4 đọc dữ liệu mới phát hiện.
                payload = make_bronze_payload(drone_id, sim_output)
                valid_rows.append((payload.model_dump_json(),))
            except Exception as e:
                stats.rejected += 1
                log.error(
                    f"Bản ghi của {drone_id} KHÔNG hợp lệ, bị từ chối "
                    f"(không ghi vào Bronze): {type(e).__name__}: {e}"
                )
                log.debug(f"Payload thô bị từ chối: {json.dumps(sim_output, default=str)}")

        if valid_rows:
            # 1 lần executemany + 1 lần commit cho CẢ BATCH — đây là điểm
            # khác biệt chính so với bản cũ (insert + commit từng dòng), cần
            # thiết để chịu được throughput ~hàng chục nghìn bản ghi/giây.
            conn.executemany(
                "INSERT INTO bronze_telemetry (raw_json_payload) VALUES (?)",
                valid_rows,
            )
            conn.commit()
            stats.written += len(valid_rows)

        in_queue.task_done()

    conn.close()
    log.info(f"Consumer đã dừng. Tổng kết: {stats.written} dòng ghi thành công, "
              f"{stats.rejected} dòng bị từ chối")


# ==============================================================================
# KHỞI TẠO DATABASE — tạo bảng + đăng ký fleet vào dim_drones (bulk insert)
# ==============================================================================

def init_db(db_path: str, fleet_config: list) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA busy_timeout = 5000;")
    conn.executescript(DDL_STATEMENTS)

    # SỬA: trước đây chỉ ghi drone_id/model_name/max_battery_capacity_wh —
    # profile (payload_kg, wind_zone...) chỉ tồn tại trong RAM lúc simulator
    # chạy, M2/M4/M5 không cách nào biết được. Giờ ghi profile của MỖI drone
    # thẳng vào dim_drones ngay lúc đăng ký fleet, để M5 query ra và giải
    # thích "vì sao" drone này đang cảnh báo.
    rows = []
    for drone_id, _ in fleet_config:
        profile = get_profile_for_drone(drone_id)
        drone = DroneInfo(
            drone_id=drone_id,
            profile_label=profile.label,
            payload_kg=profile.payload_kg,
            wind_zone=profile.wind_zone,
            battery_health=profile.battery_health,
            ambient_temp_c=profile.ambient_temp_c,
            gps_quality=profile.gps_quality,
            hardware_reliability=profile.hardware_reliability,
            network_zone=profile.network_zone,
        )
        rows.append((
            drone.drone_id, drone.model_name, drone.max_battery_capacity_wh,
            drone.profile_label, drone.payload_kg, drone.wind_zone,
            drone.battery_health, drone.ambient_temp_c, drone.gps_quality,
            drone.hardware_reliability, drone.network_zone,
        ))

    conn.executemany(
        """
        INSERT OR IGNORE INTO dim_drones (
            drone_id, model_name, max_battery_capacity_wh,
            profile_label, payload_kg, wind_zone,
            battery_health, ambient_temp_c, gps_quality,
            hardware_reliability, network_zone
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()
    conn.close()
    log.info(f"Database sẵn sàng tại '{db_path}', đã đăng ký {len(fleet_config)} drone (kèm profile)")


# ==============================================================================
# MAIN — khởi động Producer (asyncio, 1 thread) + Consumer (1 thread), dừng
#        an toàn bằng Ctrl+C
# ==============================================================================

def main() -> None:
    init_db(DB_PATH, FLEET_CONFIG)

    sim_output_maker = get_sim_output_maker()
    # maxsize tính theo BATCH, không phải theo bản ghi -> không cần lớn.
    batch_queue: "queue.Queue" = queue.Queue(maxsize=1000)
    stop_event = threading.Event()
    stats = ConsumerStats()

    def handle_shutdown(signum, frame):
        log.info("Nhận tín hiệu dừng — đang tắt an toàn (có thể mất vài giây để "
                  "flush batch cuối)...")
        stop_event.set()

    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    consumer_thread = threading.Thread(
        target=consumer_loop,
        args=(batch_queue, DB_PATH, stop_event, stats),
        name="Consumer-Writer",
    )
    producer_thread = threading.Thread(
        target=producer_loop_thread,
        args=(FLEET_CONFIG, sim_output_maker, batch_queue, stop_event),
        name="Producer-AsyncIO",
    )

    consumer_thread.start()
    producer_thread.start()

    log.info(
        f"Đang stream dữ liệu cho {len(FLEET_CONFIG)} drone (1 event loop asyncio, "
        f"batch mỗi {FLUSH_INTERVAL_SEC}s). Nhấn Ctrl+C để dừng."
    )

    producer_thread.join()
    consumer_thread.join()

    log.info("Toàn bộ hệ thống streaming đã dừng an toàn.")


if __name__ == "__main__":
    main()
