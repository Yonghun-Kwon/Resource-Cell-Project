# server.py
import json
import time
import threading
import queue
import sys
import torch
import paho.mqtt.client as mqtt
import glob
from socketio import create_socket, send_tensor, get_local_ip
import numpy as np

# ============================
# 기본 설정 / 전역 변수
# ============================

BROKER_DEFAULT = "192.168.101.22"

# 자동 포트 생성용
BASE_PORT = 9000
PORT_RANGE = 500
used_ports = set()
client_ports = {}

task_queue = queue.Queue()
client_ready = {}
client_list = set()
client_task_count = {}
request_times = {}

resource_data = {}
resource_lock = threading.Lock()

dist_start_time = None


# ============================
#  자동 포트 생성
# ============================

def get_port_for_client(cid: str):
    """client_id에 대해 항상 같은 포트를 반환한다."""
    if cid in client_ports:
        return client_ports[cid]

    raw = abs(hash(cid)) % PORT_RANGE
    port = BASE_PORT + raw

    while port in used_ports:
        port += 1

    used_ports.add(port)
    client_ports[cid] = port
    return port


# ============================
# 리소스 테이블 출력
# ============================

def load_lidar_dataset(path, max_points=1024):
    """
    path: *.npy 파일이 저장된 경로
    """
    files = glob.glob(f"{path}/*.npy")
    loaded = []

    for f in files:
        pts = np.load(f)  # (N,3)
        if pts.shape[0] > max_points:
            idx = np.random.choice(pts.shape[0], max_points, replace=False)
            pts = pts[idx]
        elif pts.shape[0] < max_points:
            pad = np.zeros((max_points - pts.shape[0], 3))
            pts = np.vstack([pts, pad])

        t = torch.tensor(pts, dtype=torch.float32)   # (N,3)
        loaded.append(t)

    print(f"[SERVER] Loaded {len(loaded)} lidar samples")
    return loaded

def clear_terminal():
    print("\033[2J\033[1;1H", end="")

def print_resource_table():
    clear_terminal()
    print("+-----------+----------------+---------+-------+-------+--------+-------+-----------+----------+--------+---------------------+")
    print("| Client ID |       IP       |  Task   | GEMM  | CONV  | DEPTH  | ATTN  | Bandwidth |  Memory  | Battery|     Timestamp       |")
    print("+-----------+----------------+---------+-------+-------+--------+-------+-----------+----------+--------+---------------------+")

    with resource_lock:
        for cid, j in resource_data.items():
            ts = j.get("timestamp", 0)
            ts_str = time.strftime("%F %T", time.localtime(ts))

            gemm, conv, depth, attn = j.get("cell_vec", [0, 0, 0, 0])
            mem = j.get("memory", 0.0)
            bat = j.get("battery", 0.0)
            bw  = j.get("bandwidth", 0.0)
            ip  = j.get("ip", "unknown")
            task = "Running" if j.get("task", False) else "Ready"

            print(
                f"| {cid:9} | {ip:14} | {task:7} |"
                f" {gemm:5d} | {conv:5d} | {depth:6d} | {attn:5d} |"
                f" {bw:9.1f} | {mem:8.1f} | {bat:6.1f} | {ts_str} |"
            )

    print("+-----------+----------------+---------+-------+-------+--------+-------+-----------+----------+--------+---------------------+")


def monitor_resource_loop():
    while True:
        print_resource_table()
        time.sleep(30)


# ============================
# MQTT 수신 핸들러
# ============================

def on_resource_message(client, userdata, msg):
    global resource_data
    try:
        j = json.loads(msg.payload.decode())
    except Exception:
        return

    cid = j.get("client_id")
    if not cid:
        return

    cell_vec = j.get("cell_vec")
    if cell_vec is None:
        legacy = j.get("cells")
        if isinstance(legacy, list):
            cell_vec = legacy
        else:
            try:
                v = int(legacy)
            except:
                v = 0
            cell_vec = [v, 0, 0, 0]

    normalized = {
        "ip": j.get("ip", "unknown"),
        "resource_cells": 0,
        "cell_vec": cell_vec,
        "memory": j.get("memory", 0.0),
        "battery": j.get("battery", 0.0),
        "bandwidth": j.get("bandwidth", 0.0),
        "task": j.get("task", False),
        "timestamp": j.get("timestamp", int(time.time())),
    }

    with resource_lock:
        resource_data[cid] = normalized
    client_list.add(cid)

    if cid not in client_ready:
        client_ready[cid] = True
        client_task_count[cid] = 0


def handle_result_message(client, userdata, msg):
    try:
        j = json.loads(msg.payload.decode())
    except Exception:
        return

    cid = j.get("client_id", "")
    status = j.get("status", "")
    vec = j.get("number_of_cell_vec")

    if cid in request_times:
        start = request_times[cid]
        turnaround = (time.time() - start) * 1000.0
        print(f"[RESULT] {cid} | {status} | {turnaround:.1f} ms")
    else:
        print(f"[RESULT] {cid} | {status}")

    if vec is not None:
        print(f"[RESULT] {cid} number_of_cell_vec = {vec}")


# ============================
# 도우미 함수
# ============================

def all_clients_ready():
    return all(client_ready.values()) if client_ready else True


# ============================
#  텐서 전송
# ============================

def send_tensor_to_client(mqtt_client: mqtt.Client, cid: str, tensor: torch.Tensor):
    server_ip = get_local_ip()
    port = get_port_for_client(cid)

    server_sock = create_socket(True, server_ip, port)

    metadata = {
        "task": "tensor_receive",
        "shape": list(tensor.shape),
        "byte_size": tensor.numel() * 4,
        "server_ip": server_ip,
        "server_port": port,
        "timestamp": int(time.time()),
    }

    mqtt_client.publish(f"input/{cid}", json.dumps(metadata), qos=1)
    request_times[cid] = time.time()

    conn, _ = server_sock.accept()
    send_tensor(conn, tensor)
    conn.close()
    server_sock.close()


def send_batch_to_client(mqtt_client, cid, tensor_list):
    if not tensor_list:
        return

    batch_tensor = torch.cat(tensor_list, dim=0)
    send_tensor_to_client(mqtt_client, cid, batch_tensor)


# ============================
#  dist 처리
# ============================

TASK_BATCH_VEC = {
    "rpi3": [1],
    "rpi4": [1],
    "rpi5": [1],
}

def handle_dist_message(client, userdata, msg):
    global dist_start_time

    try:
        j = json.loads(msg.payload.decode())
    except Exception:
        return

    cid = j.get("client_id", "")
    status = j.get("status", "")
    if status != "Inference completed":
        return

    end = time.time()
    start = request_times.get(cid, end)
    #print(f"[DIST] From {cid} | {(end - start) * 1000:.1f} ms")

    client_task_count[cid] = client_task_count.get(cid, 0) + 1
    client_ready[cid] = True

    if task_queue.empty() and all_clients_ready():
        total_ms = (time.time() - dist_start_time) * 1000.0
        print("[SERVER] All tasks completed!")
        print(f"Total Time: {total_ms:.1f} ms")
        for cid2, cnt in client_task_count.items():
            print(f"Client {cid2}: {cnt} tasks")
        return

    vec = TASK_BATCH_VEC.get(cid, [1])
    batch_size = len(vec)

    tensors = []
    for _ in range(batch_size):
        try:
            tensors.append(task_queue.get_nowait())
        except queue.Empty:
            break

    if not tensors:
        if all_clients_ready():
            total_ms = (time.time() - dist_start_time) * 1000.0
            print("[SERVER] All tasks completed!")
            print(f"Total Time: {total_ms:.1f} ms")
        return

    client_ready[cid] = False

    threading.Thread(
        target=send_batch_to_client,
        args=(client, cid, tensors),
        daemon=True
    ).start()


# ============================
# initial task dispatch
# ============================

def dispatch_initial_tasks(mqtt_client: mqtt.Client):
    global dist_start_time

    # --- dist 실행 시 카운터 초기화 ---
    dist_start_time = time.time()
    for cid in client_list:
        client_task_count[cid] = 0
        client_ready[cid] = True

    # --- 작업 생성 ---
    for _ in range(100):
        x = torch.randn(1, 3, 224, 224, dtype=torch.float32)
        task_queue.put(x)

    # --- 초기 배치 전송 ---
    for cid in list(client_list):
        vec = TASK_BATCH_VEC.get(cid, [1])
        batch_size = len(vec)

        tensors = []
        for _ in range(batch_size):
            try:
                tensors.append(task_queue.get_nowait())
            except queue.Empty:
                break

        if not tensors:
            break

        client_ready[cid] = False

        threading.Thread(
            target=send_batch_to_client,
            args=(mqtt_client, cid, tensors),
            daemon=True
        ).start()



# ============================
# benchmark task
# ============================

def send_benchmark(mqtt_client: mqtt.Client):
    data = torch.randn(1, 4096, dtype=torch.float32)
    shape = list(data.shape)
    byte_size = data.numel() * 4
    server_ip = get_local_ip()

    index = 0
    for cid in list(client_list):
        port = get_port_for_client(cid)

        server_sock = create_socket(True, server_ip, port)

        metadata = {
            "task": "benchmark",
            "shape": shape,
            "byte_size": byte_size,
            "server_ip": server_ip,
            "server_port": port,
            "timestamp": int(time.time()),
        }

        mqtt_client.publish(f"input/{cid}", json.dumps(metadata), qos=1)
        request_times[cid] = time.time()

        conn, _ = server_sock.accept()
        send_tensor(conn, data)
        conn.close()
        server_sock.close()

        print(f"[SERVER] Sent benchmark to {cid}")

        index += 1


# ============================
# user input loop
# ============================

def user_input_loop(data_client: mqtt.Client):
    global dist_start_time
    while True:
        cmd = input("Type 'bench' / 'dist' (q to quit): ").strip()
        if cmd == "bench":
            send_benchmark(data_client)
        elif cmd == "dist":
            dist_start_time = time.time()
            dispatch_initial_tasks(data_client)
        elif cmd in ("q", "quit", "exit"):
            break


# ============================
# main entry
# ============================

def run_server(broker: str = BROKER_DEFAULT):
    resource_client = mqtt.Client(
        client_id="server_resource_monitor",
        callback_api_version=mqtt.CallbackAPIVersion.VERSION1,
    )
    data_client = mqtt.Client(
        client_id="server_data",
        callback_api_version=mqtt.CallbackAPIVersion.VERSION1,
    )

    resource_client.on_message = on_resource_message
    data_client.message_callback_add("result/#", handle_result_message)
    data_client.message_callback_add("dist/#", handle_dist_message)

    resource_client.connect(broker, 1883, 60)
    data_client.connect(broker, 1883, 60)

    resource_client.subscribe("resource/#", qos=1)
    data_client.subscribe("result/#", qos=1)
    data_client.subscribe("dist/#", qos=1)

    threading.Thread(target=monitor_resource_loop, daemon=True).start()

    resource_client.loop_start()
    data_client.loop_start()

    try:
        user_input_loop(data_client)
    finally:
        resource_client.loop_stop()
        data_client.loop_stop()
        resource_client.disconnect()
        data_client.disconnect()