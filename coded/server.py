import json
import threading
import time
import queue
from datetime import datetime
from collections import defaultdict
import paho.mqtt.client as mqtt

# 상태 관리
task_queue = queue.Queue()
client_ready = defaultdict(lambda: True)
client_list = set()
client_task_count = defaultdict(int)
client_bandwidth = defaultdict(float)
request_times = {}
completion_times = {}

resource_data = {}
resource_lock = threading.Lock()

def clear_terminal():
    print("\033[2J\033[H", end="")

def print_resource_table():
    clear_terminal()
    print("+-----------+---------------+-----------+----------+--------+---------------------+")
    print("| Client ID |   Bandwidth   |  Memory   | Battery  | Task   |     Timestamp       |")
    print("+-----------+---------------+-----------+----------+--------+---------------------+")
    with resource_lock:
        for cid, data in resource_data.items():
            print(f"| {cid:9} | {data.get('bandwidth', 0):>13.2f} | "
                  f"{data.get('memory', 0):>9.2f} | {data.get('battery', 0):>8.2f} | "
                  f"{'Running' if data.get('task', False) else 'Ready':>6} | {data.get('timestamp', ''):>19} |")
    print("+-----------+---------------+-----------+----------+--------+---------------------+")

def monitor_resource_loop():
    while True:
        print_resource_table()
        time.sleep(5)

def handle_resource_message(client, userdata, msg):
    try:
        j = json.loads(msg.payload.decode())
        cid = j.get("client_id")
        if cid:
            with resource_lock:
                resource_data[cid] = {
                    "bandwidth": j.get("bandwidth", 0.0),
                    "memory": j.get("memory", 0.0),
                    "battery": j.get("battery", 0.0),
                    "task": j.get("task", False),
                    "timestamp": datetime.now().isoformat()
                }
            client_list.add(cid)
    except Exception as e:
        print(f"[ERROR] Failed to parse resource message: {e}")

def handle_result_message(client, userdata, msg):
    try:
        j = json.loads(msg.payload.decode())
        cid = j.get("client_id", "")
        status = j.get("status", "")
        if cid and status:
            end_time = time.time()
            start_time = request_times.get(cid, end_time)
            turnaround = end_time - start_time
            print(f"[RESULT] {cid} | Status: {status} | Turnaround: {turnaround:.2f}s")
    except Exception as e:
        print(f"[ERROR] Failed to parse result: {e}")

def handle_dist_message(client, userdata, msg):
    try:
        j = json.loads(msg.payload.decode())
        cid = j.get("client_id", "")
        status = j.get("status", "")
        if status == "Inference completed":
            end_time = time.time()
            start_time = request_times.get(cid, end_time)
            turnaround = end_time - start_time
            print(f"[DIST] {cid} | Inference Time: {turnaround:.2f}s")
            client_task_count[cid] += 1
            client_ready[cid] = True
    except Exception as e:
        print(f"[ERROR] Failed to parse dist: {e}")

def subscribe_all(client):
    client.message_callback_add("resource/#", handle_resource_message)
    client.message_callback_add("result/#", handle_result_message)
    client.message_callback_add("dist/#", handle_dist_message)

    client.subscribe("resource/#", qos=1)
    client.subscribe("result/#", qos=1)
    client.subscribe("dist/#", qos=1)

def run_server(broker="localhost"):
    print("[SERVER] Starting MQTT server...")
    client = mqtt.Client("python_server")
    client.connect(broker, 1883)
    subscribe_all(client)

    threading.Thread(target=monitor_resource_loop, daemon=True).start()

    client.loop_forever()