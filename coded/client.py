import json
import threading
import time
from datetime import datetime
import torch
import paho.mqtt.client as mqtt
from task import benchmark_run, run_inference
from utils import get_resource, receive_tensor_over_socket, get_local_ip, create_socket

client_state = {
    "task_in_progress": False,
    "number_of_cell": 0,
    "bandwidth": 0.0,
}

def topic_meta(client_id): return f"task/meta/{client_id}"
def topic_input(client_id): return f"task/input/{client_id}"
def topic_result(client_id): return f"result/{client_id}"
def topic_distribution(client_id): return f"dist/{client_id}"
def topic_resource(client_id): return f"resource/{client_id}"

def send_resource_loop(client, client_id):
    def loop():
        while True:
            cpu, mem, batt = get_resource()
            rc = (1 - cpu) * client_state["number_of_cell"]
            res = {
                "client_id": client_id,
                "cells": rc,
                "memory": mem,
                "bandwidth": client_state["bandwidth"],
                "battery": batt,
                "ip": get_local_ip(),
                "task": client_state["task_in_progress"],
                "timestamp": datetime.now().isoformat()
            }
            client.publish(topic_resource(client_id), json.dumps(res), qos=1)
            time.sleep(1)
    threading.Thread(target=loop, daemon=True).start()

def handle_message(client, userdata, msg):
    try:
        j = json.loads(msg.payload.decode())
        if "task" in j:
            client_state["task_in_progress"] = True
            if j["task"] == "benchmark":
                print("[CLIENT] Starting benchmark...")

                shape = j["shape"]
                server_ip = j["server_ip"]
                server_port = j["server_port"]
                byte_size = j["byte_size"]

                sock = create_socket(server_ip, server_port)
                start_time = time.perf_counter()
                tensor = receive_tensor_over_socket(sock, shape)
                end_time = time.perf_counter()
                sock.close()

                client_state["number_of_cell"] = benchmark_run()

                receive_time = end_time - start_time
                client_state["bandwidth"] = (byte_size / 1e6) / receive_time

                print(f"[CLIENT] Benchmark completed. Bandwidth: {client_state['bandwidth']:.2f} MB/s")
                result = {
                    "client_id": userdata,
                    "status": "Benchmark completed",
                    "timestamp": time.time()
                }
                client.publish(topic_result(userdata), json.dumps(result), qos=1)

            elif j["task"] == "tensor_receive":
                shape = j["shape"]
                server_ip = j["server_ip"]
                server_port = j["server_port"]
                model_path = j.get("model_path", "mobilenet.pt")

                print(f"[CLIENT] Receiving tensor from {server_ip}:{server_port}")
                sock = create_socket(server_ip, server_port)
                tensor = receive_tensor_over_socket(sock, shape)
                sock.close()

                model = torch.jit.load(model_path)
                run_inference(tensor, model)

                result = {
                    "client_id": userdata,
                    "status": "Inference completed",
                    "timestamp": time.time()
                }
                client.publish(topic_distribution(userdata), json.dumps(result), qos=1)

        client_state["task_in_progress"] = False

    except Exception as e:
        print(f"[ERROR] Failed to handle message: {e}")

def send_inference_result(broker: str, client_id: str):
    print("[CLIENT] Starting client...")
    client = mqtt.Client(client_id)
    client.user_data_set(client_id)
    client.on_message = handle_message
    client.connect(broker, 1883)
    client.subscribe(topic_input(client_id), qos=1)
    send_resource_loop(client, client_id)
    client.loop_forever()