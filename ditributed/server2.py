# ============================================
# server.py  (Training Signal Only Version)
# ============================================

import json
import time
import threading
import socket
import torch
import queue
import paho.mqtt.client as mqtt
from socketio import receive_tensor, get_local_ip

BROKER_DEFAULT = "192.168.101.22"

# 상태 저장
resource_data = {}
resource_lock = threading.Lock()
client_list = set()
client_ready = {}
client_task_count = {}

global_model = None
optimizer = None
grad_store = {}

# ============================================
# PointNet
# ============================================

import torch.nn as nn
import torch.nn.functional as F

class TNet(nn.Module):
    def __init__(self, k=3):
        super().__init__()
        self.k = k
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.fc3.weight.data.zero_()
        self.fc3.bias.data.copy_(torch.eye(k).reshape(-1))

    def forward(self, x):
        B = x.size(0)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        x = torch.max(x, 2)[0]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x.view(B, self.k, self.k)


class PointNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.tnet1 = TNet(3)
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.tnet2 = TNet(64)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.dropout = nn.Dropout(0.3)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = x.transpose(2, 1)
        T1 = self.tnet1(x)
        x = torch.bmm(T1, x)
        x = F.relu(self.conv1(x))
        T2 = self.tnet2(x)
        x = torch.bmm(T2, x)
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        x = torch.max(x, 2)[0]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)

# ============================================
# MQTT: Resource update
# ============================================

def on_resource_message(client, userdata, msg):
    try:
        j = json.loads(msg.payload.decode())
    except:
        return

    cid = j.get("client_id")
    if not cid:
        return
    
    with resource_lock:
        resource_data[cid] = j
    
    client_list.add(cid)
    if cid not in client_ready:
        client_ready[cid] = True
        client_task_count[cid] = 0

# ============================================
# Gradient receiver
# ============================================

def receive_grad_from_socket(cid, port):
    client_ip = resource_data[cid]["ip"]
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((client_ip, port))
    grad = receive_tensor(sock)
    sock.close()
    print(f"[SERVER] Gradient received from {cid}, shape={grad.shape}")
    return grad

def aggregate_and_update():
    global grad_store, global_model, optimizer

    if len(grad_store) < len(client_list):
        return  # 아직 다 안 모임

    grads = []
    for g in grad_store.values():
        grads.append(g)

    agg = sum(grads) / len(grads)

    optimizer.zero_grad()

    offset = 0
    for p in global_model.parameters():
        num = p.numel()
        p.grad = agg[offset:offset+num].reshape(p.shape)
        offset += num

    optimizer.step()
    print("[SERVER] Global model updated")

    grad_store = {}

# ============================================
# Dist message (client → server)
# ============================================

def handle_dist_message(client, userdata, msg):
    try:
        j = json.loads(msg.payload.decode())
    except:
        return

    cid = j.get("client_id")
    status = j.get("status")

    if status == "Gradient ready":
        port = j["grad_port"]
        grad = receive_grad_from_socket(cid, port)
        grad_store[cid] = grad
        aggregate_and_update()
        client_ready[cid] = True

# ============================================
# Server sends only training signal
# ============================================

def broadcast_train_signal(mqtt_client):
    msg = {
        "task": "local_train",
        "timestamp": int(time.time())
    }
    for cid in client_list:
        mqtt_client.publish(f"input/{cid}", json.dumps(msg))
        print(f"[SERVER] Training signal sent to {cid}")

# ============================================
# Input Loop
# ============================================

def user_input_loop(mqtt_client):
    while True:
        cmd = input("Command (train / q): ").strip()

        if cmd == "train":
            broadcast_train_signal(mqtt_client)

        elif cmd in ("q", "quit", "exit"):
            break

# ============================================
# Entry
# ============================================

def run_server(broker=BROKER_DEFAULT):
    global global_model, optimizer
    global_model = PointNet(num_classes=10)
    optimizer = torch.optim.SGD(global_model.parameters(), lr=0.01)

    resource_client = mqtt.Client(
        client_id="server_resource_monitor",
        callback_api_version=mqtt.CallbackAPIVersion.VERSION2
    )

    data_client = mqtt.Client(
        client_id="server_data",
        callback_api_version=mqtt.CallbackAPIVersion.VERSION2
    )


    resource_client.on_message = on_resource_message
    data_client.message_callback_add("dist/#", handle_dist_message)

    resource_client.connect(broker)
    data_client.connect(broker)

    resource_client.subscribe("resource/#")
    data_client.subscribe("dist/#")

    resource_client.loop_start()
    data_client.loop_start()

    print("[SERVER] Server started. Type 'train' to broadcast training signal.")

    user_input_loop(data_client)

    resource_client.loop_stop()
    data_client.loop_stop()
