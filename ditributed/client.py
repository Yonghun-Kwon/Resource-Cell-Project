# ============================================================
# client.py — Final version matched to training-signal-only server
# ============================================================

import json
import time
import threading
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import socket
import paho.mqtt.client as mqtt

from socketio import send_tensor, get_local_ip
from benchmark import get_resource


BROKER_DEFAULT = "192.168.101.22"
MAX_POINTS = 1024
LIDAR_DIR = "lidar_dataset"

# ------------------------------------------------------------
# PointNet
# ------------------------------------------------------------

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
        self.fc3.bias.data.copy_(torch.eye(k).flatten())

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


# ============================================================
# Global Client State
# ============================================================

class ClientState:
    model = None
    lidar_list = []
    batch_size = 1
    load_size = 0
    idx = 0
    bandwidth = 0.0
    task_in_progress = False


# ============================================================
# Loads LiDAR dataset
# ============================================================

def load_lidar_dataset(load_size):
    files = sorted(glob.glob(f"{LIDAR_DIR}/*.npy"))
    if not files:
        raise RuntimeError("[CLIENT] No .npy files found in lidar_dataset/")

    load_size = min(load_size, len(files))
    files = files[:load_size]

    dataset = []

    for f in files:
        pts = np.load(f).astype(np.float32)

        # Resize to MAX_POINTS
        if pts.shape[0] > MAX_POINTS:
            idx = np.random.choice(pts.shape[0], MAX_POINTS, replace=False)
            pts = pts[idx]
        else:
            pad = np.zeros((MAX_POINTS - pts.shape[0], 3), dtype=np.float32)
            pts = np.vstack([pts, pad])

        dataset.append(torch.tensor(pts))  # (1024,3)

    print(f"[CLIENT] Loaded LiDAR samples: {len(dataset)}")
    return dataset


# ============================================================
# Get Batch
# ============================================================

def get_next_batch():
    start = ClientState.idx
    end = start + ClientState.batch_size

    if start >= len(ClientState.lidar_list):
        return None

    batch_samples = ClientState.lidar_list[start:end]
    ClientState.idx = end

    return torch.stack(batch_samples, dim=0)  # (B,1024,3)


# ============================================================
# Local Training
# ============================================================

def run_local_training(batch):
    model = ClientState.model
    model.train()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    optimizer.zero_grad()

    pred = model(batch)
    target = torch.zeros_like(pred)
    loss = torch.mean((pred - target) ** 2)

    loss.backward()

    grads = []
    for p in model.parameters():
        if p.grad is not None:
            grads.append(p.grad.reshape(-1))

    grad_flat = torch.cat(grads).cpu()
    return grad_flat


# ============================================================
# Handle Training Signal
# ============================================================

def handle_local_train(client_id):

    batch = get_next_batch()
    if batch is None:
        print("[CLIENT] No more batches to train!")
        return None

    grad = run_local_training(batch)

    # Create gradient sending socket
    grad_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    grad_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    grad_sock.bind((get_local_ip(), 0))
    grad_sock.listen(1)

    grad_port = grad_sock.getsockname()[1]

    # Inform server before socket accept
    result_msg = {
        "client_id": client_id,
        "status": "Gradient ready",
        "grad_port": grad_port,
        "timestamp": int(time.time())
    }

    # Send gradient asynchronously
    threading.Thread(
        target=_send_grad_thread,
        args=(grad_sock, grad),
        daemon=True
    ).start()

    return result_msg


def _send_grad_thread(sock, grad):
    conn, _ = sock.accept()
    send_tensor(conn, grad)
    conn.close()
    sock.close()


# ============================================================
# MQTT Message handler
# ============================================================

def on_message(client, userdata, msg):
    client_id = userdata

    try:
        j = json.loads(msg.payload.decode())
    except:
        return

    task = j.get("task")
    if not task:
        return

    if task == "local_train":
        ClientState.task_in_progress = True

        res = handle_local_train(client_id)
        if res:
            client.publish(f"dist/{client_id}", json.dumps(res), qos=1)

        ClientState.task_in_progress = False


# ============================================================
# Resource Monitor
# ============================================================

def start_resource_loop(client, client_id):
    def loop():
        while True:
            cpu, mem, bat = get_resource()
            payload = {
                "client_id": client_id,
                "cell_vec": [1, 1, 1, 1],
                "memory": mem,
                "battery": bat,
                "bandwidth": ClientState.bandwidth,
                "ip": get_local_ip(),
                "task": ClientState.task_in_progress,
                "timestamp": int(time.time())
            }
            client.publish(f"resource/{client_id}", json.dumps(payload), qos=1)
            time.sleep(1)

    threading.Thread(target=loop, daemon=True).start()


# ============================================================
# Run Client
# ============================================================

def run_client(broker, client_id):
    ClientState.model = PointNet(num_classes=10)

    client = mqtt.Client(client_id=client_id, userdata=client_id)
    client.on_message = on_message

    client.connect(broker, 1883, 60)
    client.subscribe(f"input/{client_id}", qos=1)

    start_resource_loop(client, client_id)
    client.loop_forever()


# ============================================================
# main
# ============================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--broker", type=str, default=BROKER_DEFAULT)
    p.add_argument("--client_id", type=str, default="rpi5")
    p.add_argument("--load_size", type=int, default=100)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    ClientState.load_size = args.load_size
    ClientState.batch_size = max(1, args.load_size // 100)

    print(f"[CLIENT] Load size = {args.load_size}")
    print(f"[CLIENT] Batch size = {ClientState.batch_size}")

    ClientState.lidar_list = load_lidar_dataset(args.load_size)
    ClientState.idx = 0

    run_client(args.broker, args.client_id)
