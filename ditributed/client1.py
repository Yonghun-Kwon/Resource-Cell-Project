import json
import time
import threading
import paho.mqtt.client as mqtt
import torch.nn as nn
import torch
from benchmark import run_benchmark, get_resource
from socketio import create_socket, receive_tensor, send_tensor, get_local_ip
from task import run_inference
import torchvision.models as models

BROKER_DEFAULT = "192.168.101.22"
import torch
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

        # identity bias init
        self.fc3.weight.data.zero_()
        self.fc3.bias.data.copy_(torch.eye(k).reshape(-1))

    def forward(self, x):
        batch_size = x.size(0)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)

        x = torch.max(x, 2)[0]  # global feature
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        x = x.view(batch_size, self.k, self.k)
        return x


class PointNet(nn.Module):
    def __init__(self, num_classes=40):
        super().__init__()

        self.tnet1 = TNet(k=3)
        self.conv1 = nn.Conv1d(3, 64, 1)

        self.tnet2 = TNet(k=64)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.dropout = nn.Dropout(p=0.3)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        # x: (B, N, 3)
        x = x.transpose(2, 1)  # → (B, 3, N)

        # STN3
        T1 = self.tnet1(x)                     # (B, 3, 3)
        x = torch.bmm(T1, x)                   # (B, 3, N)

        x = F.relu(self.conv1(x))

        # STN64
        T2 = self.tnet2(x)                     # (B, 64, 64)
        x = torch.bmm(T2, x)

        x = F.relu(self.conv2(x))
        x = self.conv3(x)                      # (B,1024,N)

        x = torch.max(x, 2)[0]                 # (B,1024)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)                        # logits

        return x

REG_THROUGHPUT_OI_LINEAR = {
    "rpi3": {
        "attention": {"a": 19.0355, "b": -0.1873, "R2": 0.5571},
        "conv":      {"a": 0.4331,  "b": 0.0392,  "R2": 0.8761},
        "depthwise": {"a": 0.0382,  "b": 0.0767,  "R2": 0.2472},
        "gemm":      {"a": 11.5620, "b": -0.0196, "R2": 0.3201},
    },
    "rpi4": {
        "attention": {"a": 33.7565, "b": -0.3030, "R2": 0.3375},
        "conv":      {"a": 0.6133,  "b": 0.0851,  "R2": 0.9249},
        "depthwise": {"a": 0.3506,  "b": 0.0589,  "R2": 0.0407},
        "gemm":      {"a": 13.2079, "b": 0.1029,  "R2": 0.4976},
    },
    "rpi5": {
        "attention": {"a": 72.5994, "b": -0.4578, "R2": 0.0859},
        "conv":      {"a": 2.9602,  "b": 0.2064,  "R2": 0.8285},
        "depthwise": {"a": 3.0106,  "b": -0.5782, "R2": 0.1117},
        "gemm":      {"a": 57.6281, "b": 0.1131,  "R2": 0.2623},
    },
}
OP_ORDER = ["gemm", "conv", "depthwise", "attention"]

def compute_cell_vec_from_default(device_type: str):
    """
    디바이스 타입(rpi3/rpi4/rpi5)에 대해,
    고정된 OI 타깃값을 사용해 [GEMM, CONV, DEPTH, ATTN] 벡터를 계산.
    """

    # 연산별 OI 타깃 (네가 원하는 값)
    oi_targets = {
        "gemm": 143.2,
        "conv": 68.7,
        "depthwise": 2.2,
        "attention": 45.2,
    }

    # 디바이스 타입 fallback
    if device_type not in REG_THROUGHPUT_OI_LINEAR:
        device_type = "rpi4"

    coeffs = REG_THROUGHPUT_OI_LINEAR[device_type]

    vec = []
    for op in OP_ORDER:
        a = coeffs[op]["a"]
        b = coeffs[op]["b"]
        oi = oi_targets.get(op, 10.0)  # 혹시 키 빠지면 10.0

        thr = a + b * oi
        if thr < 0:
            thr = 0.0

        vec.append(int(thr * 100))

    return vec



class ClientState:
    task_in_progress = False
    model = None
    cell_vec = [0, 0, 0, 0]
    bandwidth = 0.0

def topic_input(client_id):   return f"input/{client_id}"
def topic_result(client_id):  return f"result/{client_id}"
def topic_dist(client_id):    return f"dist/{client_id}"
def topic_resource(client_id):return f"resource/{client_id}"
def start_resource_loop(mqtt_client: mqtt.Client, client_id: str):
    def loop():
        while True:
            cpu, mem, batt = get_resource()   # cpu: 0.0 ~ 1.0
            base_vec = ClientState.cell_vec   # [GEMM, CONV, DEPTH, ATTN]

            # 스케일 계수 (0~1로 클램프)
            scale = 1.0 - cpu
            if scale < 0.0:
                scale = 0.0
            if scale > 1.0:
                scale = 1.0

            # 각 원소에 스케일 적용
            scaled_vec = [int(v * scale) for v in base_vec]

            # 디버깅용 로그 찍어보면 왜 0인지 바로 알 수 있음
            #print(f"[RESOURCE] cpu={cpu:.2f}, scale={scale:.2f}, base_vec={base_vec}, scaled_vec={scaled_vec}")

            payload = {
                "client_id": client_id,
                "cell_vec": scaled_vec,
                "memory": mem,
                "bandwidth": ClientState.bandwidth,
                "battery": batt,
                "ip": get_local_ip(),
                "task": ClientState.task_in_progress,
                "timestamp": int(time.time()),
            }
            mqtt_client.publish(topic_resource(client_id), json.dumps(payload), qos=1)
            time.sleep(1.0)

    threading.Thread(target=loop, daemon=True).start()
def run_local_training(batch: torch.Tensor):
    """
    서버가 보낸 batch를 이용해서 1 step backward 수행하고
    gradient vector를 flat하게 만들어 반환한다.
    """
    model = ClientState.model
    model.train()

    # dummy optimizer (step은 서버에서 수행)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    optimizer.zero_grad()

    preds = model(batch)
    target = torch.zeros_like(preds)
    loss = torch.nn.functional.mse_loss(preds, target)

    loss.backward()

    # flatten gradient
    grads = []
    for p in model.parameters():
        if p.grad is not None:
            grads.append(p.grad.detach().cpu().reshape(-1))

    grad_flat = torch.cat(grads)
    return grad_flat


def handle_tensor_receive_task(j, client_id):
    shape = j["shape"]
    server_ip = j["server_ip"]
    server_port = j["server_port"]

    sock = create_socket(False, server_ip, server_port)
    batch = receive_tensor(sock)
    

    # local gradient 계산
    grad = run_local_training(batch)

    result_msg = {
        "client_id": client_id,
        "status": "Gradient completed",
        "grad_len": grad.numel(),
        "timestamp": int(time.time()),
    }

    
    send_tensor(conn, grad)
    sock.close()

    return result_msg

def handle_benchmark_task(j: dict, client_id: str):
    shape = j["shape"]
    server_ip = j["server_ip"]
    server_port = j["server_port"]
    byte_size = j["byte_size"]

    print("[CLIENT] Starting benchmark...")
    # 소켓 연결 후 텐서 수신 (네 기존 코드 유지)
    sock = create_socket(False, server_ip, server_port)
    start = time.perf_counter()
    _ = receive_tensor(sock)  # 실제 텐서 내용은 안 써도 됨
    end = time.perf_counter()
    sock.close()

    receive_time = end - start
    if receive_time > 0:
        ClientState.bandwidth = (byte_size / 1e6) / receive_time

    # 1) 새 벤치마크 실행 → summaries 얻기
    summaries = run_benchmark()   # {"GEMM": {"alpha":..., "beta":..., "r2":...}, ...}


    # 2) client_id를 디바이스 타입으로 사용 (rpi3 / rpi4 / rpi5 로 넘기면 됨)
    device_type = client_id  # 예: --client_id rpi3 이런 식으로 실행한다고 가정

    oi_targets = {
        "gemm": 143.2,
        "conv": 68.7,
        "depthwise": 2.2,
        "attention": 45.2,
    }


    # 3) 디폴트 선형식 테이블로부터 4차원 cell_vec 계산
    ClientState.cell_vec = compute_cell_vec_from_default(device_type)

    # 4) 리소스 셀 스칼라 값은 일단 합으로 정의 (원하면 바꿀 수 있음)
   # ClientState.number_of_cell = sum(cell_vec)

    print(f"[CLIENT] Benchmark completed. device={device_type}, cell_vec={ClientState.cell_vec}, bandwidth={ClientState.bandwidth:.2f} MB/s")

    # 5) 서버로 보낼 결과 메시지
    result_msg = {
        "client_id": client_id,
        "status": "Benchmark completed",
        "timestamp": int(time.time()),
    }
    return result_msg

def on_message(mqtt_client, userdata, msg):
    client_id = userdata

    try:
        j = json.loads(msg.payload.decode())
    except:
        return

    if "task" not in j:
        return

    task = j["task"]
    ClientState.task_in_progress = True

    try:
        if task == "benchmark":
            res = handle_benchmark_task(j, client_id)
            mqtt_client.publish(f"result/{client_id}", json.dumps(res), qos=1)

        elif task == "tensor_receive":
            res = handle_tensor_receive_task(j, client_id)
            mqtt_client.publish(f"dist/{client_id}", json.dumps(res), qos=1)

        else:
            print(f"[WARN] Unknown task: {task}")

    except Exception as e:
        print(f"[CLIENT ERROR] {e}")

    finally:
        ClientState.task_in_progress = False

def handle_tensor_receive_task(j, client_id):
    """
    서버가 batch를 보낸 경우 실행.
    1) batch 수신
    2) local backward → gradient 계산
    3) 서버가 지정한 소켓으로 gradient 전송
    4) MQTT로 "Gradient completed" 전송
    """
    server_ip = j["server_ip"]
    server_port = j["server_port"]

    # -----------------------------
    # 1) 서버에서 batch 수신
    # -----------------------------
    sock = create_socket(False, server_ip, server_port)
    batch = receive_tensor(sock)
    sock.close()

    # -----------------------------
    # 2) gradient 계산
    # -----------------------------
    grad = run_local_training(batch)

    # -----------------------------
    # 3) 서버에게 gradient 전송 (서버가 접속함)
    # -----------------------------
    # 서버는 이 포트를 사용해 다시 접속함
    grad_sock = create_socket(True, get_local_ip(), 0)
    conn, port = grad_sock.accept()
    send_tensor(conn, grad)
    conn.close()
    grad_sock.close()

    # -----------------------------
    # 4) MQTT로 gradient 정보만 전달
    # -----------------------------
    result_msg = {
        "client_id": client_id,
        "status": "Gradient completed",
        "grad_port": port,
        "grad_len": grad.numel(),
        "timestamp": int(time.time()),
    }
    return result_msg


def run_inference(input_data: torch.Tensor):

    x = input_data

    # dtype 변환
    if x.dtype != torch.float32:
        x = x.to(torch.float32)

    # -----------------------------
    # ① (3,224,224) → (1,3,224,224)
    # -----------------------------
    if x.dim() == 3:
        x = x.unsqueeze(0)

    # -----------------------------
    # ② 배치 형태(N,3,224,224) 허용
    # -----------------------------
    if x.dim() != 4:
        print(f"[ERROR] Unsupported input shape: {tuple(x.shape)}")
        return -1

    N, C, H, W = x.shape
    if (C, H, W) != (3, 224, 224):
        print(f"[ERROR] Input tensor shape: {tuple(x.shape)} (expected (*,3,224,224))")
        return -1

    try:
        with torch.no_grad():
            out = ClientState.model(x)

            # MobileNet: out shape = (N,1000)
            if isinstance(out, (list, tuple)):
                out = out[0]

            out = out.to(torch.float32)
            preds = out.argmax(1).tolist()   # list로 변환

        return preds   # ex) [15, 2, 9, 3, 0]

    except Exception as e:
        print(f"[ERROR] Inference failed: {e}")
        return -1


def run_client(broker: str, client_id: str):
    print("[CLIENT] Starting client...")
    ClientState.cell_vec = compute_cell_vec_from_default(client_id)
    #weights = ViT_B_16_Weights.IMAGENET1K_V1
    #ClientState.model= vit_b_16(weights=weights).to("cpu").eval()
    
    ClientState.model = PointNet(num_classes=10)   # 예시
    ClientState.model.to("cpu")
    client = mqtt.Client(
        client_id=client_id,
        callback_api_version=mqtt.CallbackAPIVersion.VERSION1,
    )
    client.user_data_set(client_id)
    client.on_message = on_message

    client.connect(broker, 1883, 60)
    client.subscribe(topic_input(client_id), qos=1)
    start_resource_loop(client, client_id)
    client.loop_forever()

