import socket
import struct
import torch
import fcntl
import array

def create_socket(ip: str, port: int, is_server: bool = False):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    if is_server:
        s.bind((ip, port))
        s.listen(1)
        print(f"[SERVER] Listening on {ip}:{port}")
        return s
    else:
        s.connect((ip, port))
        print(f"[CLIENT] Connected to server {ip}:{port}")
        return s

def receive_tensor_over_socket(sock, shape):
    total_bytes = torch.tensor(shape).prod().item() * 4  # float32
    buffer = bytearray(total_bytes)
    view = memoryview(buffer)
    while total_bytes > 0:
        nbytes = sock.recv_into(view, total_bytes)
        if nbytes == 0:
            raise RuntimeError("Socket connection closed.")
        view = view[nbytes:]
        total_bytes -= nbytes
    tensor = torch.frombuffer(buffer, dtype=torch.float32).reshape(shape).clone()
    return tensor

def send_tensor_over_socket(sock, tensor: torch.Tensor):
    tensor = tensor.contiguous()
    shape = list(tensor.shape)
    shape_len = len(shape)
    byte_size = tensor.numel() * 4

    # 전송: shape 길이 -> shape 리스트 -> 바이트 크기 -> 데이터
    sock.sendall(struct.pack("Q", shape_len))
    sock.sendall(struct.pack(f"{shape_len}Q", *shape))
    sock.sendall(struct.pack("Q", byte_size))
    sock.sendall(tensor.numpy().tobytes())

def get_local_ip():
    try:
        interfaces = ["eth0", "wlan0"]
        for iface_name in interfaces:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            iface_bytes = iface_name.encode("utf-8")
            ip = fcntl.ioctl(
                s.fileno(),
                0x8915,  # SIOCGIFADDR
                struct.pack("256s", iface_bytes[:15])
            )[20:24]
            return socket.inet_ntoa(ip)
    except Exception:
        return "127.0.0.1"

def get_resource():
    import random
    cpu = random.uniform(0.1, 0.8)
    mem = random.uniform(0.1, 0.9)
    batt = random.uniform(0.3, 1.0)
    return cpu, mem, batt