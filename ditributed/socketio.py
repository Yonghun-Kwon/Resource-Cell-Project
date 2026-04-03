# socketio.py
import socket
import struct
import torch
import netifaces

import numpy as np

def create_socket(is_server: bool, ip: str, port: int) -> socket.socket:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    if is_server:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((ip, port))
        s.listen(1)
        #print(f"[SERVER] Listening on {ip}:{port}")
    else:
        s.connect((ip, port))
        print(f"[CLIENT] Connected to server {ip}:{port}")
    return s


def send_tensor(conn: socket.socket, tensor: torch.Tensor):
    """
    C++ socketio::send_tensor와 유사:
    - shape_len (int64)
    - shape dims (int64 * shape_len)
    - byte_size (int64)
    - raw float data
    """
    tensor = tensor.contiguous().to(torch.float32)
    shape = list(tensor.shape)
    shape_len = len(shape)
    data_bytes = tensor.numpy().tobytes()
    byte_size = len(data_bytes)

    conn.sendall(struct.pack("q", shape_len))
    conn.sendall(struct.pack("q" * shape_len, *shape))
    conn.sendall(struct.pack("q", byte_size))
    conn.sendall(data_bytes)


def recv_exact(conn: socket.socket, nbytes: int) -> bytes:
    buf = b""
    while len(buf) < nbytes:
        chunk = conn.recv(nbytes - len(buf))
        if not chunk:
            raise RuntimeError("Socket closed while receiving data")
        buf += chunk
    return buf


def receive_tensor(conn: socket.socket) -> torch.Tensor:
    """
    안전한 텐서 수신:
    - recv_exact로 header 수신
    - bytearray로 raw data 수신 (writable)
    - numpy.frombuffer + copy()로 안전하게 배치
    - torch.tensor()로 새로운 텐서 생성
    """


    # ---- 1) shape_len ----
    raw = recv_exact(conn, 8)
    (shape_len,) = struct.unpack("q", raw)

    # ---- 2) shape dims ----
    raw = recv_exact(conn, 8 * shape_len)
    shape = struct.unpack("q" * shape_len, raw)

    # ---- 3) byte_size ----
    raw = recv_exact(conn, 8)
    (byte_size,) = struct.unpack("q", raw)

    # ---- 4) receive data into writable buffer ----
    buf = bytearray(byte_size)
    view = memoryview(buf)
    received = 0

    while received < byte_size:
        n = conn.recv_into(view[received:], byte_size - received)
        if n == 0:
            raise RuntimeError("Socket closed while receiving tensor data")
        received += n

    # ---- 5) 반드시 COPY(json) ----
    arr = np.frombuffer(buf, dtype=np.float32).copy()
    tensor = torch.tensor(arr).reshape(*shape)

    return tensor



def get_local_ip() -> str:
    """
    C++ getLocalIP와 비슷하게 eth0/wlan0 우선 검색.
    """
    for iface in ("wlan0", "eth0"):
        if iface in netifaces.interfaces():
            addrs = netifaces.ifaddresses(iface).get(netifaces.AF_INET, [])
            if addrs:
                return addrs[0]["addr"]
    # fallback: hostname
    return socket.gethostbyname(socket.gethostname())
