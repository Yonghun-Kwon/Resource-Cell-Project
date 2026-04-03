# task.py
import torch

def run_inference(input_tensor: torch.Tensor, model: torch.jit.ScriptModule) -> int:
    """
    C++ task::inference::run에 해당.
    """
    if input_tensor.dim() == 3:
        input_tensor = input_tensor.unsqueeze(0)
    expected_shape = (1, 3, 224, 224)
    if tuple(input_tensor.shape) != expected_shape:
        print(f"[ERROR] Input tensor shape {tuple(input_tensor.shape)} != {expected_shape}")
        return -1
    if input_tensor.dtype != torch.float32:
        input_tensor = input_tensor.to(torch.float32)

    with torch.no_grad():
        out = model(input_tensor)
        if isinstance(out, (list, tuple)):
            out = out[0]
        pred = int(out.argmax(1).item())
    return pred
