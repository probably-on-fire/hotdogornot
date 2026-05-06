"""Smoke-test the rfcai venv's torch+CUDA install on the P40 box.

Validates: torch sees the GPUs, can allocate tensors on each, runs
a real matmul (catches NVML-mismatch failures that don't show up
in is_available alone), and runs a tiny ResNet-18 forward pass.

Run as the rfcai user so it picks up that venv:
  sudo -u rfcai /opt/rfcai/repo/training/.venv/bin/python \
       /opt/rfcai/repo/training/scripts/_gpu_smoke.py
"""
import sys
import torch
import torchvision.models as M

print(f"torch={torch.__version__}  cuda_runtime={torch.version.cuda}")
print(f"is_available={torch.cuda.is_available()}  device_count={torch.cuda.device_count()}")
if not torch.cuda.is_available():
    print("FAIL: cuda not available")
    sys.exit(1)

for i in range(torch.cuda.device_count()):
    name = torch.cuda.get_device_name(i)
    cap = torch.cuda.get_device_capability(i)
    mem = torch.cuda.get_device_properties(i).total_memory / 1e9
    print(f"  cuda:{i}  {name}  sm_{cap[0]}{cap[1]}  {mem:.1f} GB")

for i in range(torch.cuda.device_count()):
    dev = f"cuda:{i}"
    a = torch.randn(2048, 2048, device=dev)
    b = torch.randn(2048, 2048, device=dev)
    c = a @ b
    torch.cuda.synchronize(dev)
    print(f"  matmul on {dev}: ok  result_norm={c.norm().item():.1f}")

m = M.resnet18(weights=None).cuda().eval()
x = torch.randn(8, 3, 224, 224, device="cuda")
with torch.no_grad():
    y = m(x)
print(f"resnet18 forward: ok  out_shape={tuple(y.shape)}")
print("ALL OK")
