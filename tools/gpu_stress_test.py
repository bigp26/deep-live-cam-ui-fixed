import torch
import time

print("CUDA:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))

x = torch.randn(8000, 8000, device="cuda")

start = time.time()
for _ in range(20):
    x = torch.matmul(x, x)

torch.cuda.synchronize()
print("Finished heavy GPU compute in", time.time() - start, "seconds")
