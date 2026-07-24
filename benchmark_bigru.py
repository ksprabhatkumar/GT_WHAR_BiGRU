import torch
import json
import time
from models.gt_whar import GT_WHAR

def run_benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GT_WHAR(hidden_dim=64).to(device)
    model.eval()

    # Create a dummy tensor exactly the size of a DSADS batch
    # [Batch=128, Timesteps=125, Nodes=5, Features=9]
    dummy_input = torch.randn(128, 125, 5, 9).to(device)

    # 1. Count Parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 2. Warm up GPU (First pass is always slow due to memory allocation)
    for _ in range(10):
        _ = model(dummy_input)

    # 3. Measure Latency over 50 passes
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(50):
        with torch.no_grad():
            _ = model(dummy_input)
    end_event.record()
    torch.cuda.synchronize()
    
    # Average latency per batch in milliseconds
    avg_latency_ms = start_event.elapsed_time(end_event) / 50.0

    # Save to JSON
    results = {
        "model_name": "BiGRU (GT-WHAR Baseline)",
        "parameters": total_params,
        "latency_ms": avg_latency_ms
    }
    
    with open("bigru_stats.json", "w") as f:
        json.dump(results, f, indent=4)
    print(f"✅ BiGRU Benchmark Saved! Params: {total_params:,} | Latency: {avg_latency_ms:.2f} ms")

if __name__ == "__main__":
    run_benchmark()