# Training on Constrained Hardware

## What It Is

This project trains 4 neural network architectures on a **CPU-only laptop with
8GB RAM** — no GPU, no cloud instances. This document captures the practical
lessons learned from making it work.

## Why It Matters for This Project

Most ML tutorials assume you have a GPU and 32GB+ RAM. This project proved
that meaningful deep learning on financial time series is possible on modest
hardware, but requires careful choices about batch size, workers, and patience.

## Practical Lessons

### Hardware Specs

- **CPU:** Intel i5-1135G7 (4 cores, 8 threads)
- **RAM:** 8 GB
- **Storage:** SSD (important for memory-mapped I/O)
- **GPU:** None (CPU-only training)

### Batch Size Tradeoffs

| Batch Size | Training Speed | Memory Usage | Gradient Quality |
|------------|---------------|--------------|-----------------|
| 32 | Slow (~3x) | Low (~50 MB) | Noisy gradients, may not converge |
| 64 | Moderate (~2x) | Moderate (~100 MB) | Reasonable |
| 126 | Good (~1.3x) | ~200 MB | Good |
| 256 (default) | Fast (1x) | ~400 MB | Smooth gradients |
| 512 | Fastest (0.8x) | ~800 MB | Very smooth, but may overfit |

**Default: 256.** This is the sweet spot for 8GB RAM — fast enough for
reasonable training times, small enough to leave room for the OS and other
processes.

### num_workers (DataLoader Parallelism)

```python
DataLoader(dataset, batch_size=256, num_workers=3, ...)
```

- `num_workers=0`: Main thread loads data (slow, blocks training)
- `num_workers=2-3`: Good for 4-core CPU (our default: 3)
- `num_workers=4+`: May cause contention with training thread

On 8GB RAM, each worker uses ~100-200 MB. With 3 workers + training process,
that's ~1.5 GB just for data loading. Don't exceed 4 workers on 8GB.

### Thread Configuration

For CPU-only PyTorch, set these environment variables:

```bash
export OMP_NUM_THREADS=4      # Match physical cores
export MKL_NUM_THREADS=4      # Match physical cores
export TORCH_NUM_THREADS=4     # PyTorch internal threads
```

Setting these higher than physical cores causes thread contention and can
slow down training.

### Realistic Training Times

| Model | Epochs | Time per Epoch | Total Time |
|-------|--------|---------------|------------|
| CNN1D | 15 | ~3-4 min | ~45-60 min |
| LSTM | 15 | ~6-8 min | ~90-120 min |
| CNN-LSTM | 15 | ~8-10 min | ~120-150 min |
| Transformer | 15 | ~10-12 min | ~150-180 min |

These are approximate — actual times depend on data size and CPU load.

### Early Stopping

With `PATIENCE=4`, training typically completes in 10-12 epochs (not the
maximum 15). The model usually converges by epoch 8-10, with the last 2-4
epochs showing no improvement.

### Memory Monitoring

The `check_memory(min_gb=1.5)` function in `train.py` reads `/proc/meminfo`
and warns if available RAM drops below 1.5 GB:

```python
def check_memory(min_gb=1.5):
    with open("/proc/meminfo") as f:
        for line in f:
            if "MemAvailable" in line:
                avail_gb = int(line.split()[1]) / 1024 / 1024
                if avail_gb < min_gb:
                    print(f"WARNING: Only {avail_gb:.1f}GB available")
```

Run this before training to ensure enough headroom.

## Common Pitfalls

1. **Using batch_size=1024 on 8GB RAM.** This will OOM. Start with 256 and
   reduce if needed.

2. **Ignoring memory-mapped datasets.** Without mmap, loading 15.8M rows into
   a PyTorch Dataset would require ~6 GB just for the data tensor. Memory
   mapping keeps RAM usage under 500 MB.

3. **Setting num_workers too high.** On 8GB RAM with 4 cores, `num_workers=8`
   would create 8 processes × 200 MB = 1.6 GB just for workers, plus the
   training process. This can trigger OOM.

4. **Running training in background without monitoring.** If a training run
   consumes too much swap, the OS may kill the process. Use `check_memory()`
   and monitor with `htop`.

5. **Expecting GPU-like speed.** CPU training is 10-50x slower than GPU for
   the same model. Plan accordingly — a full training run takes 2-3 hours,
   not minutes.
