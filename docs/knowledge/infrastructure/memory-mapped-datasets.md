# Memory-Mapped Datasets

## What It Is

**Memory mapping** (mmap) is a way to work with files on disk as if they were
in memory, without actually loading the entire file into RAM. The operating
system loads only the pages (chunks) that are actually accessed, keeping the
rest on disk.

Think of it like reading a book: instead of memorizing every page before
reading, you flip to the page you need and read just that one. The rest stays
on the shelf.

## Why It Matters for This Project

This project runs on a machine with **8GB RAM and no GPU**. The training
dataset has ~15.8M rows × 23 features × 4 bytes (float32) ≈ **1.4 GB** of
raw feature data. If you also load labels, the full dataset is ~1.5 GB.

With PyTorch DataLoader creating batches of shape (batch_size=256, window_size=60,
n_features=23), each batch needs 256 × 60 × 23 × 4 ≈ **1.3 MB**. But the
dataset object itself holds references to all windows — and with stride=1,
there are ~73 million windows. Even storing just the index for 73M windows
would consume ~600 MB of RAM.

**The OOM crash:** Before memory mapping was implemented, the project hit
`MemoryError: Unable to allocate` when trying to load the full dataset into a
PyTorch Dataset. The solution: keep data on disk, access it lazily.

## How It's Implemented Here

### Step 1: Convert parquet to numpy (`ensure_numpy_cache()` in `train.py`)

```python
def ensure_numpy_cache(data_dir, split, feature_cols):
    # Creates: feats.npy (float32 memmap), labels.npy (int64 memmap),
    #          ticker_ranges.npz
```

This function:
1. Reads the parquet file in chunks
2. Writes features to `feats.npy` using `np.memmap(mode="w+")` — direct disk
   write, no RAM overhead
3. Writes labels to `labels.npy` (same approach)
4. Records ticker boundaries in `ticker_ranges.npz` (names, starts, ends)
5. Resumes from partial conversions by checking file sizes

The result: data lives on disk as `.npy` files, only touched pages are paged
into RAM by the OS.

### Step 2: Lazy access (`LazyTickerWindows` in `train.py`)

```python
class LazyTickerWindows(Dataset):
    def __init__(self, cache_dir, window_size, stride=1, ...):
        self.feats = np.load(feats_path, mmap_mode="r")  # read-only memmap
        self.labels = np.load(labels_path, mmap_mode="r")
        # Precompute per-ticker window offsets for O(1) lookup
        self._offsets = [...]  # cumulative count of windows per ticker

    def __getitem__(self, idx):
        # Binary search to find which ticker this window belongs to
        ti = bisect.bisect_right(self._offsets, idx) - 1
        local = idx - self._offsets[ti]
        # Compute the bar range for this window
        start = ticker_starts[ti] + local * stride
        end = start + window_size
        # Slice the memmap (only accesses the needed pages)
        x = self.feats[start:end]  # shape: (window_size, n_features)
        y = self.labels[end - 1]   # label = last bar of window
        # Normalize
        x = (x - self.norm_mean) / (self.norm_std + 1e-8)
        np.nan_to_num(x, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.from_numpy(x), y
```

**Key insight:** `self.feats[start:end]` accesses only 60 × 23 × 4 ≈ 5.5 KB
per window, regardless of the total dataset size (1.4 GB). The OS pages in only
the needed 60 rows from the 15.8M-row file.

**Peak RAM:** ~batch_size × window_size × n_features × 4 bytes + overhead.
For batch_size=256: ~1.3 MB per batch. The full dataset never touches RAM.

### Step 3: Binary search for ticker lookup

The `_offsets` array stores cumulative window counts:
```
Ticker 0: windows 0-9999
Ticker 1: windows 10000-24999
Ticker 2: windows 25000-39999
...
```

`bisect.bisect_right(self._offsets, idx) - 1` finds which ticker a flat index
belongs to in O(log n) time, avoiding a full scan.

## Tunable Parameters

| Parameter | Default | What It Does | Effect of Changing |
|-----------|---------|--------------|-------------------|
| `stride` | 15 | How many bars between windows | Lower = more windows, more disk I/O, more RAM for batches |
| `batch_size` | 256 | Windows per batch | Higher = more RAM per batch, faster training |
| Chunk size for cache | 65,536 | Rows per write during conversion | Larger = faster conversion, more RAM during conversion |

## Common Pitfalls

1. **Not using mmap_mode="r".** Opening memmaps without read-only mode can
   corrupt data if multiple processes access them. Our `LazyTickerWindows`
   uses `mmap_mode="r"`.

2. **Loading the full memmap into memory.** Doing `np.array(self.feats)` or
   iterating over all elements would defeat the purpose. Always slice.

3. **Forgetting that memmaps are slower than RAM.** Each access involves a
   potential page fault (OS loads from disk). For sequential access patterns
   (like training), the OS prefetches pages, making it nearly as fast as RAM.
   For random access, it's slower.

4. **Not checking for partial conversions.** If a previous conversion crashed
   mid-way, the `.npy` file may be incomplete. `ensure_numpy_cache()` checks
   file sizes against expected byte counts and resumes from where it left off.
