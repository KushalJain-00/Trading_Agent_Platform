# Gradient Clipping and Stability

## What It Is

**Exploding gradients** happen when gradient values grow exponentially during
backpropagation, causing massive parameter updates that destabilize training.
The loss jumps to NaN or oscillates wildly without converging.

**Gradient clipping** caps the maximum gradient norm to prevent this. The most
common method is `clip_grad_norm_` from PyTorch, which rescales all gradients
so their total norm doesn't exceed a threshold:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

This preserves the direction of the gradient but limits its magnitude.

## Why It Matters for This Project

LSTMs and RNNs are particularly prone to exploding gradients because they
backpropagate through time — gradients are multiplied at each timestep in the
sequence. With a window size of 60, gradients propagate through 60 timesteps.
If the gradient grows by even 5% at each step, after 60 steps it's 1.05^60 ≈ 18x
larger. This is a conservative estimate; in practice, gradients can grow much
faster.

This project experienced instability during initial training runs where the
loss would spike to NaN after a few epochs. Adding gradient clipping resolved
this.

## How It's Implemented Here

In `train_one()` in `train.py` (line ~291):
```python
# After loss.backward() and before optimizer.step()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

The `max_norm=1.0` threshold is a conservative choice — it clips gradients
aggressively. For our model sizes (LSTM with 64 hidden units, CNN with 64
channels), this keeps training stable without slowing convergence noticeably.

The training loop also includes a safety check:
```python
if torch.isnan(loss) or torch.isinf(loss):
    print(f"  WARNING: NaN/inf loss at batch {batch_idx}, skipping")
    continue
```
This skips problematic batches rather than crashing the entire training run.

## Tunable Parameters

| Parameter | Default | What It Does | Effect of Increasing | Effect of Decreasing |
|-----------|---------|--------------|---------------------|---------------------|
| `max_norm` | 1.0 | Maximum allowed gradient norm | Allows larger updates, may destabilize | More conservative, slower convergence |
| NaN check | Enabled | Skips batches with NaN/inf loss | No effect (safety net) | Training may crash on bad batches |

**When to increase max_norm:** If training is too slow and the loss decreases
steadily without spikes. Try 2.0 or 5.0.

**When to decrease max_norm:** If loss oscillates wildly or hits NaN. Try 0.5
or 0.1.

## Common Pitfalls

1. **Clipping too aggressively.** Setting `max_norm=0.01` effectively freezes
   learning because gradients are scaled down to near-zero. The model won't
   converge.

2. **Only clipping, not investigating.** Gradient clipping is a bandaid. If you
   need aggressive clipping (max_norm < 0.1), the real problem is usually the
   learning rate being too high, bad data, or a model architecture issue.

3. **Forgetting to clip for some models.** If you train LSTM, CNN, and CNN-LSTM
   but only clip for LSTM, the other models may train unstably. Our
   `train_one()` function is shared across all architectures, so clipping
   applies uniformly.
