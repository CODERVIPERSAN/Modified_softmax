# Modified Softmax: A Novel Sigmoid-Based Probability Normalization

## The Problem with Standard Softmax

Standard softmax is **biased towards larger values** due to exponential amplification:

```
For inputs [25, 50]:
  Standard Softmax: [0.000000, 1.000000]  ← 25 gets ZERO probability!
```

Even though 25 is a "good number", it gets completely overshadowed.

## Novel Solution: Modified Softmax

### Basic Modified Softmax
Apply sigmoid to each value, then normalize:

```
p_i = σ(x_i) / Σ σ(x_j)
```

**Problem**: For large values, sigmoid saturates to 1.0:
```
σ(25) ≈ 1.0, σ(50) ≈ 1.0 → Both get ~50% probability
```

### Weighted Modified Softmax (Recommended)
Scale inputs before sigmoid:

```
p_i = σ(α·x_i) / Σ σ(α·x_j)
```

Where `α` (alpha) controls sensitivity (typically 0.05-0.2).

## Results Comparison

### For inputs [25, 50]:
| Method | P(25) | P(50) |
|--------|-------|-------|
| Standard Softmax | 0.00% | 100.00% |
| Modified Softmax | 50.00% | 50.00% |
| Weighted (α=0.1) | 48.20% | 51.80% |

### For frequencies [100, 50, 25, 10, 5]:
| Method | Ratio (class 0 vs class 4) |
|--------|---------------------------|
| Standard Softmax | ∞ (completely dominated) |
| Modified Softmax | 1.01x (too equal) |
| Weighted (α=0.05) | 1.77x (balanced) |

## Key Findings

1. **Standard Softmax**: Exponential bias makes small values negligible
2. **Modified Softmax**: Sigmoid saturation treats all "good" values equally
3. **Weighted Modified**: Best of both - preserves distinctions with controlled smoothing

## Usage

```python
import numpy as np

def weighted_modified_softmax(logits, alpha=0.1):
    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    sig_logits = sigmoid(alpha * np.array(logits))
    return sig_logits / np.sum(sig_logits)

# Example
probs = weighted_modified_softmax([75, 25], alpha=0.1)
# Output: [0.52, 0.48] - 25 still has meaningful influence!
```

## Novelty

This approach appears to be **novel** - combining sigmoid transformation with normalization to address softmax's inherent bias has not been prominently explored in literature.

## Files
- `softmax_comparison.py` - Full implementation and experiments

## Author's Contribution
Original idea of using sigmoid averaging for probability normalization to capture the influence of smaller values that standard softmax ignores.
