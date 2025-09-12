# FlashAttention-keep-first-tokens

* This is a modified version of [FlashAttention2 2.8.3](https://github.com/Dao-AILab/flash-attention), which supports keeping attention to the first N tokens in the sequence.

## Why is this useful?
* Sometimes we need to keep the attention to the first a few tokens when using sliding window, like what [Streaming-LLM](https://arxiv.org/abs/2309.17453) does.
* This can make the attention distribution more stable, when applying sliding window attention to a model trained with full attention.

## Usage
```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import time
from flash_attn import flash_attn_func

# Prepare some random data
query_length = 10000
query_states = torch.randn(1, query_length, 32, 128, dtype=torch.bfloat16, device="cuda") 
key_states = torch.randn(1, query_length, 4, 128, dtype=torch.bfloat16, device="cuda") 
value_states = torch.randn(1, query_length, 4, 128, dtype=torch.bfloat16, device="cuda") 

# Try different sliding window sizes. -1 means full attention
for sliding_window in [-1,100,1000,4000,8000]:

    start_time = time.time()
    # flash attention forward
    attn_output = flash_attn_func(
        query_states,
        key_states,
        value_states,
        causal=True,
        window_size=(sliding_window, sliding_window), 
        keep_first=4, # for example, we keep the attention to the first 4 tokens
    )

    end_time = time.time()
    time_taken= end_time - start_time
    print(f"Sliding window: {sliding_window}, Time taken: {time_taken:.4f} seconds")

```