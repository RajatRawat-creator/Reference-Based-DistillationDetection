# Training: controlled student SFT

Two scripts, one per student family. Each scans `SFT_DATASETS_DIR` for
`*.jsonl` files (each containing `question` / `response` records) and SFTs
the configured students on every dataset found.

| Script                     | Students                                     | Template                            |
|----------------------------|----------------------------------------------|-------------------------------------|
| `train_qwen.py`            | Qwen-2.5-1.5B, Qwen-2.5-3B                   | Plain text "Problem:\\n... Solution:\\n..." |
| `train_llama_gemma.py`     | Llama-3.2-3B-Instruct (commented), Gemma-3-4B-PT | Chat template (Llama), plain text (Gemma) |

## Required env

- `HF_TOKEN`: needed for gated models (Llama, Gemma).
- `SFT_DATASETS_DIR` (optional): override the default `<repo>/data/SFTDatasets`.
- `SFT_OUTPUT_DIR` / `SFT_OUTPUT_DIR_LLAMA_GEMMA` (optional): override
  default checkpoint roots under `<repo>/checkpoints/...`.

## Hyperparameters

Same across both scripts (edit constants at the top to change):

```
block_size              = 4096
learning_rate           = 1e-5
num_train_epochs        = 3
per_device_batch_size   = 4
gradient_accumulation   = 4
warmup_ratio            = 0.05
lr_scheduler            = cosine
```

## Submitting on Slurm

The `.sh` files are Slurm sbatch wrappers configured for a single H200.
Make sure `HF_TOKEN` is exported in the submitting shell, then:

```bash
sbatch training/train_qwen.sh
sbatch training/train_llama_gemma.sh
```

Outputs land in `OUTPUT_BASE_DIR/Student=<name>/Student=<name>_<jsonl_basename>/`.
