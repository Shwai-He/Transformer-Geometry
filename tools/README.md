# Tools & Utilities

This directory contains repository utilities for managing model and dataset synchronization.

## Hugging Face Hub Transfer (`hf_transfer_folders.py`)

`tools/hf_transfer_folders.py` is a unified CLI for uploading local experiment directories or downloading remote snapshots from Hugging Face Hub.

### Requirements

```bash
pip install huggingface_hub
export HF_TOKEN="your_hf_token"
```

### Upload Usage

```bash
python tools/hf_transfer_folders.py upload \
  --repo-id <username>/<repo_name> \
  --repo-type dataset \
  path/to/folder1 path/to/folder2
```

### Download Usage

```bash
python tools/hf_transfer_folders.py download \
  --repo-id <username>/<repo_name> \
  --repo-type dataset \
  --local-dir path/to/local_destination
```
