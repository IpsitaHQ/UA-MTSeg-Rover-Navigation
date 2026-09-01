# Dataset

This project uses the BASEPROD dataset hosted on Hugging Face.

The dataset is **not stored in this repository** because of its size.

It is imported automatically using the Hugging Face `datasets` library.

Example:
```python
from datasets import load_dataset
dataset = load_dataset("your_huggingface_dataset_name")

The dataset is saved as: 
Run this once in PowerShell:
setx HF_TOKEN "your_token_here"

setx (unlike $env:) writes it into your Windows user environment variables permanently. You'll need to close and reopen PowerShell/VS Code/Jupyter for it to take effect, but after that every new session will already have HF_TOKEN set — no more pasting.