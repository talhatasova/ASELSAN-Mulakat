"""Download turkish-wikiNER from the Hugging Face Hub and save it locally under data/."""
import sys
from datasets import load_dataset

sys.stdout.reconfigure(encoding="utf-8")  # Windows console cannot print Turkish chars (ı, ş, ğ) otherwise

ds = load_dataset("turkish-nlp-suite/turkish-wikiNER")
print(ds)
print(ds["train"].features)
print(ds["train"][0])
ds.save_to_disk("data/turkish-wikiNER")
