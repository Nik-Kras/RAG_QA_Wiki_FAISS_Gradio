from datasets import load_dataset
import unicodedata
import re

print("📥 Loading Wikipedia dataset...")
dataset = load_dataset("not-lain/wikipedia", split="train")  # sample for demo

# 1. Remove text after "may also refer to:"
def trim_refer_to(text):
    match = re.search(r"(may also refer to:)", text, re.IGNORECASE)
    if match:
        return text[:match.start()].strip()
    return text

dataset = dataset.map(lambda x: {"text": trim_refer_to(x["text"])})

# 2. Clean and normalize text
def clean_text(text):
    # Normalize unicode
    text = unicodedata.normalize("NFKC", text)

    # Remove Wikipedia section headings like '== Heading =='
    text = re.sub(r'==+[^=]+==+', '', text)

    # Remove citation brackets like [1], [12], [citation needed]
    text = re.sub(r'\[[^\]]*\]', '', text)

    # Remove non-ASCII characters (can keep emojis or foreign symbols if needed)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)

    # Remove URLs and file/image links
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'File:[^\s]+', '', text)

    # Remove excessive whitespace, tabs, and newlines
    text = re.sub(r'\s+', ' ', text)

    # Remove leading/trailing whitespace
    text = text.strip()

    return text

dataset = dataset.map(lambda x: {"text": clean_text(x["text"])})

print(f"✅ Cleaned dataset contains {len(dataset)} articles.")
