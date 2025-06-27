import re
from pathlib import Path

CHUNK_SIZE = 100  # Approx. number of words per chunk
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def clean_text(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def chunk_text(text, chunk_size=CHUNK_SIZE):
    words = text.split()
    return [
        " ".join(words[i:i + chunk_size])
        for i in range(0, len(words), chunk_size)
    ]

def process_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    text = clean_text(text)
    return chunk_text(text)

def save_chunks(filename, chunks):
    base_name = filename.stem
    for i, chunk in enumerate(chunks):
        out_path = PROCESSED_DIR / f"{base_name}_chunk{i}.txt"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(chunk)

def run_preprocessing():
    for file in RAW_DIR.glob("*.txt"):
        chunks = process_file(file)
        save_chunks(file, chunks)
        print(f"Processed {file.name} into {len(chunks)} chunks.")

if __name__ == "__main__":
    run_preprocessing()
