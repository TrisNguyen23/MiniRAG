from datasets import load_dataset

def load_and_chunk():
    dataset = load_dataset("wikipedia", "20220301.en", split="train[:0.05%]")
    texts = []
    for article in dataset:
        content = article['text']
        paragraphs = content.split('\n')
        for p in paragraphs:
            if len(p.strip()) > 50:
                texts.append(p.strip())
    return texts

if __name__ == "__main__":
    chunks = load_and_chunk()
    print(f"Loaded {len(chunks)} text chunks")
