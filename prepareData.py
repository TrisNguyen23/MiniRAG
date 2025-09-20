from datasets import load_dataset

def load_and_chunk():
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:1%]")
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
