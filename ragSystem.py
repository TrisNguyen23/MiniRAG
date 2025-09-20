from sentence_transformers import SentenceTransformer
import faiss
from transformers import pipeline

class MiniRAG:
    def __init__(self, texts):
        self.texts = texts
        # Create embeddings
        self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = self.embed_model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        # FAISS index
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.embeddings)
        # Generator
        self.generator = pipeline('text-generation', model='gpt2', max_length=300)

    def retrieve(self, query, top_k=5):
        query_vec = self.embed_model.encode([query])
        distances, indices = self.index.search(query_vec, top_k)
        return [self.texts[i] for i in indices[0]]

    def generate_answer(self, query):
        context_docs = self.retrieve(query)
        context = "\n".join(context_docs)
        prompt = f"Answer the question based on the context below:\n{context}\nQuestion: {query}\nAnswer:"
        output = self.generator(prompt, max_new_tokens=200, do_sample=True)[0]['generated_text']
        answer = output.split('Answer:')[-1].strip()
        return answer

