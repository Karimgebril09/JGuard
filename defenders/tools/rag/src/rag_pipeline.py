from defenders.tools.rag.src.chunker import DocumentChunker
from defenders.tools.rag.src.vector_store import VectorStore
from defenders.tools.rag.src.embedder import Embedder


class RAGPipeline:

    def __init__(self, chunk_size=300, overlap=50, embed_model="tfidf-svd-128", top_k=5):

        self.chunker = DocumentChunker(chunk_size, overlap)

        self.embedder = Embedder(embed_model)

        self.store = None

        self.top_k = top_k

        self._all_chunks = []

    def ingest_many(self, docs):

        new_chunks = []

        # 1. Chunk all documents
        for source in docs:

            text = docs[source]

            chunks = self.chunker.chunk_text(text, source)

            print("[ingest]", len(chunks), "chunks from", source)

            for chunk in chunks:
                new_chunks.append(chunk)

        self._all_chunks.extend(new_chunks)

 
        embeddings = self.embedder.embed_chunks(self._all_chunks)

        # 3. Create vector store
        self.store = VectorStore(self.embedder.dim)
        # 4. Store everything
        self.store.add(self._all_chunks, embeddings)

        print("[ingest] vector size =", self.embedder.dim)

        return len(new_chunks)

    def ingest(self, text, source="inline"):

        return self.ingest_many({source: text})

    def ingest_file(self, path):
        chunks = self.chunker.load_file(path)
        print("[ingest]", len(chunks), "chunks from", path)

        self._all_chunks.extend(chunks)

        embeddings = self.embedder.embed_chunks(self._all_chunks)
        self.store = VectorStore(self.embedder.dim)
        self.store.add(self._all_chunks, embeddings)

        print("[ingest] vector size =", self.embedder.dim)
        return len(chunks)

    def retrieve(self, query):

        if self.store is None:
            raise Exception("No documents added yet")

        query_vector = self.embedder.embed_query(query)

        return self.store.search(query_vector, self.top_k)
    

rag = RAGPipeline(
    chunk_size=100,   
    overlap=20,      
    top_k=4,         
)

rag.ingest_file("../txt/rag_text.txt")