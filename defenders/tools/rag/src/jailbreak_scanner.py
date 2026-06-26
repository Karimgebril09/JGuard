
import numpy as np
from defenders.tools.rag.src.rag_pipeline import rag
from defenders.tools.rag.src.chunk_graph_analyzer import ChunkGraphAnalyzer
from defenders.tools.rag.src.patterns import compiled_patterns

class ScanResult:
    def __init__(self, chunk_id,text,is_malicious,matched_patterns):
        self.chunk_id =chunk_id
        self.text =text
        self.is_malicious =is_malicious
        self.matched_patterns =matched_patterns
    

class InjectionScanner:
    """scanner for rag system"""
    def scan_text(self, text,chunk_id=None):
        matched =[]
        for pat in compiled_patterns:
            if pat.search(text):
                matched.append(pat.pattern)
        return ScanResult(
            chunk_id=chunk_id,
            text=text,
            is_malicious=len(matched) > 0,
            matched_patterns=matched,
        )

    def scan_chunks(self,chunks) :
        # scan any defined pattern 
        clean =[]
        for chunk in chunks:
            result =self.scan_text(chunk.text, chunk.chunk_id  )
            if not result.is_malicious:
                clean.append(chunk)
            else :
                print('[scan sanitization removed]'+chunk.text[:100])
            # else :
            #     print(f'[scan] Chunk {chunk.chunk_id} flagged as malicious due to patterns: {result.matched_patterns}')
        return clean
    
    def cosine_similarity(self,vec_a, vec_b):
        
        vec_a =np.array(vec_a).flatten()
        vec_b =np.array(vec_b).flatten()

        dot =np.dot(vec_a,vec_b)
        norm_a =np.linalg.norm(vec_a)
        norm_b =np.linalg.norm(vec_b)

        if norm_a ==0 or norm_b ==0:
            return 0.0

        return float(dot/(norm_a * norm_b))

    def check_similarity(self,chunks, query):
        # check similarity with the query 
        embeddings =rag.embedder.embed_chunks(chunks)
        query_embedding =rag.embedder.embed_query(query)

        query_embedding =np.array(query_embedding).flatten()

        safe_chunks =[]
        safe_embeddings =[]

        for chunk, emb in zip(chunks,embeddings):
            emb =np.array(emb).flatten()
            cos_sim =self.cosine_similarity(emb, query_embedding)
            if cos_sim < 0.5:
                print('[similarity check removed]'+chunk.text[:10])
                continue

            safe_chunks.append(chunk)
            safe_embeddings.append(emb)

        return safe_chunks, safe_embeddings


    def check_consistency(self, chunks,embeddings=None):
        # check if the chunks with each other are similar
        graph_analyzer =ChunkGraphAnalyzer()
        if embeddings is None:
            embeddings =rag.embedder.embed_chunks(chunks)
        g, node_info =graph_analyzer.build_graph(chunks,embeddings)
        not_outliers ={info.chunk.chunk_id for info in node_info if not (info.is_outlier and info.no_neighbors ==0)}
        
        for c in chunks:
            if c.chunk_id not in not_outliers:
                print('[consistency check removed]'+c.text[:100])
        # print (f'[consistency check] {len(not_outliers)}/{len(chunks)} chunks are consistent with others based on graph analysis')
        return [c for c in chunks if c.chunk_id in not_outliers]


    def check_jailbreak(self,chunks,query):
       # go throgh the three stages 
        clean_chunks =self.scan_chunks(chunks)
        if not clean_chunks:
            return []

        similar_chunks, similar_embeddings =self.check_similarity(clean_chunks, query)
        if not similar_chunks:
            return []

        return self.check_consistency(similar_chunks, embeddings=similar_embeddings)