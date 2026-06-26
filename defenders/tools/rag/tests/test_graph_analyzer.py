
from defenders.tools.rag.src.chunker import Chunk
import numpy as np
from defenders.tools.rag.src.chunk_graph_analyzer import ChunkGraphAnalyzer



class TestChunkGraphAnalyzer:
    #only help for create dummy chnks
    def _identical_embeddings(self,n=4, dim=8):
        base=np.random.rand(dim).tolist()
        return [Chunk(chunk_id=str(i), text=f"chunk {i}", source='',start_word=i, end_word=i+1) for i in range(n)], [base[:] for _ in range(n)]
    
    def _orthogonal_embeddings(self):
        #help for create dummy chunks orthogonal
        
        chunks=[Chunk(chunk_id="0",text="a", source='', start_word=0,end_word=1), Chunk(chunk_id="1",text="b", source='', start_word=1, end_word=2)]
        embs=[[1.0, 0.0],[0.0, 1.0]]
        return chunks, embs
    
    def test_graph_has_entry_per_chunk(self):
        #see if the graph has an entry for each chunk
        
        analyzer=ChunkGraphAnalyzer()
        chunks,embs=self._identical_embeddings(3)
        graph, _=analyzer.build_graph(chunks, embs)
        assert set(graph.keys())=={0, 1, 2}  #see if all chunks have entry in graph
 
    def test_identical_chunks_all_connected(self):
        #see if identical chunks are all connected
        
        analyzer=ChunkGraphAnalyzer(similarity_threshold=0.9)
        chunks,embs=self._identical_embeddings(3)
        graph, _=analyzer.build_graph(chunks,embs)
        for i in range(3):
            assert len(graph[i])==2  # check for connectivity
 
    def test_orthogonal_chunks_not_connected(self):
        #see if orthogonal chunks are not connected
        
        analyzer=ChunkGraphAnalyzer(similarity_threshold=0.5)
        chunks, embs=self._orthogonal_embeddings()
        graph, _=analyzer.build_graph(chunks,embs)
        assert graph[0]==[] and graph[1]==[] # check for no connectivity
 
 
    def test_similar_chunks_not_outliers(self):
        #see if similar chunks are not flagged as outliers
    
        analyzer=ChunkGraphAnalyzer(outlier_threshold=0.1)
        chunks, embs=self._identical_embeddings(4)
        _, node_info=analyzer.build_graph(chunks, embs)
        assert all(not n.is_outlier for n in node_info) #should not be outlier in here
 
    def test_outlier_chunk_flagged(self):
        #see if an outlier chunk is flagged
        
        chunks=[Chunk(chunk_id="0",text="a",source='',start_word=0, end_word=1), Chunk(chunk_id="1",text="b", source='', start_word=1,end_word=2),Chunk(chunk_id="2", text="c", source='',start_word=2, end_word=3)]
        embs=[[1.0, 0.0],[0.9, 0.1],[0.0, 0.0]]  # last is zero vector → sim=0
        analyzer=ChunkGraphAnalyzer(outlier_threshold=0.3)
        _, node_info=analyzer.build_graph(chunks, embs) 
       
        assert node_info[2].is_outlier #as the emb show diff away pattern

if __name__=="__main__":
    test_analyzer=TestChunkGraphAnalyzer()
    test_analyzer.test_graph_has_entry_per_chunk()
    test_analyzer.test_identical_chunks_all_connected()
    test_analyzer.test_orthogonal_chunks_not_connected()
    test_analyzer.test_similar_chunks_not_outliers()
    test_analyzer.test_outlier_chunk_flagged()
