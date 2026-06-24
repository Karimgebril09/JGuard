from defenders.multi_turn.NBF.inference.ssm_feature_extractor import StateFeatureExtractor
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
import pandas as pd

def test_embedding_shape(feature_extractor):
    embedding = feature_extractor.embed_input("Hello, how are you?")
    assert embedding.dim() ==2
    assert embedding.shape[1] == feature_extractor.embedding_model.get_sentence_embedding_dimension()


def test_history_management(feature_extractor):
    # Test that the history is managed correctly
    user_ip = feature_extractor.embed_input("Hello, how are you?")
    x_first = feature_extractor.ssm_model.get_next_state(user_ip)
    tmp = feature_extractor.get_long_term_state(x_first)

    assert len(feature_extractor.q) == 1
    assert torch.equal(feature_extractor.q[0], x_first)

    # Add more states to fill the queue
    for i in range(3):
        user_ip = feature_extractor.embed_input(f"Message {i}")
        x_curr = feature_extractor.ssm_model.get_next_state(user_ip)
        feature_extractor.get_long_term_state(x_curr)

    assert len(feature_extractor.q) == 4
    assert torch.equal(feature_extractor.q[0], x_first)

    user_ip = feature_extractor.embed_input(f"Message {i}")
    x_curr = feature_extractor.ssm_model.get_next_state(user_ip)
    feature_extractor.get_long_term_state(x_curr) 

    # make sure its still 4
    assert len(feature_extractor.q) == 4   




def test_similarity(feature_extractor):
    x = torch.tensor([[30.0, 6.0]])
    y = torch.tensor([[30.0, 6.0]])
    similarity = feature_extractor.get_similarity(x, y)
    assert np.isclose(similarity, 1.0, atol=1e-6)
    torch_similarity = torch.nn.functional.cosine_similarity(x, y).item()
    assert np.isclose(similarity, torch_similarity, atol=1e-6)

    y = torch.tensor([[-2.0, 3.0]])
    similarity = feature_extractor.get_similarity(x, y)
    assert similarity < 1.0
    torch_similarity = torch.nn.functional.cosine_similarity(x, y).item()
    assert np.isclose(similarity, torch_similarity, atol=1e-6)

def test_dimensionality_reduction(feature_extractor):
    dummy_vector = torch.zeros(1,1536)
    reduced_vector = feature_extractor.reduce_dimensionality(dummy_vector)
    assert reduced_vector.shape[1] == 390


def test_transformations(feature_extractor):
    data = [
        {'col1': 0.0, 'col2': 4.0, 'col3': 9.0},
        {'col1': 1.0, 'col2': 16.0, 'col3': 25.0},
    ]
    df = pd.DataFrame(data)
    transformed_selected_features = ['col1_log','col2_sqrt','col3_orig']
    transformed_df = feature_extractor.apply_selected_transformations(df, transformed_selected_features)

    # Check if the transformations are applied correctly
    assert np.allclose(transformed_df['col1_log'], np.log(df["col1"] + 1 + 1e-6))
    assert np.allclose(transformed_df['col2_sqrt'], np.sqrt(df["col2"] + 1 + 1e-6))
    assert np.allclose(transformed_df['col3'], df['col3'])





def run_tests():
    embedding_model=SentenceTransformer("all-mpnet-base-v2")
    feature_extractor = StateFeatureExtractor(embedding_model)
    test_embedding_shape(feature_extractor)
    test_history_management(feature_extractor)
    test_similarity(feature_extractor)
    test_dimensionality_reduction(feature_extractor)
    test_transformations(feature_extractor)
    print("All tests passed.")

if __name__ == "__main__":
    run_tests()