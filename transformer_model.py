from typing import Optional
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util


from similarity_model import SimilarityModel

class TransformerModel(SimilarityModel):
    def __init__(self, docs: list[str], model_name: str):
        super().__init__()
        self.docs = docs
        self.sim_matrix = np.array([])

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"{self.device} will be used.")
        self.model = SentenceTransformer(model_name, device=self.device)

    def calculate_similarity(self, pretrained_path: Optional[str] = None):
        self.sim_matrix = self.model.encode(self.docs, device=self.device, show_progress_bar=True)

    def get_recommendations(self, idx: int) -> list[int]:
        query_embedding = self.sim_matrix[idx]
        cosine_scores = util.cos_sim(query_embedding, self.sim_matrix)
        similar_books_indices = cosine_scores.argsort()
        indices = similar_books_indices[0].numpy()[::-1][1:11]
        return indices
    
    def save_model(self, path: str):
        self.model.save(path)
        