from typing import Union
import numpy as np
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer

from similarity_model import SimilarityModel

class TfiDfModel(SimilarityModel):
    def __init__(self, docs: list[str]):
        super().__init__()
        self.docs = docs
        self.sim_matrix = np.array([])
        self.tfidf = TfidfVectorizer(stop_words='english')

    def calculate_similarity(self):
        tfidf_matrix = self.tfidf.fit_transform(self.docs)
        self.sim_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)        

    def get_recommendations(self, idx: Union[int, list[int]]) -> list[int]:
        if isinstance(idx, int):
            means = self.sim_matrix[idx]
        else:
            means = np.mean(self.sim_matrix[idx], axis=0)
        sim_scores = list(enumerate(means))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        # sim_scores = sim_scores[1:11]
        idx = [idx] if isinstance(idx, int) else idx
        indices = [i[0] for i in sim_scores if i[0] not in idx]
        return indices[:10]