import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from similarity_model import SimilarityModel

class TfiDfModel(SimilarityModel):
    def __init__(self, docs: list[str]):
        super().__init__()
        self.docs = docs
        self.sim_matrix = np.array([])
        self.tfidf = TfidfVectorizer(stop_words='english')

    def calculate_similiarity(self):
        tfidf_matrix = self.tfidf.fit_transform(self.docs)
        self.sim_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)

    def get_recommendations(self, idx: int) -> list[int]:
        sim_scores = list(enumerate(self.sim_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:11]
        movie_indices = [i[0] for i in sim_scores]
        return movie_indices