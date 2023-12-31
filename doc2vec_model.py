import numpy as np
from typing import Optional
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import linear_kernel
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from similarity_model import SimilarityModel


class Doc2VecModel(SimilarityModel):
    def __init__(self, docs: list[str], vector_size: int, alpha: float, min_count: int, epochs: int):
        super().__init__()
        self.docs = docs
        self.sim_matrix = np.array([])
        self.model = Doc2Vec(
            vector_size=vector_size, alpha=alpha, min_count=min_count, epochs=epochs
        )

    def calculate_similarity(self, pretrained_path: Optional[str] = None):
        if pretrained_path is not None:
            self.model = Doc2Vec.load(pretrained_path)
        else:
            train_doc2vec = [
                TaggedDocument((word_tokenize(s)), tags=[idx])
                for idx, s in enumerate(self.docs)
            ]
            self.model.build_vocab(train_doc2vec)
            self.model.train(
                train_doc2vec,
                total_examples=self.model.corpus_count,
                epochs=self.model.epochs,
            )
        matrix = np.array([self.model.dv[key] for key in range(len(self.docs))])
        self.sim_matrix = linear_kernel(matrix, matrix)

    def get_recommendations(self, idx: int) -> list[int]:
        sim_scores = list(enumerate(self.sim_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:11]
        indices = [i[0] for i in sim_scores]
        return indices

    def save_model(self, path: str):
        self.model.save(path)
