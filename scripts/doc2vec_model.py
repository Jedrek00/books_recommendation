import os
import numpy as np
from typing import Optional, Union
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import linear_kernel
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from .similarity_model import SimilarityModel


class Doc2VecModel(SimilarityModel):
    def __init__(self, docs: list[str], vector_size: int, alpha: float, min_count: int, epochs: int):
        super().__init__()
        self.docs = docs
        self.sim_matrix = np.array([])
        self.matrix_name = "doc2vec_sim_matrix.npy"
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

    def get_recommendations(self, idx: Union[int, list[int]]) -> list[int]:
        if isinstance(idx, int):
            means = self.sim_matrix[idx]
        else:
            means = np.mean(self.sim_matrix[idx], axis=0)
        sim_scores = list(enumerate(means))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        idx = [idx] if isinstance(idx, int) else idx
        indices = [i[0] for i in sim_scores if i[0] not in idx]
        return indices[:10]

    def save_model(self, path: str):
        self.model.save(path)

    def save_sim_matrix(self, save_dir: str) -> None:
        with open(os.path.join(save_dir, self.matrix_name), 'wb') as f:
            np.savez_compressed(f, self.sim_matrix)
