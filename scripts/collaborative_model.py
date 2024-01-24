import numpy as np
import pandas as pd
from typing import Dict, Union
from scipy import spatial
from scripts.similarity_model import SimilarityModel


class CollaboartiveModel(SimilarityModel):
    def __init__(self, docs: pd.DataFrame):
        super().__init__()
        self.docs = docs
        self.reviews = pd.DataFrame(None, index=0, columns=docs.columns)

    def calculate_similarity(self, idx: Union[int, list[int]]) -> list[int]:
        similiarities = np.zeros_like(self.docs.shape[0])
        for x in self.docs.index:
            similiarities[x] = spatial.distance.cosine(self.docs[x], self.reviews[0])
        return np.argmin(similiarities)

    def get_recommendations(self, idx: Union[Dict[int][int], pd.Series]) -> list[int]:
        for key, val in zip(idx.keys(), idx.values):
            self.reviews[key] = val
        self.reviews = self.reviews.sub(self.reviews.mean(axis=1), axis=0).fillna(0)
        closest_user = self.calculate_similarity()
        closest_reviews = self.docs.iloc[closest_user].squeeze()
        closest_reviews = closest_reviews.drop(labels=idx.keys())
        closest_reviews = closest_reviews.sort_values(ascending=False)
        return closest_reviews.keys()[:10]
