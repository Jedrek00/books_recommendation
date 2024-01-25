import numpy as np
import pandas as pd
from typing import Dict, Union
from scipy import spatial

from .similarity_model import SimilarityModel


class CollaborativeModel(SimilarityModel):
    def __init__(self, docs: pd.DataFrame):
        super().__init__()
        self.docs = docs
        self.reviews = pd.DataFrame(None, index=[0], columns=docs.columns)

    def calculate_similarity(self) -> int:
        similiarities = np.zeros(self.docs.shape[0])
        for x in self.docs.index:
            similiarities[x] = spatial.distance.cosine(self.docs.loc[x].tolist(), self.reviews.loc[0].tolist())
        return np.argmin(similiarities)

    def get_recommendations(self, idx: Union[Dict[str, int], pd.Series]) -> list[int]:
        if type(idx) == dict:
            idx = pd.Series(idx)
        for key, val in zip(idx.keys(), idx.values):
            self.reviews.loc[0, str(key)] = val
        self.reviews = self.reviews.sub(self.reviews.mean(axis=1), axis=0).fillna(0)
        closest_user = self.calculate_similarity()
        closest_reviews = self.docs.iloc[closest_user].squeeze()
        for key in idx.keys():
            if str(key) in closest_reviews.keys():
                closest_reviews = closest_reviews.drop(labels = str(key))
        closest_reviews = closest_reviews.sort_values(ascending=False)
        return [int(key) for key in closest_reviews.keys()[:10]]
    
    def save_sim_matrix(self, save_dir: str) -> None:
        return None
