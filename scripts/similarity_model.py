from abc import ABC, abstractmethod

class SimilarityModel(ABC):
    @abstractmethod
    def calculate_similarity(self) -> None:
        pass

    @abstractmethod
    def get_recommendations(self, idx: int) -> list[int]:
        pass

    @abstractmethod
    def save_sim_matrix(self, save_dir: str) -> None:
        pass