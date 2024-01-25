import pandas as pd


def merge_rankings(
    user_ranking: list[int], description_ranking: list[int], decider: int
):
    merged = pd.Series()
    for id, place in zip(user_ranking, range(len(user_ranking), 0, -1)):
        merged[id] = place * decider
    for id, place in zip(description_ranking, range(len(description_ranking), 0, -1)):
        if id not in merged.keys():
            merged[id] = place * (100 - decider)
        else:
            merged[id] += place * (100 - decider)
    merged = merged.sort_values(ascending=False)
    return merged.keys()
