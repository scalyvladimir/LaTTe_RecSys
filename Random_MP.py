import numpy as np

def build_random_model(trainset, trainset_description):
    itemid = trainset_description['items']
    n_items = trainset[itemid].max() + 1
    random_state = np.random.RandomState(42)
    return n_items, random_state

def random_model_scoring(params, testset, testset_description):
    n_items, random_state = params
    n_users = testset_description['n_test_users']
    scores = random_state.rand(n_users, n_items)
    return scores

def simple_model_recom_func(scores, topn=20):
    recommendations = np.apply_along_axis(topidx, 1, scores, topn)
    return recommendations

def topidx(a, topn):
    parted = np.argpartition(a, -topn)[-topn:]
    return parted[np.argsort(-a[parted])]

def build_popularity_model(trainset, trainset_description):
    itemid = trainset_description['items']
    item_popularity = trainset[itemid].value_counts()
    return item_popularity

def popularity_model_scoring(params, testset, testset_description):
    item_popularity = params
    n_items = item_popularity.index.max() + 1
    n_users = testset_description['n_test_users']
    # fill in popularity scores for each item with indices from 0 to n_items-1
    popularity_scores = np.zeros(n_items,)
    popularity_scores[item_popularity.index] = item_popularity.values
    # same scores for each test user
    scores = np.tile(popularity_scores, n_users).reshape(n_users, n_items)
    return scores
