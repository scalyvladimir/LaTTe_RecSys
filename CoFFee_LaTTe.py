from IPython.utils import io
import numpy as np
from sa_hooi import sa_hooi, get_scaling_weights
import pandas as pd
from scipy.linalg import solve_triangular
from polara.lib.sparse import tensor_outer_at
from tqdm import tqdm 
from polara.evaluation.pipelines import random_grid
from Modules import valid_mlrank, model_evaluate, make_prediction
from evaluation import topn_recommendations, downvote_seen_items

def tf_model_build(config, data, data_description, testset, holdout, attention_matrix):
    userid = data_description["users"]
    itemid = data_description["items"]
    feedback = data_description["feedback"]

    idx = data[[userid, itemid, feedback]].values
    idx[:, -1] = idx[:, -1] - data_description['min_rating'] # works only for integer ratings!
    val = np.ones(idx.shape[0], dtype='f8')
    
    n_users = data_description["n_users"]
    n_items = data_description["n_items"]
    n_ratings = data_description["n_ratings"]
    shape = (n_users, n_items, n_ratings)
    core_shape = config['mlrank']
    num_iters = config["num_iters"]
        
    attention_matrix = np.array(attention_matrix)

    item_popularity = (
        data[itemid]
        .value_counts(sort=False)
        .reindex(range(n_items))
        .fillna(1)
        .values
    )

    scaling_weights = get_scaling_weights(item_popularity, scaling=config["scaling"])

    with io.capture_output() as captured:
        u0, u1, u2 = sa_hooi(
            idx, val, shape, config["mlrank"],
            attention_matrix = attention_matrix,
            scaling_weights = scaling_weights,
            testset = testset,
            holdout = holdout,
            data_description = data_description,
            max_iters = config["num_iters"],
            parallel_ttm = True,
            randomized = config["randomized"],
            growth_tol = config["growth_tol"],
            seed = config["seed"],
            iter_callback = None,
        )
    
    return u0, u1, u2, attention_matrix

def tf_scoring(params, data, data_description, context=["3+4+5"]):
    user_factors, item_factors, feedback_factors, attention_matrix = params
    userid = data_description["users"]
    itemid = data_description["items"]
    feedback = data_description["feedback"]

    data = data.sort_values(userid) 
    data_new = data.assign(
        userid = pd.factorize(data['userid'])[0]
    )
    useridx = data_new[userid]
    itemidx = data_new[itemid].values
    ratings = data_new[feedback].values
    ratings = ratings - data_description['min_rating'] # NEW
    
    n_users = useridx.nunique()
    n_items = data_description['n_items']
    n_ratings = data_description['n_ratings']
    
    inv_attention = solve_triangular(attention_matrix, np.eye(n_ratings), lower=True)
    
    tensor_outer = tensor_outer_at('cpu')
    matrix_softmax = inv_attention.T @ feedback_factors
    #
    if (n_ratings == 10):
        coef = 2
    else:
        coef = 1
        
    if (context == "5"):
        inv_aT_feedback = matrix_softmax[(-1 * coef) , :]
    elif (context == "4+5"):
        inv_aT_feedback = np.sum(matrix_softmax[(-2 * coef):, :], axis=0)
    elif (context == "3+4+5"):
        inv_aT_feedback = np.sum(matrix_softmax[(-3 * coef):, :], axis=0)
    elif (context == "3+4+5-2-1"):
        inv_aT_feedback = np.sum(matrix_softmax[(-3 * coef):, :], axis=0) - np.sum(matrix_softmax[:(2 * coef), :], axis=0)
        
    scores = tensor_outer(
        1.0,
        item_factors,
        attention_matrix @ feedback_factors,
        itemidx,
        ratings
    )
    scores = np.add.reduceat(scores, np.r_[0, np.where(np.diff(useridx))[0]+1])
    scores = np.tensordot(
        scores,
        inv_aT_feedback,
        axes=(2, 0)
    ).dot(item_factors.T)

    return scores


def full_pipeline(config, training, data_description, testset_valid, holdout_valid, testset, holdout, attention_matrix, factor=None):

    config["mlrank"] = (64, 64, data_description["n_ratings"])
    print("Starting pipeline...")
    print(f"Tuning model for all contexts...\n")

    rank_grid = []
    for i in range(5, 9):
        rank_grid.append(2 * 2 ** i)
        rank_grid.append(3 * 2 ** i)
    
    rank_grid = np.array(rank_grid)
    tf_hyper = {
    'scaling': [factor] if factor else np.linspace(0, 2, 21),
    'r1': rank_grid,
    'r3': range(2, 6, 1) if data_description["n_ratings"] == 5 else range(2, 11, 2)
    }

    grid, param_names = random_grid(tf_hyper, n=0)
    tf_grid = [tuple(mlrank) for mlrank in grid if valid_mlrank(mlrank)]

    hr_tf = {}
    hr_pos_tf = {}
    hr_neg_tf = {}
    mrr_tf = {}
    mrr_pos_tf = {}
    mrr_neg_tf = {}
    cov_tf = {}
    C_tf = {}
    
    seen_data = testset_valid
    
    for mlrank in tqdm(tf_grid):
        with io.capture_output() as captured:
            r1, r3 = mlrank[1:]
            cur_mlrank = tuple((r1, r1, r3))
            config['mlrank'] = cur_mlrank
            config['scaling'] = mlrank[0]
            tf_params = tf_model_build(config, training, data_description, testset_valid, holdout_valid, attention_matrix=attention_matrix)
            for context in ["5", "4+5", "3+4+5", "3+4+5-2-1"]:
                tf_scores = tf_scoring(tf_params, seen_data, data_description, context)
                downvote_seen_items(tf_scores, seen_data, data_description)
                tf_recs = topn_recommendations(tf_scores, topn=10)
                
                hr, hr_pos, hr_neg, mrr, mrr_pos, mrr_neg, cov, C = model_evaluate(tf_recs, holdout_valid, data_description, topn=10)
                hr_tf[(context, cur_mlrank, mlrank[0])] = hr
                hr_pos_tf[(context, cur_mlrank, mlrank[0])] = hr_pos
                hr_neg_tf[(context, cur_mlrank, mlrank[0])] = hr_neg
                mrr_tf[(context, cur_mlrank, mlrank[0])] = mrr
                mrr_pos_tf[(context, cur_mlrank, mlrank[0])] = mrr_pos
                mrr_neg_tf[(context, cur_mlrank, mlrank[0])] = mrr_neg
                cov_tf[(context, cur_mlrank, mlrank[0])] = cov
                C_tf[(context, cur_mlrank, mlrank[0])] = C

    print(f'Best HR={pd.Series(hr_tf).max():.4f} achieved with context {pd.Series(hr_tf).idxmax()[0]} and mlrank = {pd.Series(hr_tf).idxmax()[1]} and scale factor = {pd.Series(hr_tf).idxmax()[2]}')
    print(f'Best HR_pos={pd.Series(hr_pos_tf).max():.4f} achieved with context {pd.Series(hr_pos_tf).idxmax()[0]} and mlrank = {pd.Series(hr_pos_tf).idxmax()[1]} and scale factor = {pd.Series(hr_pos_tf).idxmax()[2]}')
    print(f'Best HR_neg={pd.Series(hr_neg_tf).min():.4f} achieved with context {pd.Series(hr_neg_tf).idxmin()[0]} and mlrank = {pd.Series(hr_neg_tf).idxmin()[1]} and scale factor = {pd.Series(hr_neg_tf).idxmin()[2]}')
    
    print(f'Best MRR={pd.Series(mrr_tf).max():.4f} achieved with context {pd.Series(mrr_tf).idxmax()[0]} and mlrank = {pd.Series(mrr_tf).idxmax()[1]} and scale factor = {pd.Series(mrr_tf).idxmax()[2]}')
    print(f'Best MRR_pos={pd.Series(mrr_pos_tf).max():.4f} achieved with context {pd.Series(mrr_pos_tf).idxmax()[0]} and mlrank = {pd.Series(mrr_pos_tf).idxmax()[1]} and scale factor = {pd.Series(mrr_pos_tf).idxmax()[2]}')
    print(f'Best MRR_neg={pd.Series(mrr_neg_tf).min():.4f} achieved with context {pd.Series(mrr_neg_tf).idxmin()[0]} and mlrank = {pd.Series(mrr_neg_tf).idxmin()[1]} and scale factor = {pd.Series(mrr_neg_tf).idxmin()[2]}')
    
    print(f'Best Matthews={pd.Series(C_tf).max():.4f} achieved with context {pd.Series(C_tf).idxmax()[0]} and mlrank = {pd.Series(C_tf).idxmax()[1]} and scale factor = {pd.Series(C_tf).idxmax()[2]}')
                          
    print(f'COV={pd.Series(cov_tf)[pd.Series(C_tf).idxmax()]:.4f} (based on best Matthews value)')
    print("---------------------------------------------------------")
    print("Evaluation of the best model on test holdout in progress...\n")
    
    print("Best by MRR@10:\n")
    config["mlrank"] = pd.Series(mrr_pos_tf).idxmax()[1]
    tf_params = tf_model_build(config, training, data_description, testset, holdout, attention_matrix=attention_matrix)

    seen_data = testset
    tf_scores = tf_scoring(tf_params, seen_data, data_description, pd.Series(mrr_pos_tf).idxmax()[0])
    downvote_seen_items(tf_scores, seen_data, data_description)
    cur_mrr, cur_hr, cur_C = make_prediction(tf_scores, holdout, data_description, "Test", pd.Series(mrr_pos_tf).idxmax()[0])
    
    print("---------------------------------------------------------")
    
    print("Best by HR@10:\n")
    config["mlrank"] = pd.Series(hr_pos_tf).idxmax()[1]
    tf_params = tf_model_build(config, training, data_description, testset, holdout, attention_matrix=attention_matrix)

    seen_data = testset
    tf_scores = tf_scoring(tf_params, seen_data, data_description, pd.Series(hr_pos_tf).idxmax()[0])
    downvote_seen_items(tf_scores, seen_data, data_description)
    cur_mrr, cur_hr, cur_C = make_prediction(tf_scores, holdout, data_description, "Test", pd.Series(hr_pos_tf).idxmax()[0])
    
    print("---------------------------------------------------------")
    
    print("Best by Matthews@10:\n")
    config["mlrank"] = pd.Series(C_tf).idxmax()[1]
    tf_params = tf_model_build(config, training, data_description, testset, holdout, attention_matrix=attention_matrix)

    seen_data = testset
    tf_scores = tf_scoring(tf_params, seen_data, data_description, pd.Series(C_tf).idxmax()[0])
    downvote_seen_items(tf_scores, seen_data, data_description)
    cur_mrr, cur_hr, cur_C = make_prediction(tf_scores, holdout, data_description, "Test", pd.Series(C_tf).idxmax()[0])
    print("Pipeline ended.")

def sigmoid_func(x):
    return 1.0 / (1 + np.exp(-x))

def arctan(x):
    return 0.5 * np.arctan(x) + 0.5

def sq3(x):
    return 0.5 * np.cbrt(x) + 0.5

def get_similarity_matrix(mode, n_ratings = 10):
    matrix = np.zeros((n_ratings, n_ratings))
    if (mode == "sigmoid"):
        x_space = np.linspace(-6, 6, n_ratings)
        for i in range(n_ratings):
            for j in range(i, n_ratings, 1):
                matrix[i, j] = 1.0 - np.abs(sigmoid_func(x_space[i]) - sigmoid_func(x_space[j]))
                matrix[j, i] = matrix[i, j]
                
    elif (mode == "linear"):
        x_space = np.linspace(0, 1, n_ratings)
        for i in range(n_ratings):
            for j in range(i, n_ratings, 1):
                matrix[i, j] = 1.0 - np.abs(x_space[i] - x_space[j])
                matrix[j, i] = matrix[i, j]
                
    elif (mode == "arctan"):
        x_space = np.linspace(-np.pi / 2.0, np.pi / 2.0, n_ratings)
        for i in range(n_ratings):
            for j in range(i, n_ratings, 1):
                matrix[i, j] = 1.0 - np.abs(arctan(x_space[i]) - arctan(x_space[j]))
                matrix[j, i] = matrix[i, j]
                
    elif (mode == "sq3"):
        x_space = np.linspace(-1, 1, n_ratings)
        for i in range(n_ratings):
            for j in range(i, n_ratings, 1):
                matrix[i, j] = 1.0 - np.abs(sq3(x_space[i]) - sq3(x_space[j]))
                matrix[j, i] = matrix[i, j]
                
    return matrix
