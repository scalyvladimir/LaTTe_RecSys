import warnings
from functools import wraps
from itertools import takewhile, count, islice

import numpy as np
from scipy.sparse.linalg import svds
from scipy.sparse import diags, SparseEfficiencyWarning
from scipy.linalg import solve_banded

from polara.lib.sparse import arrange_indices
from polara.lib.tensor import ttm3d_seq, ttm3d_par

try:
    from sklearn.utils.extmath import randomized_svd
except ImportError:
    randomized_svd = None
    
from evaluation import topn_recommendations, downvote_seen_items
from scipy.linalg import solve_triangular
from polara.lib.sparse import tensor_outer_at
from scipy.sparse import csr_matrix
from numba import jit

class SeqTFError(Exception):
    pass

def initialize_columnwise_orthonormal(dims, random_state=None):
    if random_state is None:
        random_state = np.random
    u = random_state.rand(*dims)
    u = np.linalg.qr(u, mode='reduced')[0]
    return u

def tf_scoring(params, data, data_description, context=["3+4+5"]):
    user_factors, item_factors, feedback_factors, attention_matrix = params
    userid = data_description["users"]
    itemid = data_description["items"]
    feedback = data_description["feedback"]
    
    
    data = data.sort_values(userid)
    useridx = data[userid]
    itemidx = data[itemid].values
    ratings = data[feedback].values
    
    ratings = ratings - data_description['min_rating']
    
    n_users = useridx.nunique()
    n_items = data_description['n_items']
    n_ratings = data_description['n_ratings']
    
    inv_attention = solve_triangular(attention_matrix, np.eye(data_description['n_ratings']), lower=True) # NEW
    
    tensor_outer = tensor_outer_at('cpu')
    matrix_softmax = inv_attention.T @ feedback_factors
    
    if (n_ratings == 10):
        coef = 2
    else:
        coef = 1
        
    if (context == "5"): # make softmax 
        inv_aT_feedback = matrix_softmax[(-1 * coef) , :]
    elif (context == "4+5"):
        inv_aT_feedback = np.sum(matrix_softmax[(-2 * coef):, :], axis=0)
    elif (context == "3+4+5"):
        inv_aT_feedback = np.sum(matrix_softmax[(-3 * coef):, :], axis=0)
    #elif (context == "2+3+4+5"):
    #    inv_aT_feedback = np.sum(matrix_softmax[-4:, :], axis=0)
    elif (context == "3+4+5-2-1"):
        inv_aT_feedback = np.sum(matrix_softmax[(-3 * coef):, :], axis=0) - np.sum(matrix_softmax[:(2 * coef), :], axis=0)
        
    scores = tensor_outer(
        1.0,
        item_factors,
        attention_matrix @ feedback_factors,
        itemidx,
        ratings
    )
    scores = np.add.reduceat(scores, np.r_[0, np.where(np.diff(useridx))[0]+1]) # sort by users
    scores = np.tensordot(
        scores,
        inv_aT_feedback,
        axes=(2, 0)
    ).dot(item_factors.T)

    return scores

def model_evaluate(recommended_items, holdout, holdout_description, alpha=3, topn=10, dcg=False):
    itemid = holdout_description['items']
    rateid = holdout_description['feedback']
    holdout_items = holdout[itemid].values
    alpha = 3 if holdout_description["n_ratings"] == 5 else 6
    n_test_users = recommended_items.shape[0]
    assert recommended_items.shape[0] == len(holdout_items)
    
    hits_mask = recommended_items[:, :topn] == holdout_items.reshape(-1, 1)
    pos_mask = (holdout[rateid] >= alpha).values
    neg_mask = (holdout[rateid] < alpha).values
    
    # HR calculation
    #hr = np.sum(hits_mask.any(axis=1)) / n_test_users
    hr_pos = np.sum(hits_mask[pos_mask].any(axis=1)) / n_test_users
    hr_neg = np.sum(hits_mask[neg_mask].any(axis=1)) / n_test_users
    hr = hr_pos + hr_neg
    
    # MRR calculation
    hit_rank = np.where(hits_mask)[1] + 1.0
    mrr = np.sum(1 / hit_rank) / n_test_users
    pos_hit_rank = np.where(hits_mask[pos_mask])[1] + 1.0
    mrr_pos = np.sum(1 / pos_hit_rank) / n_test_users
    neg_hit_rank = np.where(hits_mask[neg_mask])[1] + 1.0
    mrr_neg = np.sum(1 / neg_hit_rank) / n_test_users
    
    # Matthews correlation
    TP = np.sum(hits_mask[pos_mask]) # + 
    FP = np.sum(hits_mask[neg_mask]) # +
    
    cond = (hits_mask.sum(axis = 1) == 0)
    FN = np.sum(cond[pos_mask])
    TN = np.sum(cond[neg_mask])
    N = TP+FP+TN+FN
    S = (TP+FN)/N
    P = (TP+FP)/N
    C = (TP/N - S*P) / np.sqrt(P*S*(1-P)*(1-S))
    
    # DCG calculation
    if dcg:
        pos_hit_rank = np.where(hits_mask[pos_mask])[1] + 1.0
        neg_hit_rank = np.where(hits_mask[neg_mask])[1] + 1.0
        ndcg = np.mean(1 / np.log2(pos_hit_rank+1))
        ndcl = np.mean(1 / np.log2(neg_hit_rank+1))
    
    # coverage calculation
    n_items = holdout_description['n_items']
    cov = np.unique(recommended_items).size / n_items
    if dcg:
        return hr, hr_pos, hr_neg, mrr, mrr_pos, mrr_neg, cov, C, ndcg, ndcl
    else:
        return hr, hr_pos, hr_neg, mrr, mrr_pos, mrr_neg, cov, C

def make_prediction(tf_scores, holdout, data_description, mode, context="", print_mode=True):
    if (mode and print_mode):
        print(f"for context {context} evaluation ({mode}): \n")
    for n in [5, 10, 20]:
        tf_recs = topn_recommendations(tf_scores, n)
        hr, hr_pos, hr_neg, mrr, mrr_pos, mrr_neg, cov, C = model_evaluate(tf_recs, holdout, data_description, topn=n)
        if (print_mode):
            print(f"HR@{n} = {hr:.4f}, MRR@{n} = {mrr:.4f}, Coverage@{n} = {cov:.4f}")
            print(f"HR_pos@{n} = {hr_pos:.4f}, HR_neg@{n} = {hr_neg:.4f}")
            print(f"MRR_pos@{n} = {mrr_pos:.4f}, MRR_neg@{n} = {mrr_neg:.4f}")
            print(f"Matthews@{n} = {C:.4f}")
            print("-------------------------------------")
        if (n == 10):
            mrr10 = mrr
            hr10 = hr
            c10 = C
    return mrr10, hr10, c10

def core_growth_callback(growth_tol):
    def check_core_growth(step, core_norm, factors, tf_params, best_metric, testset, holdout, data_description):
        g_growth = (core_norm - check_core_growth.core_norm) / core_norm
        check_core_growth.core_norm = core_norm
        print(f'growth of the core: {g_growth}')
        tf_scores = tf_scoring(tf_params, testset, data_description, "4+5")
        downvote_seen_items(tf_scores, testset, data_description)
        cur_mrr, cur_hr, cur_C = make_prediction(tf_scores, holdout, data_description, "Validation", "4+5", print_mode=False)
        if (cur_C <= best_metric):
            print(f'Metric is no more growing. Best metric: {best_metric}.')
            raise StopIteration
        else:
            return cur_C
        if g_growth < growth_tol:
            print(f'Core is no longer growing. Norm of the core: {core_norm}.')
            raise StopIteration
    check_core_growth.core_norm = 0
    return check_core_growth


def sa_hooi(
        idx, val, shape, mlrank, attention_matrix, scaling_weights, testset, holdout, data_description,
        max_iters = 10,
        parallel_ttm = (True, True, True),
        growth_tol = 0.001,
        randomized=True,
        seed = None,
        iter_callback=None,
    ):

    best_metric = -1
    assert valid_mlrank(mlrank)
    n_users, n_items, n_positions = shape
    r0, r1, r2 = mlrank
    
    tensor_data = idx, val, shape
    if not isinstance(parallel_ttm, (list, tuple)):
        parallel_ttm = [parallel_ttm] * len(shape)

    assert len(shape) == len(parallel_ttm)

    index_data = arrange_indices(idx, parallel_ttm)
    ttm = [ttm3d_par if par else ttm3d_seq for par in parallel_ttm]

    random_state = np.random if seed is None else np.random.RandomState(seed)
    u1 = initialize_columnwise_orthonormal((n_items, r1), random_state)
    uw = u1 * scaling_weights[:, np.newaxis]
    u2 = initialize_columnwise_orthonormal((n_positions, r2), random_state)
    ua = attention_matrix.dot(u2)

    if randomized:
        svd = randomized_svd
        svd_config = lambda rank: dict(n_components=rank, random_state=seed)
    else:
        svd = svds
        svd_config = lambda rank: dict(k=rank, return_singular_vectors='u')
    
    if iter_callback is None:
        iter_callback = core_growth_callback(growth_tol)
        
    
    for step in range(max_iters):
        ttm0 = ttm[0](*tensor_data, ua, uw, ((2, 0), (1, 0)), *index_data[0]).reshape(shape[0], r1*r2)
        u0, *_ = svd(ttm0, **svd_config(r0))

        ttm1 = ttm[1](*tensor_data, ua, u0, ((2, 0), (0, 0)), *index_data[1]).reshape(shape[1], r0*r2)
        u1, *_ = svd(ttm1, **svd_config(r1))
        uw = u1 * scaling_weights[:, np.newaxis]

        ttm2 = ttm[2](*tensor_data, uw, u0, ((1, 0), (0, 0)), *index_data[2]).reshape(shape[2], r0*r1)
        u2, ss, _ = svd(ttm2, **svd_config(r2))
        ua = attention_matrix.dot(u2)

        factors = (u0, u1, u2)
        tf_params = u0, u1, u2, attention_matrix
        try:
            cur_metric = iter_callback(step, np.linalg.norm(ss), factors, tf_params, best_metric, testset, holdout, data_description)
            best_metric = cur_metric
        except StopIteration:
            break
    return factors

def exp_decay(decay_factor, n):
    return np.e**(-(n-1)*decay_factor)

def lin_decay(decay_factor, n):
    return n**(-decay_factor)



def attention_weights(decay_factor, cutoff, max_elements=None, exponential_decay=False, reverse=False):
    if (decay_factor == 0 or cutoff == 0) and (max_elements is None or max_elements <= 0):
        raise SeqTFError('Infinite sequence.')
    decay_function = exp_decay if exponential_decay else lin_decay
    weights = takewhile(lambda x: x>=cutoff, (decay_function(decay_factor, n) for n in count(1, 1)))
    if max_elements is not None:
        weights = islice(weights, max_elements)
    if reverse:
        return list(reversed(list(weights)))
    return list(weights)

def form_attention_matrix(size, decay_factor, cutoff=0, span=0, exponential_decay=False, reverse=False, format='csc', stochastic_axis=None, dtype=None):
    stochastic = stochastic_axis is not None
    span = min(span or np.iinfo('i8').max, size)
    weights = attention_weights(decay_factor, cutoff=cutoff, max_elements=span, exponential_decay=exponential_decay, reverse=reverse)
    diag_values = [np.broadcast_to(w, size) for w in weights]
    matrix = diags(diag_values, offsets=range(0, -len(diag_values), -1), format=format, dtype=dtype)
    if stochastic:
        scalings = matrix.sum(axis=stochastic_axis).A.squeeze()
        if stochastic_axis == 0:
            matrix = matrix.dot(diags(1./scalings))
        else:
            matrix = diags(1./scalings).dot(matrix)
    return matrix.asformat(format)


def generate_banded_form(matrix):
    matrix = matrix.todia()
    bands = matrix.data
    offsets = matrix.offsets
    num_l = (offsets < 0).sum()
    num_u = (offsets > 0).sum()
    return (num_l, num_u), bands[np.argsort(offsets)[::-1], :]


def generate_position_projector(attention_matrix, position_factors):
    shape, bands = generate_banded_form(attention_matrix.T)
    wl = solve_banded(shape, bands, position_factors)
    wr = attention_matrix.dot(position_factors)
    last_position_projector = wr @ wl[-1, :]
    return last_position_projector


def get_scaling_weights(frequencies, scaling=1.0):
    return np.power(frequencies, 0.5*(scaling-1.0))


def valid_mlrank(mlrank):
    prod = np.prod(mlrank)
    return all(prod//r > r for r in mlrank)