# To restrict number of CPU cores uncomment the lines of code below.

# import os
# os.environ["OMP_NUM_THREADS"] = "24"
# os.environ["MKL_NUM_THREADS"] = "24"
# os.environ["NUMBA_NUM_THREADS"] = "24"

import numpy as np
import pandas as pd
from tqdm import tqdm

from evaluation import downvote_seen_items, topn_recommendations
from scipy.linalg import sqrtm
#from IPython.utils import io

from Modules import read_amazon_data, full_preproccessing, make_prediction, model_evaluate
from Random_MP import build_random_model, random_model_scoring, build_popularity_model, popularity_model_scoring
from NormPureSVD_EASEr import build_svd_model, svd_model_scoring, easer, easer_scoring, matrix_from_observations
from CoFFee_LaTTe import full_pipeline, get_similarity_matrix
import sys

# +
# Torch imports
import torch.nn.functional as F
import torch.nn as nn
import torch

from multivae.pytorch_models import MultiVAE
# -

# VAE imports
from multivae.vae_utils import init_weights, vae_loss_fn, early_stopping
from scipy.sparse import csr_matrix

names = ["Movielens_1M", "Movielens_10M", "Video_Games", "CDs_and_Vinyl", "Electronics", "Video_Games_nf"]

if len(sys.argv) > 1:
    data_name = sys.argv[1]
    assert data_name in names, f"Name of dataset must be one the following {names}"
else:
    data_name = "Movielens_1M"

print(f"Starting tuning models for dataset {data_name}.\n")

if (data_name != "Electronics"):
    q = 0.8
else:
    q = 0.95

if data_name == "Movielens_1M":
    data = None
    name = "ml-1m.zip"
elif data_name == "Movielens_10M":
    data = None
    name = "ml-10m.zip"
elif data_name == "Video_Games_nf":
    data = pd.read_csv("ratings_Video_Games.csv")
    name = None
else:
    data = read_amazon_data(name = data_name)
    data.rename(columns = {'reviewerID' : 'userid', 'asin' : 'movieid', "overall" : "rating", "unixReviewTime" : "timestamp"}, inplace = True) 
    name = None


training, testset_valid, holdout_valid, testset, holdout, data_description, data_index = full_preproccessing(data, name, q)

print("Tuning PureSVD on progress...")

rank_grid = []
for i in range(5, 9):
    rank_grid.append(2 * 2 ** i)
    rank_grid.append(3 * 2 ** i)

rank_grid = np.array(rank_grid)
f_grid = np.linspace(0, 2, 21)

hr_tf = {}
mrr_tf = {}
C_tf = {}
for f in tqdm(f_grid):
    svd_config = {'rank': rank_grid[-1], 'f': f}
    svd_params = build_svd_model(svd_config, training, data_description)
    for r in rank_grid:
        svd_scores = svd_model_scoring(svd_params[:, :r], testset_valid, data_description)
        downvote_seen_items(svd_scores, testset_valid, data_description)
        svd_recs = topn_recommendations(svd_scores, topn=10)
        hr, hr_pos, hr_neg, mrr, mrr_pos, mrr_neg, cov, C = model_evaluate(svd_recs, holdout_valid, data_description, alpha=3, topn=10, dcg=False)
        hr_tf[f'r={r}, f={f:.2f}'] = hr
        mrr_tf[f'r={r}, f={f:.2f}'] = mrr
        C_tf[f'r={r}, f={f:.2f}'] = C

print("Validation tuning result:")
print("HR")
hr_sorted = sorted(hr_tf, key=hr_tf.get, reverse=True)
for i in range(1):
    print(hr_sorted[i], hr_tf[hr_sorted[i]])

print("MRR")
mrr_sorted = sorted(mrr_tf, key=mrr_tf.get, reverse=True)
for i in range(1):
    print(mrr_sorted[i], mrr_tf[mrr_sorted[i]])

print("MCC")
C_sorted = sorted(C_tf, key=C_tf.get, reverse=True)
for i in range(1):
    print(C_sorted[i], C_tf[C_sorted[i]])


print("Evaluation on testset (Random, MP, SVD) in progress...")


data_description["test_users"] = holdout[data_index['users'].name].drop_duplicates().values
data_description["n_test_users"] = holdout[data_index['users'].name].nunique()

print("Random:")

rnd_params = build_random_model(training, data_description)
rnd_scores = random_model_scoring(rnd_params, None, data_description)
downvote_seen_items(rnd_scores, testset, data_description)
_ = make_prediction(rnd_scores, holdout, data_description, mode="Test")
print()

print("MP:")

pop_params = build_popularity_model(training, data_description)
pop_scores = popularity_model_scoring(pop_params, None, data_description)
downvote_seen_items(pop_scores, testset, data_description)
_ = make_prediction(pop_scores, holdout, data_description, mode="Test")
print()


print('Normalized PureSVD:')

for_hr = sorted(hr_tf, key=hr_tf.get, reverse=True)[0]
for_mrr = sorted(mrr_tf, key=mrr_tf.get, reverse=True)[0]
for_mc = sorted(C_tf, key=C_tf.get, reverse=True)[0]

svd_config_hr = {'rank': int(for_hr.split(",")[0][2:]), 'f': float(for_hr.split(",")[1][3:])}
svd_config_mrr = {'rank': int(for_mrr.split(",")[0][2:]), 'f': float(for_mrr.split(",")[1][3:])}
svd_config_mc = {'rank': int(for_mc.split(",")[0][2:]), 'f': float(for_mc.split(",")[1][3:])}

svd_configs = [(svd_config_hr, "Tuned by HR"), (svd_config_mrr, "Tuned by MRR"), (svd_config_mc, "Tuned by MCC")]

for svd_config in svd_configs:
    print(svd_config)
    svd_params = build_svd_model(svd_config[0], training, data_description)
    svd_scores = svd_model_scoring(svd_params, testset, data_description)
    downvote_seen_items(svd_scores, testset, data_description)

    _ = make_prediction(svd_scores, holdout, data_description, mode="Test")

print("Evaluation on testset (Random, MP, SVD) ended.\n")

factor = float(for_mc.split(",")[1][3:])

print("CoFFee tuning in progress...")    

config = {
    "scaling": 1,
    "mlrank": (30, 30, data_description['n_ratings']),
    "n_ratings": data_description['n_ratings'],
    "num_iters": 5,
    "params": None,
    "randomized": True,
    "growth_tol": 1e-4,
    "seed": 42
}

data_description["test_users"] = holdout_valid[data_index['users'].name].drop_duplicates().values
data_description["n_test_users"] = holdout_valid[data_index['users'].name].nunique()

attention_matrix = np.eye(data_description["n_ratings"])
full_pipeline(config, training, data_description, testset_valid, holdout_valid, testset, holdout, attention_matrix=attention_matrix, factor = factor)

print("LaTTe tuning in progress...\n")

modes = [ "linear", "sq3", "sigmoid", "arctan"]

for mode in modes:
    print(f"For similarity matrix '{mode}'' tuning...")
    similarity_matrix = get_similarity_matrix(mode, data_description["n_ratings"])
    attention_matrix = sqrtm(similarity_matrix).real
    full_pipeline(config, training, data_description, testset_valid, holdout_valid, testset, holdout, attention_matrix=attention_matrix, factor = float(for_mc.split(",")[1][3:]))
    print("_____________________________________________________")

print("LaTTe tuning ended.\n")

print("EASEr tuning in progress...\n")

lambda_grid = np.arange(50, 1000, 50)

hr_tf = {}
mrr_tf = {}
C_tf = {}

for lmbda in tqdm(lambda_grid):
    easer_params = easer(training, data_description, lmbda=lmbda)
    easer_scores = easer_scoring(easer_params, testset_valid, data_description)
    downvote_seen_items(easer_scores, testset_valid, data_description)
    easer_recs = topn_recommendations(easer_scores, topn=10)
    hr, hr_pos, hr_neg, mrr, mrr_pos, mrr_neg, cov, C = model_evaluate(easer_recs, holdout_valid, data_description, alpha=3, topn=10, dcg=False)
    hr_tf[lmbda] = hr
    mrr_tf[lmbda] = mrr
    C_tf[lmbda] = C

print("Validation tuning result:")

print("HR")
hr_sorted = sorted(hr_tf, key=hr_tf.get, reverse=True)
for i in range(1):
    print(hr_sorted[i], hr_tf[hr_sorted[i]])

print("MRR")
mrr_sorted = sorted(mrr_tf, key=mrr_tf.get, reverse=True)
for i in range(1):
    print(mrr_sorted[i], mrr_tf[mrr_sorted[i]])

print("MCC")
C_sorted = sorted(C_tf, key=C_tf.get, reverse=True)
for i in range(1):
    print(C_sorted[i], C_tf[C_sorted[i]])

print("Evaluation on testset (EASEr) in progress...")

data_description["test_users"] = holdout[data_index['users'].name].drop_duplicates().values
data_description["n_test_users"] = holdout[data_index['users'].name].nunique()

easer_params = easer(training, data_description, lmbda=C_sorted[i])
easer_scores = easer_scoring(easer_params, testset, data_description)
downvote_seen_items(easer_scores, testset, data_description)

_ = make_prediction(easer_scores, holdout, data_description, mode='Test')

print("EASEr tuning ended.\n")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

csr_training = matrix_from_observations(training, data_description)
csr_testset_valid = matrix_from_observations(testset_valid, data_description)


# +
def train_step(model, optimizer, data, epoch, batch_size, anneal_cap, device='cpu'):

    model.train()
    running_loss = 0.0
    global update_count
    N = data.shape[0]
    idxlist = list(range(N))
    np.random.shuffle(idxlist)
    training_steps = len(range(0, N, batch_size))

    for batch_idx, start_idx in enumerate(range(0, N, batch_size)):

        end_idx = min(start_idx + batch_size, N)
        X_inp = data[idxlist[start_idx:end_idx]]
        X_inp = torch.FloatTensor(X_inp.toarray()).to(device)

        anneal = min(anneal_cap, update_count / total_anneal_steps)
        update_count += 1

        optimizer.zero_grad()
        
        X_out, mu, logvar = model(X_inp)
        loss = vae_loss_fn(X_inp, X_out, mu, logvar, anneal)
        train_step.anneal = anneal

        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        avg_loss = running_loss / (batch_idx + 1)

def eval_step(model, csr_data, data, holdout, batch_size, data_type="valid", holdout_description=None, device='cpu'):

    model.eval()
    running_loss = 0.0
    eval_idxlist = list(range(csr_data.shape[0]))
    eval_N = csr_data.shape[0]
    eval_steps = len(range(0, eval_N, batch_size))
    
    n_users = data['userid'].nunique()

    hr_list, mrr_list, C_list = [], [], []
    mrr_pos_list, mrr_neg_list, cov_set = [], [], set()
    hr_pos_list, hr_neg_list = [], [] 

    users_pred = []
    
    with torch.no_grad():
        for batch_idx, start_idx in enumerate(range(0, eval_N, batch_size)):

            end_idx = min(start_idx + batch_size, eval_N)
            X_tr = csr_data[eval_idxlist[start_idx:end_idx]]
            X_tr_inp = torch.FloatTensor(X_tr.toarray()).to(device)
        
            X_out, mu, logvar = model(X_tr_inp)
            loss = vae_loss_fn(X_tr_inp, X_out, mu, logvar, train_step.anneal)
            
            running_loss += loss.item()
            avg_loss = running_loss / (batch_idx + 1)

            # Exclude examples from training set
            X_out = X_out.cpu().numpy()
            X_out[X_tr.nonzero()] = -np.inf

            recs = topn_recommendations(X_out, topn=10)

            users_pred.append(recs)

        user_pred = np.vstack(users_pred)[holdout.userid,:]
                                
    hr, hr_pos, hr_neg, mrr, mrr_pos, mrr_neg, cov, C = model_evaluate(user_pred, holdout, holdout_description, alpha=3, topn=10, dcg=False)

    return avg_loss, hr, mrr, hr_pos, hr_neg, mrr_pos, mrr_neg, cov, C



# +
n_epochs = 200

batch_size = 500
# -

dim_grid = [
    [50, 150],
    [100, 300],
    [50, 300],
    [200, 400],
    [200, 600],
    [400, 800],
    [400, 1200]
]

print("MultiVAE tuning in progress...\n")

# +
c_list = []
anneal_list = []

n_items = training.movieid.nunique()

for tuned_dims in tqdm(dim_grid):

    model = MultiVAE(
        p_dims=tuned_dims + [n_items],
        q_dims=[n_items] + tuned_dims[::-1],
        dropout_enc=[0.5, 0.],
        dropout_dec=[0., 0.],
    )

    init_weights(model)

    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    ###########################################
    # Finding best anneal factor for training #
    ###########################################

    anneal_cap = 1.0

    training_steps = len(range(0, training.shape[0], batch_size))
    try:
        total_anneal_steps = (
            training_steps * (n_epochs - int(n_epochs * 0.2))
        ) / anneal_cap
    except ZeroDivisionError:
        assert (
            constant_anneal
        ), "if 'anneal_cap' is set to 0.0 'constant_anneal' must be set to 'True"

    ################################
    # Training with early stopping #
    ################################

    stop_step = 0
    update_count = 0
    stop = False

    patience_val = 10 # steps to stop after plateau or decreasing metric
    best_c = -np.inf

    for epoch in range(n_epochs):

        if stop:
            break

        train_step(model, optimizer, csr_training, epoch, batch_size, anneal_cap, device)

        c_val = eval_step(model, csr_testset_valid, testset_valid, holdout_valid, batch_size, holdout_description=data_description, device=device)[-1]

        best_c, stop_step, stop = early_stopping(c_val, best_c, stop_step, patience_val, score_fn='metric')

    # Found best param    
    anneal_cap = train_step.anneal
    
    training_steps = len(range(0, training.shape[0], batch_size))
    try:
        total_anneal_steps = (
            training_steps * (n_epochs - int(n_epochs * 0.2))
        ) / anneal_cap
    except ZeroDivisionError:
        assert (
            constant_anneal
        ), "if 'anneal_cap' is set to 0.0 'constant_anneal' must be set to 'True"

    init_weights(model)

    stop_step = 0
    update_count = 0
    stop = False

    patience_val = 20 # steps to stop after plateau or decreasing metric
    best_c = -np.inf

    for epoch in range(n_epochs):
        if stop:
            break

        train_step(model, optimizer, csr_training, epoch, batch_size, anneal_cap, device)

        c_val = eval_step(model, csr_testset_valid, testset_valid, holdout_valid, batch_size, holdout_description=data_description, device=device)[-1]
        
        best_c, stop_step, stop = early_stopping(c_val, best_c, stop_step, patience_val, score_fn='metric')

    c_list.append(best_c)
    anneal_list.append(anneal_cap)

best_id = np.argsort(c_list)[-1]

best_params = {
    'dims': dim_grid[best_id],
    'anneal': anneal_list[best_id],
}
# -

print('best found params:', best_params)

dims = best_params['dims']

# +
model = MultiVAE(
    p_dims=dims + [n_items],
    q_dims=[n_items] + dims[::-1],
    dropout_enc=[0.5, 0.],
    dropout_dec=[0., 0.],
)

init_weights(model)

model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# +
anneal_cap = best_params['anneal']

training_steps = len(range(0, training.shape[0], batch_size))
try:
    total_anneal_steps = (
        training_steps * (n_epochs - int(n_epochs * 0.2))
    ) / anneal_cap
except ZeroDivisionError:
    assert (
        constant_anneal
    ), "if 'anneal_cap' is set to 0.0 'constant_anneal' must be set to 'True"
# -

print("Training with best params in progress...")

# +
stop_step = 0
update_count = 0
stop = False

patience_val = 20 # steps to stop after plateau or decreasing metric
best_c = -np.inf

for epoch in range(n_epochs):
    
    if stop:
        break
    
    train_loss = train_step(model, optimizer, csr_training, epoch, batch_size, anneal_cap, device=device)
    
    val_loss, hr, mrr, _, _, _, _, _, c_val = eval_step(model, csr_testset_valid, testset_valid, holdout_valid, batch_size, holdout_description=data_description, device=device)

    print("=" * 80)
    print(
        "| valid loss {:4.3f} | HR@10 {:4.3f} | MRR@10 {:4.3f} | "
        "Matthew's {:4.3f}".format(val_loss, hr, mrr, c_val)
    )
    print("=" * 80)

    best_c, stop_step, stop = early_stopping(c_val, best_c, stop_step, patience_val, score_fn='metric')    
# -

print("Evaluation on testset (MultiVAE) in progress...")

csr_testset = matrix_from_observations(testset, data_description)

_, hr, mrr, hr_pos, hr_neg, mrr_pos, mrr_neg, cov, C = eval_step(model, csr_testset, testset, holdout, batch_size, holdout_description=data_description, device=device)

print("=" * 80)
print(f"HR@10 = {hr:.4f}, MRR@10 = {mrr:.4f}, Coverage@10 = {cov:.4f}")
print(f"HR_pos@10 = {hr_pos:.4f}, HR_neg@10 = {hr_neg:.4f}")
print(f"MRR_pos@10 = {mrr_pos:.4f}, MRR_neg@10 = {mrr_neg:.4f}")
print(f"Matthews@10 = {C:.4f}")
print("=" * 80)

print("MultiVAE tuning ended.\n")
