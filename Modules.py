# +
import gzip
import tempfile
from ast import literal_eval
from  urllib import request

from io import BytesIO
import numpy as np
import pandas as pd

try:
    from pandas.io.common import ZipFile
except ImportError:
    from zipfile import ZipFile
# -

from dataprep import transform_indices
from polara.preprocessing.dataframes import leave_one_out, reindex
from evaluation import topn_recommendations

def amazon_data_reader(path):
    with gzip.open(path, 'rt') as gz:
        for line in gz:
            yield literal_eval(line)

def read_amazon_data(path=None, name=None):
    '''Data is taken from https://jmcauley.ucsd.edu/data/amazon/'''
    if path is None and name is None:
            raise ValueError('Either the name of the dataset to download \
                or a path to a local file must be specified.')
    if path is None:
        file_url = f'http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_{name}_5.json.gz'
        print(f'Downloading data from: {file_url}')
        with request.urlopen(file_url) as response:
            file = response.read()
            with tempfile.NamedTemporaryFile(delete=False) as temp:
                temp.write(file)
                path = temp.name
                print(f'Temporarily saved file at: {path}')
    return pd.DataFrame.from_records(
        amazon_data_reader(path),
        columns=['reviewerID', 'asin', 'overall', 'unixReviewTime']
    )


def full_preproccessing(data = None, name="ml-10m.zip", q=0.8):
    if data is None:
        data = get_movielens_data(name, include_time=True)
    test_timepoint = data['timestamp'].quantile(
    q=q, interpolation='nearest'
    )
    
    labels, levels = pd.factorize(data.movieid)
    data.loc[:, 'movieid'] = labels

    labels, levels = pd.factorize(data.userid)
    data.loc[:, 'userid'] = labels
    
    if (data["rating"].nunique() > 5):
        data["rating"] = data["rating"] * 2
        
    data["rating"] = data["rating"].astype(int)

    test_data_ = data.query('timestamp >= @test_timepoint')
    train_data_ = data.query(
    'userid not in @test_data_.userid.unique() and timestamp < @test_timepoint'
    )
    
    training, data_index = transform_indices(train_data_.copy(), 'userid', 'movieid')
    test_data = reindex(test_data_, data_index['items'])

    testset_, holdout_ = leave_one_out(
    test_data, target='timestamp', sample_top=True, random_state=0
    )
    testset_valid_, holdout_valid_ = leave_one_out(
        testset_, target='timestamp', sample_top=True, random_state=0
    )

    test_users_val = np.intersect1d(testset_valid_.userid.unique(), holdout_valid_.userid.unique())
    testset_valid = testset_valid_.query('userid in @test_users_val').sort_values('userid')
    holdout_valid = holdout_valid_.query('userid in @test_users_val').sort_values('userid')

    test_users = np.intersect1d(testset_.userid.unique(), holdout_.userid.unique())
    testset = testset_.query('userid in @test_users').sort_values('userid')
    holdout = holdout_.query('userid in @test_users').sort_values('userid')
    
    assert holdout_valid.set_index('userid')['timestamp'].ge(
        testset_valid
        .groupby('userid')
        ['timestamp'].max()
    ).all()

    data_description = dict(
        users = data_index['users'].name,
        items = data_index['items'].name,
        feedback = 'rating',
        n_users = len(data_index['users']),
        n_items = len(data_index['items']),
        n_ratings = training['rating'].nunique(),
        min_rating = training['rating'].min(),
        test_users = holdout_valid[data_index['users'].name].drop_duplicates().values, # NEW
        n_test_users = holdout_valid[data_index['users'].name].nunique() # NEW
    )

    return training, testset_valid, holdout_valid, testset, holdout, data_description, data_index


def model_evaluate(recommended_items, holdout, holdout_description, alpha=3, topn=10, dcg=False):
    itemid = holdout_description['items']
    rateid = holdout_description['feedback']
    alpha = 3 if holdout_description["n_ratings"] == 5 else 6
    n_test_users = recommended_items.shape[0]
    holdout_items = holdout[itemid].values
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

def valid_mlrank(mlrank):
    '''
    Only allow ranks that are suitable for truncated SVD computations
    on unfolded compressed tensor (the result of ttm product in HOOI).
    '''
    s, r1, r3 = mlrank
    r2 = r1
    return r1*r2 > r3 and r1*r3 > r2 and r2*r3 > r1


# +
from requests import get

def get_movielens_data(filename=None, get_ratings=True, get_genres=False,
                       split_genres=True, mdb_mapping=False, get_tags=False, include_time=False):
    '''Downloads movielens data and stores it in pandas dataframe.
    '''
    fields = ['userid', 'movieid', 'rating']

    if include_time:
        fields.append('timestamp')

    if filename not in ['ml-1m.zip', 'ml-10m.zip']:
        raise NameError
        
    zip_file_url = f'http://files.grouplens.org/datasets/movielens/{filename}'
    zip_response = get(zip_file_url)
    zip_contents = BytesIO(zip_response.content)

    ml_data = ml_genres = ml_tags = mapping = None
    # loading data into memory
    with ZipFile(zip_contents) as zfile:
        zip_files = pd.Series(zfile.namelist())
        zip_file = zip_files[zip_files.str.contains('ratings')].iat[0]
        is_new_format = ('latest' in zip_file) or ('20m' in zip_file)
        delimiter = ','
        header = 0 if is_new_format else None
        if get_ratings:
            zdata = zfile.read(zip_file)
            zdata = zdata.replace(b'::', delimiter.encode())
            # makes data compatible with pandas c-engine
            # returns string objects instead of bytes in that case
            ml_data = pd.read_csv(BytesIO(zdata), sep=delimiter, header=header, engine='c', names=fields, usecols=fields)

        if get_genres:
            zip_file = zip_files[zip_files.str.contains('movies')].iat[0]
            zdata =  zfile.read(zip_file)
            if not is_new_format:
                # make data compatible with pandas c-engine
                # pandas returns string objects instead of bytes in that case
                delimiter = '^'
                zdata = zdata.replace(b'::', delimiter.encode())
            genres_data = pd.read_csv(BytesIO(zdata), sep=delimiter, header=header,
                                      engine='c', encoding='unicode_escape',
                                      names=['movieid', 'movienm', 'genres'])

            ml_genres = get_split_genres(genres_data) if split_genres else genres_data

        if get_tags:
            zip_file = zip_files[zip_files.str.contains('/tags')].iat[0] #not genome
            zdata =  zfile.read(zip_file)
            if not is_new_format:
                # make data compatible with pandas c-engine
                # pandas returns string objects instead of bytes in that case
                delimiter = '^'
                zdata = zdata.replace(b'::', delimiter.encode())
            fields[2] = 'tag'
            ml_tags = pd.read_csv(BytesIO(zdata), sep=delimiter, header=header,
                                      engine='c', encoding='latin1',
                                      names=fields, usecols=range(len(fields)))

        if mdb_mapping and is_new_format:
            # imdb and tmdb mapping - exists only in ml-latest or 20m datasets
            zip_file = zip_files[zip_files.str.contains('links')].iat[0]
            with zfile.open(zip_file) as zdata:
                mapping = pd.read_csv(zdata, sep=',', header=0, engine='c',
                                        names=['movieid', 'imdbid', 'tmdbid'])

    res = [data for data in [ml_data, ml_genres, ml_tags, mapping] if data is not None]
    if len(res)==1: res = res[0]
    return res
