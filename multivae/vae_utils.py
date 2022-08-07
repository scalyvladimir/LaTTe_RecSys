# +
import torch.nn.functional as F
import torch.nn as nn
import torch

import numpy as np


# +
def init_weights(model):
    for name, param in model.named_parameters():
        if "weight" in name:
            nn.init.xavier_uniform_(param.data)
        elif "bias" in name:
            param.data.normal_(std=0.001)           

def vae_loss_fn(inp, out, mu, logvar, anneal):
    neg_ll = -torch.mean(torch.sum(F.log_softmax(out, 1) * inp, -1))
    KLD = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    return neg_ll + anneal * KLD


def early_stopping(curr_value, best_value, stop_step, patience, score_fn):
    if (score_fn == "loss" and curr_value <= best_value) or (
        score_fn == "metric" and curr_value >= best_value
    ):
        stop_step, best_value = 0, curr_value
    else:
        stop_step += 1
    if stop_step >= patience:
        print(f'Early stopping triggered. patience: {patience} log: {best_value:.3}')
        stop = True
    else:
        stop = False
    return best_value, stop_step, stop
