DATASET = 'imdb'
FEATURES_FILE = f'data/{DATASET}_features.csv'
INFERENCE_FILE = f'data/{DATASET}_inference.csv'
SVC_FEATURES_FILE = f'data/{DATASET}_svc.csv'

import arviz as az
import numpy as np
import pandas as pd
import seaborn as sns

from scipy import stats
from scipy.special import expit as logistic

from datetime import datetime
import re 
import sys
import os
import sklearn.metrics

import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import FormatStrFormatter
from matplotlib import rc
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator, FixedLocator)

# Note for reproducability:
# use numpyro == 0.8.0
#     jax     == 0.3.0
#     jaxlib  == 0.3.0
import numpyro
import numpyro.distributions as dist
import numpyro.optim as optim
from numpyro.diagnostics import effective_sample_size, hpdi, print_summary
from numpyro.infer import NUTS, MCMC, Predictive, SVI, Trace_ELBO, init_to_value
from numpyro.infer.autoguide import AutoLaplaceApproximation
from numpyro import sample, deterministic
numpyro.set_host_device_count(8)

from numpyro.diagnostics import effective_sample_size, print_summary

import warnings

import jax
import jax.numpy as jnp
from jax import lax, random
from jax.scipy.special import expit

# color definitions
base_color = "#1696D2"
accent_color = "#FCB918"
neutral_gray= "#C6C6C6"

RANDOM_SEED = 8927
np.random.seed(RANDOM_SEED)
az.style.use("arviz-white")
az.rcParams["stats.hdi_prob"] = 0.89
az.rcParams["stats.ic_scale"] = "deviance"
az.rcParams["stats.information_criterion"] = "loo"

def standardize(x):
    a = np.mean(x)
    s = np.std(x)
    return((x-a)/s)

def simpleBar(var, idata, ytype='topics',  filename='plots/test.pdf'):
    rc('text', usetex=False)
    variable = var
    
    means = idata.posterior[variable].mean(("chain", "draw"))
    sorted_means = np.sort(means)

    az.style.use("arviz-white")
    ylabels = np.array(idata.posterior.coords[ytype])
    sorted_ylabels = idata.posterior[ytype].sortby(means)
    print(sorted_means.mean())

    n = len(sorted_ylabels)
    ratio = 1.62  # golden ratio
    if n > 5:
        factor = np.log(n) / np.log(5)
    else:
        factor = 1.0

    z = (5 * factor, 5 * factor*ratio)
    fig, axs = sns.mpl.pyplot.subplots(1, 1, figsize=z)
    palette = sns.light_palette(base_color, as_cmap=False, n_colors=n)

    z = axs
    z.spines['right'].set_visible(False)
    z.spines['top'].set_visible(False)
    z.spines['left'].set_visible(False)
    z.spines['bottom'].set_visible(False)
    z.spines['bottom'].set_position(('outward', 10))

    z.xaxis.set_minor_locator(AutoMinorLocator(5))
    z.tick_params(axis='x', labelsize=10)
    z.tick_params(axis='y', labelsize=10)

    z.tick_params(which='both', width=2)
    z.tick_params(which='major', length=4, color='k')
    z.tick_params(which='minor', length=2, color=neutral_gray)

    z.autoscale(enable=True, axis='x', tight=None)

    sns.barplot(x=sorted_means, y=np.array(sorted_ylabels), color=base_color, ax=axs, palette=palette)
    plt.savefig(filename)

def model0(arg_a_id, arg_b_id, topic, a_std_feat, b_std_feat, pref=None):

    N_a = len(np.unique(arg_a_id))
    N_b = len(np.unique(arg_b_id))
    N_t = len(np.unique(topic))

    p_bar = sample("p_bar", dist.Normal(0, 0.5))  # the intercept

    # pooled coefficients for a
    α_σ = sample("α_σ", dist.Exponential(1.0))
    α_bar = sample("α_bar", dist.Normal(0.0, 0.25))
    α_var = sample("α_var", dist.Normal(0.0, 1.0).expand([N_a]))
    α = α_bar + α_var * α_σ

    # pooled coefficients for b
    β_σ = sample("β_σ", dist.Exponential(1.0))
    β_bar = sample("β_bar", dist.Normal(0.0, 0.25))
    β_var = sample("β_var", dist.Normal(0.0, 1.0).expand([N_b]))
    β = β_bar + β_var * β_σ

    # pooled coefficients for the slopes
    γ_σ = sample("γ_σ", dist.Exponential(1.0))
    γ_bar = sample("γ_bar", dist.Normal(0.0, 0.25))
    γ_var = sample("γ_var", dist.Normal(0.0, 1.0).expand([N_t]))
    γ = γ_bar + γ_var * γ_σ

    # the equation below models: what can directly influence the outcome?
    # there could be a base probability;
    # it's possible that there is something interesting about the arguments themselves and not captured by the feature
    # the feature difference matters and the slope is related to the topic
    logit_p = p_bar + (α[arg_a_id] -
                       β[arg_b_id]) + γ[topic] * (a_std_feat - b_std_feat)

    sample("preference", dist.Binomial(logits=logit_p), obs=pref)

def get_feature_correlation(feature):
    a_idx, label_a = pd.factorize(df["a_id"], sort = True)
    b_idx, label_b = pd.factorize(df["b_id"], sort = True)
    topic_idx, label_t = pd.factorize(df["topic"], sort = True)

    arg_a_id = jnp.asarray(a_idx)
    arg_b_id = jnp.asarray(b_idx)
    topic = jnp.asarray(topic_idx)

    a_std_feat = jnp.asarray(standardize(df['a_' + feature]))
    b_std_feat = jnp.asarray(standardize(df['b_' + feature]))
    pref = jnp.asarray(df.gold_label.values)

    data_dict = {
        "arg_a_id" : arg_a_id,
        "arg_b_id" : arg_b_id,
        "topic" : topic,
        "a_std_feat" : a_std_feat,
        "b_std_feat" : b_std_feat,   
        "pref" : pref
    }

    dims = {
        "α_var": ["Argument A"],
        "β_var": ["Argument B"],
        "γ_var": ["topics"]
    }

    coords = {"Argument A": label_a, "Argument B": label_b, "topics": label_t}

    comparisionmodel = model0

    kernel = NUTS(comparisionmodel, target_accept_prob=0.80)
    sample_kwargs = dict(
        sampler=kernel,
        num_warmup=1000,
        num_samples=1000,
        num_chains=4,
        chain_method="parallel"
    )
    rng_key = random.PRNGKey(0)
    f_mcmc = MCMC(**sample_kwargs)
    f_mcmc.run(rng_key, **data_dict)
    
    f_data = az.from_numpyro(f_mcmc, dims=dims, coords=coords)
    
    df_summary = az.summary(data = f_data, var_names = ['γ_var'], round_to=2)
    df_summary.to_csv(f'summaries/test_{feature}.csv')
    
    avg_logits = f_data.posterior["γ_var"].mean(("chain", "draw")).mean()
    return f_data, f'{feature}, {avg_logits}\n'

if __name__ == "__main__":

    assert len(sys.argv) >= 4 or not len(sys.argv)

    if len(sys.argv) >= 4:
        DATASET = sys.argv[1]
        FEATURES_FILE = sys.argv[2]
        INFERENCE_FILE = sys.argv[3]
        if len(sys.argv) > 4:
            SVC_FEATURES_FILE = sys.argv[4]

    features_df = pd.read_csv(FEATURES_FILE, index_col=0)
    inference_df = pd.read_csv(INFERENCE_FILE, index_col=0)

    if DATASET == 'imdb':
        inference_df = inference_df.drop(['mem_line_no', 'nonmem_line_no', 'title'], axis=1)

    if SVC_FEATURES_FILE:
        svc_df = pd.read_csv(SVC_FEATURES_FILE, index_col=0)
        features_df = pd.concat([features_df, svc_df], axis=1)

    new_dfs = []
    features_df['text'] = features_df['text'].apply(lambda x : x.strip())
    inference_df['sentence_a'] = inference_df['sentence_a'].apply(lambda x : x.strip())
    inference_df['sentence_b'] = inference_df['sentence_b'].apply(lambda x : x.strip())

    for i, row in inference_df.iterrows():
        a_feats = features_df[features_df['text'] == row['sentence_a']]
        a_feats.columns = list(map(lambda x: 'a_' + x, a_feats.columns))
        b_feats = features_df[features_df['text'] == row['sentence_b']]
        b_feats.columns = list(map(lambda x: 'b_' + x, b_feats.columns))
        
        a_feats.reset_index(inplace=True)
        b_feats.reset_index(inplace=True)
        
        new_df = pd.concat([a_feats, b_feats], axis=1)
        new_df['gold_label'] = row['label']
        new_df['pred_confidence'] = row['confidence']
        new_dfs.append(new_df.head(1))

    df = pd.concat(new_dfs)
    df['topic'] = 0

    features = list(set(df.columns) - set(['a_id', 'b_id', 'a_text', 'b_text', 'topic_id', 'pred_confidence', 'gold_label', 'topic']))
    features = list(sorted(list(set(map(lambda x: x[2:], features)))))

    for feature in features:
        with open(f'{DATASET}_correlations.txt', 'a+') as f:
            _, avg_str = get_feature_correlation(feature)
            f.write(avg_str)