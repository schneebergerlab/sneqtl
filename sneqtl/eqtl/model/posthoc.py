import itertools as it

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
import torch

from .lm import _as_tensors


LOD = 2 * np.log(10)


def _wald_test_param_pairs(coefs, cov, ref_value=0.0):
    n_hap = coefs.shape[1]
    idx_i, idx_j = torch.triu_indices(n_hap, n_hap, offset=1)
    delta = torch.abs(coefs[:, idx_i] - coefs[:, idx_j])

    v_ii = cov[:, idx_i, idx_i]
    v_jj = cov[:, idx_j, idx_j]
    v_ij = cov[:, idx_i, idx_j]
    var_diff = v_ii + v_jj - 2 * v_ij

    stat = delta.square() / var_diff
    pvals = 1 - torch.distributions.chi2.Chi2(df=1).cdf(stat)
    lod = stat / LOD

    return lod, pvals, delta


def _build_geno_haplo_pairs(y_full, haplo_variables, pair_idx):
    y_pairs = _as_tensors(*(y_full[:, g_idx] for g_idx, _ in pair_idx))
    if isinstance(y_pairs, torch.Tensor):
        y_pairs = y_pairs.unsqueeze(0)
    else:
        y_pairs = torch.stack(list(y_pairs))
    haplo_pairs = _as_tensors(*(haplo_variables[:, h_idx, :] for _, h_idx in pair_idx))
    if isinstance(haplo_pairs, torch.Tensor):
        haplo_pairs = haplo_pairs.unsqueeze(0)
    else:
        haplo_pairs = torch.stack(list(haplo_pairs))
    return y_pairs, haplo_pairs


def _refit_gene_haplotype_pairs(y_full, covariates, haplo_variables, pairs, device='cpu'):
    y_pairs, haplo_pairs = _build_geno_haplo_pairs(y_full, haplo_variables, pairs)
    covariates = _as_tensors(covariates)
    if not isinstance(covariates, torch.Tensor):
        covariates = list(covariates)[0]

    n_pairs, n, n_hvars = haplo_pairs.shape
    n_cov = covariates.shape[1]
    p = n_cov + n_hvars

    ht = haplo_pairs.transpose(1, 2)
    hth = torch.bmm(ht, haplo_pairs)
    htc = torch.bmm(ht, covariates.unsqueeze(0).expand(n_pairs, -1, -1))
    hty = torch.bmm(ht, y_pairs.unsqueeze(2).expand(n_pairs, -1, 1))

    ctc = covariates.T @ covariates
    cty = covariates.T @ y_pairs.T
    cty = cty.transpose(0, 1).unsqueeze(2)

    top = torch.cat([hth, htc], dim=2)
    bottom = torch.cat([htc.transpose(1, 2), ctc.expand(n_pairs, -1, -1)], dim=2)
    XtX = torch.cat([top, bottom], dim=1)
    Xty = torch.cat([hty, cty], dim=1)

    L = torch.linalg.cholesky(XtX)
    beta = torch.cholesky_solve(Xty, L).squeeze(2)

    yty = torch.sum(y_pairs ** 2, dim=1)
    Xty_flat = Xty.squeeze(2)
    beta_dot = torch.sum(beta * Xty_flat, dim=1)
    sse = yty - beta_dot
    df = n - p
    sigma2 = sse / df

    XtX_inv = torch.cholesky_inverse(L)
    cov = XtX_inv * sigma2.view(-1, 1, 1)

    return beta, cov


def _wald_tests(y_full, covariates, haplo_variables, pairs):
    beta, cov = _refit_gene_haplotype_pairs(y_full, covariates, haplo_variables, pairs)
    n_hvars = beta.shape[1] - covariates.shape[1]
    coefs = torch.cat([
      torch.zeros(beta.shape[0], 1, dtype=beta.dtype, device=beta.device),
      beta[:, :n_hvars], # trim off non-haplotype coefficients
    ], dim=-1)

    cov_hap = cov[:, :n_hvars+1, :n_hvars+1]
    # move intercept to the front
    cov = torch.cat([
        torch.cat([cov_hap[:, -1:, -1:], cov_hap[:, -1:, :-1]], dim=2),
        torch.cat([cov_hap[:, :-1, -1:], cov_hap[:, :-1, :-1]], dim=2)
    ], dim=1)

    lod_scores, pvals, deltas = _wald_test_param_pairs(coefs, cov)
    lod_scores = lod_scores.cpu().numpy()
    pvals = pvals.cpu().numpy()
    deltas = deltas.cpu().numpy()

    return lod_scores, pvals, deltas


def _hierarchical_cluster_deltas(deltas, threshold=0.25):
    return np.array([
        fcluster(linkage(d, method='complete'), t=threshold, criterion='distance')
        for d in deltas
    ])


def posthoc_wald_tests_and_clustering(sig_eqtl, exprs, covariates, haplotype_variables,
                                      ref_var_name='ref', cluster_threshold=0.25):
    gene_haplo_pairs = []
    for _, eqtl in sig_eqtl.iterrows():
        gene_idx = exprs._index_map[eqtl.gene_id]
        hap_idx = haplotype_variables._column_map[(eqtl.chrom_eqtl, eqtl.pos_eqtl)]
        gene_haplo_pairs.append((gene_idx, hap_idx))

    lod_scores, pvals, deltas = _wald_tests(
        exprs._data.T,
        covariates._data,
        haplotype_variables._data,
        pairs=gene_haplo_pairs
    )
    clusters = _hierarchical_cluster_deltas(deltas, cluster_threshold)

    var_names = [ref_var_name,] + haplotype_variables.variable_names
    ix_columns = pd.MultiIndex.from_tuples(it.combinations(var_names, r=2))
    lod_scores = pd.DataFrame(lod_scores, index=sig_eqtl.index, columns=ix_columns)
    pvals = pd.DataFrame(pvals, index=sig_eqtl.index, columns=ix_columns)    
    deltas = pd.DataFrame(deltas, index=sig_eqtl.index, columns=ix_columns)
    clusters = pd.DataFrame(clusters, index=sig_eqtl.index, columns=var_names)
    return lod_scores, pvals, deltas, clusters