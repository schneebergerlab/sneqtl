import numpy as np
from scipy import stats

try:
    import torch
    TORCH_AVAILABLE = True
except Exception:  # ImportError or others
    TORCH_AVAILABLE = False
    torch = None

LOD = 2 * np.log(10)
DEFAULT_RNG = np.random.default_rng(101)


def _compute_projections(X, y):
    """
    Compute cross-products used in linear model fitting.

    Parameters
    ----------
    X : torch.Tensor, shape (n, p)
        Design matrix.
    y : torch.Tensor, shape (n,)
        Response vector.

    Returns
    -------
    XtX : torch.Tensor, shape (p, p)
        Cross-product X^T X.
    Xty : torch.Tensor, shape (p,)
        Product X^T y.
    yty : torch.Tensor
        Inner product y^T y.
    """
    XtX = X.T @ X
    Xty = X.T @ y
    yty = (y * y).sum()
    return XtX, Xty, yty


def _fit_lm(X, y, return_projections=False):
    """
    Fit a linear model via Cholesky decomposition.

    Parameters
    ----------
    X : torch.Tensor, shape (n, p)
        Design matrix.
    y : torch.Tensor, shape (n,)
        Response vector.
    return_projections : bool, optional
        If True, return XtX, Xty, and yty along with model fit.

    Returns
    -------
    beta : torch.Tensor, shape (p,)
        Estimated model coefficients.
    sse : torch.Tensor
        Sum of squared residuals.
    resid : torch.Tensor, shape (n,)
        Residuals of the fit.
    XtX : torch.Tensor, shape (p, p), optional
        Cross-product X^T X (only if return_projections=True).
    Xty : torch.Tensor, shape (p,), optional
        Product X^T y (only if return_projections=True).
    yty : torch.Tensor, optional
        Inner product y^T y (only if return_projections=True).
    """
    XtX, Xty, yty = _compute_projections(X, y)
    L = torch.linalg.cholesky(XtX)
    beta = torch.cholesky_solve(Xty.unsqueeze(1), L).squeeze(1)
    resid = y - X @ beta
    sse = torch.sum(resid ** 2)
    if return_projections:
        return beta, sse, resid, XtX, Xty, yty
    else:
        return beta, sse, resid


def _ll(sse, n):
    """
    Compute log-likelihood of a linear model.

    Parameters
    ----------
    sse : torch.Tensor
        Sum of squared residuals.
    n : int
        Number of observations.

    Returns
    -------
    ll : torch.Tensor
        Log-likelihood of the model.
    """
    return -0.5 * n * (np.log(2 * np.pi) + torch.log(sse / n) + 1)


def _lrt(ll_nest, ll_full):
    """
    Compute LOD score from nested and full model log-likelihoods.

    Parameters
    ----------
    ll_nest : torch.Tensor
        Log-likelihood of the nested model.
    ll_full : torch.Tensor
        Log-likelihood of the full model.

    Returns
    -------
    lod : torch.Tensor
        Logarithm of odds score.
    """
    chi2 = -2 * (ll_nest - ll_full)
    return chi2 / LOD


def _get_params(covariates, haplo_variables, haplo_variables_alt):
    """
    Extract key model dimensions and degrees of freedom.

    Parameters
    ----------
    covariates : np.ndarray
    haplo_variables : np.ndarray
    haplo_variables_alt : np.ndarray or None

    Returns
    -------
    n, n_cov, n_hap, n_hvars, n_hvars_alt, df_resid, df_resid_alt : int or None
        Extracted model parameters.
    """
    n, n_hap, n_hvars = haplo_variables.shape
    n_cov = covariates.shape[1]
    df_resid = n - (n_cov + n_hvars)
    if haplo_variables_alt is not None:
        _, _, n_hvars_alt = haplo_variables_alt.shape
        df_resid_alt = n - (n_cov + n_hvars_alt)
    else:
        n_hvars_alt = None
        df_resid_alt = None
    return n, n_cov, n_hap, n_hvars, n_hvars_alt, df_resid, df_resid_alt


def _as_tensors(*args, dtype=torch.float64, device='cpu'):
    """
    Convert input arrays to PyTorch tensors.

    Parameters
    ----------
    *args : array-like
        Arrays to convert to tensors.
    dtype : torch.dtype, optional
        Desired tensor dtype (default: torch.float64).
    device : str, optional
        Device to place tensors on (default: 'cpu').

    Returns
    -------
    tensors : torch.Tensor or tuple of torch.Tensor
        Converted tensor(s). Returns a single tensor if one argument is given.
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is not available")
    if len(args) == 1:
        return torch.tensor(args[0], dtype=dtype, device=device)
    return (torch.tensor(arr, dtype=dtype, device=device) for arr in args)


def _fit_batch(*, y, covar, hap, ctc, cty, yty, df_resid):
    """
    Fit linear models in batch over haplotype slices.

    Parameters
    ----------
    y : torch.Tensor, shape (n,)
        Response vector.
    covar : torch.Tensor, shape (n, n_cov)
        Covariate matrix.
    hap : torch.Tensor, shape (n, batch_size, n_hvars)
        Haplotype variables in batch.
    ctc : torch.Tensor, shape (n_cov, n_cov)
        Precomputed covar^T covar.
    cty : torch.Tensor, shape (n_cov,)
        Precomputed covar^T y.
    yty : torch.Tensor
        Precomputed y^T y.
    df_resid : int
        Degrees of freedom for residuals.

    Returns
    -------
    beta : torch.Tensor, shape (batch_size, p)
        Fitted coefficients.
    sse : torch.Tensor, shape (batch_size,)
        Residual sums of squares.
    param_lods : torch.Tensor, shape (batch_size, n_hvars)
        Parameter-level LOD scores.
    """
    n, bsz, n_hvars = hap.shape
    n_cov = covar.shape[1]

    hb = hap.permute(1, 2, 0)  # (b, n_hvars, n)

    hth = torch.bmm(hb, hap.permute(1, 0, 2))  # (b, n_hvars, n_hvars)
    htc = hb @ covar                         # (b, n_hvars, n_cov)
    hty = hb @ y                           # (b, n_hvars)

    # Assemble XtX blocks
    top = torch.cat([hth, htc], dim=2)  # (b, n_hvars, p)
    bottom = torch.cat(
        [htc.transpose(1, 2), ctc.repeat(bsz, 1, 1)], dim=2
    )
    XtX = torch.cat([top, bottom], dim=1)  # (b, p, p)

    # Assemble XtY
    Xty = torch.cat([hty, cty.repeat(bsz, 1)], dim=1)  # (b, p)

    # Solve via Cholesky
    L = torch.linalg.cholesky(XtX)  # (b, p, p)
    beta = torch.cholesky_solve(Xty.unsqueeze(2), L).squeeze(2)  # (b, p)

    sse = yty - (beta * Xty).sum(dim=1)

    # Standard errors from Cholesky inverse
    XtX_inv = torch.cholesky_inverse(L)  # (b, p, p)
    diag = torch.diagonal(XtX_inv, dim1=1, dim2=2)  # (b, p)
    se = torch.sqrt(diag * (sse / df_resid).unsqueeze(1))  # (b, p)

    t_stats = beta / se  # (b, p)
    param_chi2 = t_stats[:, :n_hvars] ** 2
    param_lods = param_chi2 / LOD

    return beta, sse, param_lods


def _process_batch(hap_idx, haplo_variables, haplo_variables_alt, *,
                   ll_nest, df_resid, df_resid_alt, **fit_kwargs):
    """
    Process a batch of haplotype variables to compute regression metrics.

    Parameters
    ----------
    hap_idx : list or slice
        indices used to slice the haplo_variables and haplo_variables_alt tensors
    haplo_variables : array-like, shape (n, n_hap, n_hvars)
        Haplotype encodings to test.
    haplo_variables_alt : array-like, shape (n, n_hap, n_hvars_alt), optional
        Alternative haplotype encoding (for nested model comparison).
    ll_nest : float
        Log-likihood of nested (covariate only) model
    df_resid : int
        Degrees of freedom for residuals of haplo_variables.
    df_resid_alt : int
        Degrees of freedom for residuals of haplo_variables_alt.
    y : torch.Tensor, shape (n,)
        Response vector.
    covar : torch.Tensor, shape (n, n_cov)
        Covariate matrix.
    ctc : torch.Tensor, shape (n_cov, n_cov)
        Precomputed covar^T covar.
    cty : torch.Tensor, shape (n_cov,)
        Precomputed covar^T y.
    yty : torch.Tensor
        Precomputed y^T y.

    Returns
    -------
    beta : torch.Tensor, shape (n_tested_haps, n_hvars)
        Estimated coefficients for the main model.
    lod_scores : torch.Tensor, shape (n_tested_haps,)
        LOD scores of the main model vs nested model.
    param_lods : torch.Tensor, shape (n_tested_haps, n_hvars)
        Per-parameter LOD scores for the main model.
    beta_alt : torch.Tensor or None, shape (n_tested_haps, n_hvars_alt)
        Estimated coefficients for the alternative model, if provided.
    lod_scores_alt : torch.Tensor or None, shape (n_tested_haps,)
        LOD scores for the alternative model vs main model.
    param_lods_alt : torch.Tensor or None, shape (n_tested_haps, n_hvars_alt)
        Per-parameter LOD scores for the alternative model.
    """
    n = haplo_variables.shape[0]
    hap = _as_tensors(haplo_variables[:, hap_idx, :])
    beta, sse, param_lods = _fit_batch(
        hap=hap, df_resid=df_resid, **fit_kwargs
    )
    ll_main = _ll(sse, n)
    lod_scores = _lrt(ll_nest, ll_main)

    if haplo_variables_alt is not None:
        hap_alt = _as_tensors(haplo_variables_alt[:, hap_idx, :])
        beta_alt, sse_alt, param_lods_alt = _fit_batch(
            hap=hap_alt, df_resid=df_resid_alt, **fit_kwargs
        )
        lod_scores_alt = _lrt(ll_main, _ll(sse_alt, n))
    else:
        beta_alt = None
        lod_scores_alt = None
        param_lods_alt = None
    return beta, lod_scores, param_lods, beta_alt, lod_scores_alt, param_lods_alt


def _permute_residuals(X, beta, resid, *, rng, n_perms=50):
    """
    Generate permuted residual phenotypes using Freedman-Lane strategy.

    Parameters
    ----------
    X : array-like of shape (n, n_cov)
        Covariate matrix.
    beta : torch.Tensor of shape (n_cov,)
        Estimated coefficients.
    resid : torch.Tensor of shape (n,)
        Residuals from the linear model fit of `y` ~ `X`.
    rng : np.random.Generator
        Random number generator instance.
    n_perms : int
        Number of permutations.

    Yields
    ------
    y_perm : np.ndarray of shape (n,)
        Permuted phenotype preserving covariate structure.
    """
    t_rng = torch.Generator()
    t_rng.manual_seed(int(rng.integers(0, 2**31 - 1)))
    n = resid.shape[0]
    fitted = X @ beta
    for _ in range(n_perms):
        permuted = resid[t_rng.randperm(n)]
        yield fitted + permuted


@torch.no_grad()
def _calculate_null_lod_dists(y, covariates, haplo_variables, haplo_variables_alt=None,
                              n_perm=50, n_haps_per_perm=200, alpha=0.01, rng=DEFAULT_RNG):
    """
    Estimate empirical LOD score thresholds via residual permutation.

    Parameters
    ----------
    y : array-like, shape (n,)
        Response vector, e.g. gene expression.
    covariates : array-like, shape (n, n_cov)
        Covariate matrix.
    haplo_variables : array-like, shape (n, n_hap, n_hvars)
        Haplotype encodings to test.
    haplo_variables_alt : array-like, shape (n, n_hap, n_hvars_alt), optional
        Alternative haplotype encoding (for nested model comparison).
    n_perms : int
        Number of phenotype permutations.
    n_haps_per_perm : int
        Number of loci to sample per permutation.
    alpha : float
        Significance threshold.
    rng : np.random.Generator or None
        RNG instance (optional).

    Returns
    -------
    lod_threshold : float
        Empirical LOD score threshold for full model.
    param_lod_thresh : float
        Empirical maximum per-parameter LOD threshold.
    lod_threshold_alt : float
        Empirical LOD score threshold for alternative model.
    param_lod_thresh_alt : float
        Empirical maximum per-parameter LOD threshold for alternative model.
    """
    (n, n_cov, n_hap,
     n_hvars, n_hvars_alt,
     df_resid, df_resid_alt) = _get_params(
        covariates, haplo_variables, haplo_variables_alt
     )

    beta_nest, sse_nest, resid_nest = _fit_lm(covariates, y)

    lod_score_dist = []
    param_lod_dist = []
    if haplo_variables_alt is not None:
        lod_score_alt_dist = []
        param_lod_alt_dist = []
    else:
        lod_score_alt_dist = None
        param_lod_alt_dist = None

    for y_perm in _permute_residuals(covariates, beta_nest, resid_nest, rng=rng, n_perms=n_perms):
        _, sse_nest, _, ctc, ctyp, yptyp = _fit_lm(covariates, y_perm, return_projections=True)
        ll_nest = _ll(sse_nest, n)

        hap_idx = rng.choice(n_hap, size=n_haps_per_perm, replace=False)
        (_, lod_scores, param_lods,
         _, lod_scores_alt, param_lods_alt) = _process_batch(
            hap_idx, haplo_variables, haplo_variables_alt,
            y=y, covariates=covariates, ll_nest=ll_nest,
            ctc=ctc, cty=ctyp, yty=yptyp,
            df_resid=df_resid, df_resid_alt=df_resid_alt
        )
        lod_score_dist.append(lod_scores)
        param_lod_dist.append(torch.max(param_lods, dim=1)[0])

        if haplo_variables_alt is not None:
            lod_score_alt_dist.append(lod_scores_alt)
            param_lod_alt_dist.append(torch.max(param_lods_alt, dim=1)[0])

    lod_score_dist = torch.sort(torch.cat(lod_score_dist)).values
    param_lod_dist = torch.sort(torch.cat(param_lod_dist)).values
    if haplo_variables_alt is not None:
        lod_score_alt_dist = torch.sort(torch.cat(lod_score_dist)).values
        param_lod_alt_dist = torch.sort(torch.cat(param_lod_dist)).values
    return lod_score_dist, param_lod_dist, lod_score_alt_dist, param_lod_alt_dist


def _empirical_p_value(obs_lod, null_lod):
    n = null_lod.shape[0]
    shape = obs_lod.shape
    indices = torch.bucketize(obs_lod.ravel(), null_lod, right=False)
    pval = (n - indices + 1).float() / (n + 1)
    return pval.reshape(shape)


def fit_models(y, covariates, haplo_variables, haplo_variables_alt=None, batch_size=500,
               n_perm=50, n_haps_per_perm=200, alpha=0.01, rng=DEFAULT_RNG):
    """
    Fit linear models for QTL analysis using PyTorch.

    Parameters
    ----------
    y : array-like, shape (n,)
        Response vector, e.g. gene expression.
    covariates : array-like, shape (n, n_cov)
        Covariate matrix.
    haplo_variables : array-like, shape (n, n_hap, n_hvars)
        Haplotype encodings to test.
    haplo_variables_alt : array-like, shape (n, n_hap, n_hvars_alt), optional
        Alternative haplotype encoding (for nested model comparison).
    batch_size : int, optional
        Number of haplotypes to process in each batch.
    n_perms : int
        Number of phenotype permutations.
    n_haps_per_perm : int
        Number of loci to sample per permutation.
    alpha : float
        Significance threshold.
    rng : np.random.Generator or None
        RNG instance (optional).

    Returns
    -------
    results : np.ndarray, shape (n_hap, n_columns)
        Concatenated model statistics per haplotype, including:
        - Main model LOD
        - Main model parameter LODs and betas
        - (optional) Alt model LODs and betas if `haplo_variables_alt` is given
    """
    y, covariates = _as_tensors(y, covariates)

    (n, n_cov, n_hap,
     n_hvars, n_hvars_alt,
     df_resid, df_resid_alt) = _get_params(
        covariates, haplo_variables, haplo_variables_alt
     )

    # start by calculating the permutation-test based thresholds for main and alt models
    (lod_score_null_dist, param_lod_null_dist,
     lod_score_alt_null_dist, param_lod_alt_null_dist) = _calculate_permutation_thresholds(
        y, covariates, haplo_variables, haplo_variables_alt=haplo_variables_alt,
        n_perm=n_perm, n_haps_per_perm=n_haps_per_perm, alpha=alpha, rng=rng
    )

    # Nested model with covariates only
    _, sse_nest, _, ctc, cty, yty = _fit_lm(covariates, y, return_projections=True)
    ll_nest = _ll(sse_nest, n)

    results = []

    for start in range(0, n_hap, batch_size):
        end = min(start + batch_size, n_hap)

        hap_idx = slice(start, end)
        (beta, lod_scores, param_lods,
         beta_alt, lod_scores_alt, param_lods_alt) = _process_batch(
            hap_idx, haplo_variables, haplo_variables_alt,
            y=y, covariates=covariates, ll_nest=ll_nest,
            ctc=ctc, cty=cty, yty=yty,
            df_resid=df_resid, df_resid_alt=df_resid_alt
        )
        lrt_pval = _empirical_p_value(lod_scores, lod_score_null_dist)
        param_pval = _empirical_p_value(param_lods, param_lod_null_dist)

        batch_res = torch.cat([
            lod_scores[:, None],
            lrt_pval[:, None],
            param_lods,
            param_pvals,
            beta,
        ], dim=1)

        if haplo_variables_alt is not None:
            lrt_pval_alt = _empirical_p_value(lod_scores_alt, lod_score_alt_null_dist)
            param_pval_alt = _empirical_p_value(param_lods_alt, param_lod_alt_null_dist)
            batch_res = torch.cat([
                batch_res,
                lod_scores_alt[:, None],
                lrt_pval_alt[:, None],
                param_lods_alt,
                param_pval_alt,
                beta_alt
            ], dim=1)

        results.append(batch_res)

    return torch.cat(results, dim=0).cpu().numpy()
