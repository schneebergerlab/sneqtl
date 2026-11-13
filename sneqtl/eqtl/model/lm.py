import numpy as np
from scipy import stats
import torch


LOD = 2 * np.log(10)
DEFAULT_RNG = np.random.default_rng(101)


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
    if len(args) == 1:
        return torch.tensor(args[0], dtype=dtype, device=device)
    return (torch.tensor(arr, dtype=dtype, device=device) for arr in args)


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
    n_hap, n, n_hvars = haplo_variables.shape
    n_cov = covariates.shape[1]
    df_resid = n - (n_cov + n_hvars)
    if haplo_variables_alt is not None:
        _, _, n_hvars_alt = haplo_variables_alt.shape
        df_resid_alt = n - (n_cov + n_hvars_alt)
    else:
        n_hvars_alt = None
        df_resid_alt = None
    return n, n_cov, n_hap, n_hvars, n_hvars_alt, df_resid, df_resid_alt


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
    yty = (y * y).sum(dim=0)
    return XtX, Xty, yty


def _fit_lm(X, y, return_projections=False):
    """
    Fit one or more ordinary least squares models by Cholesky solve.

    Parameters
    ----------
    X : torch.Tensor, shape (n, p)
        Design matrix.
    y : torch.Tensor, shape (n,) or (n, k)
        Response vector or matrix of responses.
    return_projections : bool, optional
        If True, also return XtX, Xty and yty.

    Returns
    -------
    beta : torch.Tensor
        Estimated coefficients with shape (p,) for a single response or
        (p, k) for multiple responses.
    sse : torch.Tensor
        Residual sum of squares for each response.
    resid : torch.Tensor
        Residual vector or matrix.
    XtX : torch.Tensor, optional
        Cross-product X.T @ X.
    Xty : torch.Tensor, optional
        Cross-product X.T @ y.
    yty : torch.Tensor, optional
        Column-wise sum of squares of y.
    """
    is_1d = (y.ndim == 1)
    if is_1d:
        y = y.unsqueeze(1)  # (n, 1)

    XtX, Xty, yty = _compute_projections(X, y)
    L = torch.linalg.cholesky(XtX)

    beta = torch.cholesky_solve(Xty, L)          # (p, k)
    resid = y - X @ beta                         # (n, k)
    sse = (resid ** 2).sum(dim=0)                # (k,)

    if is_1d:
        beta = beta.squeeze(1)                   # (p,)
        resid = resid.squeeze(1)                 # (n,)
        sse = sse.squeeze(0)                     # scalar
        Xty = Xty.squeeze(1)                     # (p,)
        yty = yty.squeeze(0)                     # scalar

    if return_projections:
        return beta, sse, resid, XtX, Xty, yty
    return beta, sse, resid


def _diag_inv(L):
    """
    Compute the diagonal of (L L.T)^(-1) from a batched Cholesky factor.

    Parameters
    ----------
    L : torch.Tensor, shape (bsz, p, p)
        Lower-triangular Cholesky factors.

    Returns
    -------
    torch.Tensor, shape (bsz, p)
        Diagonal of the inverse covariance matrix for each batch element.
    """
    # L: (bsz, p, p)
    bsz, p, _ = L.shape
    I = torch.eye(p, dtype=L.dtype, device=L.device).expand(bsz, p, p)

    # Solve L X = I  →  X = L^{-1}
    Linv = torch.linalg.solve_triangular(L, I, upper=False)

    # diag(A^{-1}) = column-wise sum of squares of Linv
    diag_inv = (Linv ** 2).sum(dim=1)  # (bsz, p)

    return diag_inv


def _compute_var_covar_cholesky(covars, hb, ctc=None):
    """
    Construct batched normal-equation matrices and compute their Cholesky factors.

    Parameters
    ----------
    covars : torch.Tensor, shape (n, n_cov)
        Covariate design matrix.
    hb : torch.Tensor, shape (bsz, n_hvars, n)
        Haplotype design matrix
    ctc : torch.Tensor, shape (n_cov, n_cov), optional
        Precomputed covars.T @ covars.

    Returns
    -------
    torch.Tensor, shape (bsz, p, p)
        Batched lower-triangular Cholesky factors of X.T @ X, where
        p = n_cov + n_hvars.
    """
    bsz = hb.shape[0]
    if ctc is None:
        ctc = covars.T @ covars

    hth = torch.bmm(hb, hb.transpose(1, 2))  # (bsz, n_hvars, n_hvars)
    htc = hb @ covars                         # (bsz, n_hvars, n_cov)

    # Assemble XtX blocks
    top = torch.cat([hth, htc], dim=2)  # (bsz, n_hvars, p)
    bottom = torch.cat(
        [htc.transpose(1, 2), ctc.repeat(bsz, 1, 1)], dim=2
    )
    XtX = torch.cat([top, bottom], dim=1)  # (bsz, p, p)

    # Compute Cholesky decomposition once
    L = torch.linalg.cholesky(XtX)  # (bsz, p, p)
    return L


def _create_empty_if_none(arr, shape):
    if arr is None:
        return np.empty(shape)
    return arr


def precompute_var_covar_cholesky(covariates, haplo_variables, *, haplo_variables_alt=None,
                                  L_out=None, diag_out=None, L_out_alt=None, diag_out_alt=None,
                                  batch_size=500):
    """
    Precompute and store per-locus Cholesky factors and inverse diagonals.

    This function builds the phenotype-invariant linear algebra objects used
    by the batched regression code, allowing them to be reused across many
    phenotypes.

    Parameters
    ----------
    covariates : array-like, shape (n, n_cov)
        Covariate design matrix.
    haplo_variables : array-like, shape (n_hap, n, n_hvars)
        Main haplotype design tensor.
    haplo_variables_alt : array-like, shape (n_hap, n, n_hvars_alt), optional
        Alternative haplotype design tensor.
    L_out : ndarray-like, optional
        Output array or memmap for main-model Cholesky factors with shape
        (n_hap, p, p).
    diag_out : ndarray-like, optional
        Output array or memmap for main-model inverse diagonals with shape
        (n_hap, p).
    L_out_alt : ndarray-like, optional
        Output array or memmap for alternative-model Cholesky factors.
    diag_out_alt : ndarray-like, optional
        Output array or memmap for alternative-model inverse diagonals.
    batch_size : int, optional
        Number of loci to process at once.

    Returns
    -------
    L_out : ndarray-like
        Stored Cholesky factors for the main model.
    diag_out : ndarray-like
        Stored inverse diagonals for the main model.
    L_out_alt : ndarray-like or None
        Stored Cholesky factors for the alternative model.
    diag_out_alt : ndarray-like or None
        Stored inverse diagonals for the alternative model.
    """
    (n, n_cov, n_hap,
     n_hvars, n_hvars_alt,
     *_) = _get_params(
        covariates, haplo_variables, haplo_variables_alt
     )
    p = n_cov + n_hvars

    L_out = _create_empty_if_none(L_out, (n_hap, p, p))
    diag_out = _create_empty_if_none(diag_out, (n_hap, p))
    if haplo_variables_alt is not None:
        p_alt = n_cov + n_hvars_alt
        L_out_alt = _create_empty_if_none(L_out_alt, (n_hap, p_alt, p_alt))
        diag_out_alt = _create_empty_if_none(diag_out_alt, (n_hap, p_alt))
    else:
        L_out_alt, diag_out_alt = None, None

    covariates = _as_tensors(covariates)
    ctc = ctc = covariates.T @ covariates

    for start in range(0, n_hap, batch_size):
        end = min(start + batch_size, n_hap)
        hb = _as_tensors(haplo_variables[start:end, :, :]).transpose(1, 2)
        L_batch = _compute_var_covar_cholesky(covariates, hb, ctc)
        diag_batch = _diag_inv(L_batch)
        L_out[start:end, :, :] = L_batch.cpu().numpy()
        diag_out[start:end, :] = diag_batch.cpu().numpy()
        if haplo_variables_alt is not None:
            hb_alt = _as_tensors(haplo_variables_alt[start:end, :, :])
            hb_alt = hb_alt.transpose(1, 2)
            L_batch_alt = _compute_var_covar_cholesky(covariates, hb_alt, ctc)
            diag_batch_alt = _diag_inv(L_batch_alt)
            L_out_alt[start:end, :, :] = L_batch_alt.cpu().numpy()
            diag_out_alt[start:end, :] = diag_batch_alt.cpu().numpy()

    return L_out, diag_out, L_out_alt, diag_out_alt


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


def _lrt(ll_nest, ll_full, df):
    """
    Compute LOD score from nested and full model log-likelihoods.

    Parameters
    ----------
    ll_nest : torch.Tensor
        Log-likelihood of the nested model.
    ll_full : torch.Tensor
        Log-likelihood of the full model.
    df : int, optional
        Degrees of freedom for chi-squared test.

    Returns
    -------
    lod : torch.Tensor
        Logarithm of odds score.
    pval : torch.Tensor, optional
        P-values from chi-squared distribution
    """
    chi2 = -2 * (ll_nest - ll_full)
    chi2 = torch.clamp(chi2, min=0.0)
    lod = chi2 / LOD
    pval = torch.special.gammaincc(
        torch.as_tensor(df / 2, dtype=chi2.dtype, device=chi2.device),
        chi2 / 2
    )
    return lod, pval


def _wald_test(beta, se):
    """
    Compute two-sided Wald test p-values for model coefficients.

    Parameters
    ----------
    beta : torch.Tensor
        Estimated coefficients. Shape can be (...,) or (..., k).
    se : torch.Tensor
        Standard errors for coefficients, same shape as beta.

    Returns
    -------
    pval : torch.Tensor
        Two-sided Wald test p-values, same shape as beta.
    """
    se = torch.clamp(se, min=torch.finfo(se.dtype).tiny)
    z = beta / se
    abs_z = torch.abs(z)
    sqrt2 = torch.as_tensor(np.sqrt(2.0), dtype=abs_z.dtype, device=abs_z.device)
    pval = torch.erfc(abs_z / sqrt2)
    return pval


def _fit_batch(*, y_perms, covariates, hap, ctc, cty_perms, yty_perms, df_resid,
               L_precomp=None, diag_precomp=None):
    """
    Fit linear models in batch over haplotype slices.

    Parameters
    ----------
    y_perms : torch.Tensor, shape (n, n_perm + 1)
        Observed phenotype in column 0 and permuted phenotypes in later columns.
    covariates : torch.Tensor, shape (n, n_cov)
        Covariate design matrix.
    hap : torch.Tensor, shape (bsz, n, n_hvars)
        Haplotype variables for the current batch of loci.
    ctc : torch.Tensor, shape (n_cov, n_cov)
        Precomputed covariates.T @ covariates.
    cty_perms : torch.Tensor, shape (n_cov, n_perm + 1)
        Precomputed covariates.T @ y_perms.
    yty_perms : torch.Tensor, shape (n_perm + 1,)
        Precomputed column-wise sums of squares of y_perms.
    df_resid : int
        Residual degrees of freedom for the fitted model.
    L_precomp : torch.Tensor, optional
        Precomputed Cholesky factors for this batch with shape (bsz, p, p).
    diag_precomp : torch.Tensor, optional
        Precomputed diagonal of (X.T @ X)^(-1) for this batch with shape (bsz, p).

    Returns
    -------
    coefs : torch.Tensor, shape (bsz, n_hvars, n_perm + 1)
        Estimated haplotype coefficients.
    sse : torch.Tensor, shape (bsz, n_perm + 1)
        Residual sums of squares for each locus and phenotype column.
    param_lods : torch.Tensor, shape (bsz, n_hvars, n_perm + 1)
        Parameter-level LOD scores derived from Wald statistics.
    param_pvals : torch.Tensor, shape (bsz, n_hvars, n_perm + 1)
        Parameter-level two-sided Wald p-values.
    """
    bsz, n, n_hvars = hap.shape
    n_cov = covariates.shape[1]
    p = n_cov + n_hvars

    hb = hap.transpose(1, 2)  # (bsz, n_hvars, n)
    hty = hb @ y_perms                           # (bsz, n_hvars, n_perms + 1)
    Xty = torch.cat([hty, cty_perms.unsqueeze(0).expand(bsz, -1, -1)], dim=1)  # (bsz, p)

    if L_precomp is None:
        L = _compute_var_covar_cholesky(covariates, hb, ctc=ctc)
    else:
        assert L_precomp.shape == (bsz, p, p)
        L = L_precomp

    if diag_precomp is None:
        diag = _diag_inv(L)
    else:
        assert diag_precomp.shape == (bsz, p)
        diag = diag_precomp

    # Solve via Cholesky
    beta = torch.cholesky_solve(Xty, L).squeeze(2)  # (bsz, p, n_perms + 1)

    # (batch_size, n_perms)
    sse = yty_perms.unsqueeze(0) - (beta * Xty).sum(dim=1)

    se = torch.sqrt(diag.unsqueeze(-1) * (sse.unsqueeze(1) / df_resid))  # (bsz, p, n_perms + 1)

    # Compute t-statistics, LODs and wald pvals
    t_stats = beta / se  # (bsz, p, n_perms)
    param_chi2 = t_stats[:, :n_hvars, :] ** 2  # (bsz, n_hvars, n_perms + 1)
    param_lods = param_chi2 / LOD
    coefs = beta[:, :n_hvars, :]  # (batch_size, n_hvars, n_perms + 1)
    param_pvals = _wald_test(coefs, se[:, :n_hvars, :])

    return coefs, sse, param_lods, param_pvals


def _process_batch(hap_idx, haplo_variables, haplo_variables_alt, *,
                   ll_nest, df_resid, df_resid_alt,
                   L_precomp=None, diag_precomp=None,
                   L_precomp_alt=None, diag_precomp_alt=None, **fit_kwargs):
    """
    Fit the main and optional alternative models for one batch of loci.

    Parameters
    ----------
    hap_idx : slice
        Slice selecting the loci in the current batch.
    haplo_variables : array-like, shape (n_hap, n, n_hvars)
        Main haplotype design tensor.
    haplo_variables_alt : array-like, shape (n_hap, n, n_hvars_alt), optional
        Alternative haplotype design tensor.
    ll_nest : torch.Tensor
        Log-likelihood of the reduced model for the observed and permuted
        phenotypes, typically shape (1, n_perm + 1).
    df_resid : int
        Residual degrees of freedom for the main model.
    df_resid_alt : int or None
        Residual degrees of freedom for the alternative model.
    L_precomp : array-like, optional
        Precomputed main-model Cholesky factors for all loci.
    diag_precomp : array-like, optional
        Precomputed main-model inverse diagonals for all loci.
    L_precomp_alt : array-like, optional
        Precomputed alternative-model Cholesky factors for all loci.
    diag_precomp_alt : array-like, optional
        Precomputed alternative-model inverse diagonals for all loci.
    **fit_kwargs
        Additional keyword arguments passed to `_fit_batch`.

    Returns
    -------
    lod_scores : torch.Tensor
        Omnibus LOD scores for the main model.
    lrt_pvals : torch.Tensor
        Likelihood-ratio p-values for the main model.
    param_lods : torch.Tensor
        Parameter-level LOD scores for the main model.
    coefs : torch.Tensor
        Haplotype coefficients for the main model.
    param_pvals : torch.Tensor
        Parameter-level Wald p-values for the main model.
    lod_scores_alt : torch.Tensor or None
        Omnibus LOD scores for the alternative model relative to the main model.
    lrt_pvals_alt : torch.Tensor or None
        Likelihood-ratio p-values for the alternative model.
    param_lods_alt : torch.Tensor or None
        Parameter-level LOD scores for the alternative model.
    coefs_alt : torch.Tensor or None
        Haplotype coefficients for the alternative model.
    param_pvals_alt : torch.Tensor or None
        Parameter-level Wald p-values for the alternative model.
    """
    _, n, n_hvars = haplo_variables.shape
    hap_batch = _as_tensors(haplo_variables[hap_idx, :, :])
    if L_precomp is not None:
        L_precomp_batch = _as_tensors(L_precomp[hap_idx, :])
    else:
        L_precomp_batch = None
    if diag_precomp is not None:
        diag_precomp_batch = _as_tensors(diag_precomp[hap_idx, :])
    else:
        diag_precomp_batch = None
    coefs, sse, param_lods, param_pvals = _fit_batch(
        hap=hap_batch,
        df_resid=df_resid,
        L_precomp=L_precomp_batch,
        diag_precomp=diag_precomp_batch,
        **fit_kwargs
    )
    ll_main = _ll(sse, n)
    lod_scores, lrt_pvals = _lrt(ll_nest, ll_main, df=n_hvars)

    if haplo_variables_alt is not None:
        *_, n_hvars_alt = haplo_variables_alt.shape
        hap_alt = _as_tensors(haplo_variables_alt[hap_idx, :, :])
        if L_precomp_alt is not None:
            L_precomp_alt_batch = _as_tensors(L_precomp_alt[hap_idx, :])
        else:
            L_precomp_alt_batch = None
        if diag_precomp_alt is not None:
            diag_precomp_alt_batch = _as_tensors(diag_precomp_alt[hap_idx, :])
        else:
            diag_precomp_alt_batch = None
        coefs_alt, sse_alt, param_lods_alt, param_pvals_alt = _fit_batch(
            hap=hap_alt, df_resid=df_resid_alt,
            L_precomp=L_precomp_alt_batch,
            diag_precomp=diag_precomp_alt_batch,
            **fit_kwargs
        )
        lod_scores_alt, lrt_pvals_alt = _lrt(ll_main, _ll(sse_alt, n), df=n_hvars_alt - n_hvars)
    else:
        coefs_alt = None
        lod_scores_alt = None
        param_lods_alt = None
        param_pvals_alt = None
        lrt_pvals_alt = None
    return (lod_scores, lrt_pvals, param_lods, coefs, param_pvals,
            lod_scores_alt, lrt_pvals_alt, param_lods_alt, coefs_alt, param_pvals_alt)


def _generate_permuted_phenotypes(X, beta, resid, *, rng, n_perm=50):
    """
    Generate randomised phenotypes using residual permutation.

    Parameters
    ----------
    X : torch.Tensor, shape (n, n_cov)
        Reduced-model design matrix.
    beta : torch.Tensor, shape (n_cov,)
        Reduced-model coefficient estimates.
    resid : torch.Tensor, shape (n,)
        Reduced-model residuals.
    rng : np.random.Generator
        Random number generator.
    n_perm : int, optional
        Number of phenotype permutations to generate.

    Returns
    -------
    y_perms : torch.Tensor, shape (n, n_perm + 1)
        Column 0 contains the observed phenotype reconstructed as fitted plus
        residuals. Remaining columns contain fitted values plus permuted
        residuals.
    """
    t_rng = torch.Generator(device=resid.device)
    t_rng.manual_seed(int(rng.integers(0, 2**31 - 1)))

    n = resid.shape[0]
    fitted = X @ beta

    perms = torch.empty((n, n_perm), dtype=resid.dtype, device=resid.device)
    for k in range(n_perm):
        perms[:, k] = resid[torch.randperm(n, generator=t_rng, device=resid.device)]

    y_perms = torch.empty((n, n_perm + 1), dtype=resid.dtype, device=resid.device)
    y_perms[:, 0] = fitted + resid
    y_perms[:, 1:] = fitted[:, None] + perms
    return y_perms


def _minP(pvals):
    return torch.amin(pvals, dim=0)  # (n_perm,) or (n_vars, n_perm)


def _calculate_minP_threshold(pval_perms, minP_threshold, bonf_correction=1.0):
    """
    Compute empirical minP threshold(s) from permutation p-values.

    Parameters
    ----------
    pval_perms : torch.Tensor
        Empirical minP null distribution with permutations on the last axis.
        Can be one-dimensional, shape (n_perm,), or two-dimensional, shape
        (n_tests, n_perm).
    minP_threshold : float
        Lower-tail probability used to define the empirical threshold.
    bonf_correction : float, optional
        Bonferroni multiplier applied on the p-value scale.

    Returns
    -------
    torch.Tensor
        Empirical threshold(s). Returns a scalar for 1D input or one threshold
        per row for 2D input.
    """
    n_perms = pval_perms.shape[-1]
    vals = torch.sort(pval_perms, dim=-1).values
    t_idx = int(np.ceil(minP_threshold * n_perms)) - 1
    t_idx = max(0, min(t_idx, n_perms - 1))
    thresh = vals[..., t_idx]
    return torch.clamp(thresh / bonf_correction, max=1.0)


def fit_models(y, covariates, haplo_variables, *, haplo_variables_alt=None, batch_size=500,
               n_perm=1000, minP_threshold=0.01, bonf_correction=None,
               L_precomp=None, diag_precomp=None,
               L_precomp_alt=None, diag_precomp_alt=None,
               rng=DEFAULT_RNG):
    """
    Fit batched haplotype association models with permutation-based thresholds.

    This function fits the main model, optionally an alternative nested model,
    computes primary likelihood-ratio statistics, parameter-level Wald statistics,
    and empirical minP thresholds based on phenotype permutations.

    Parameters
    ----------
    y : array-like, shape (n,)
        Observed phenotype vector.
    covariates : array-like, shape (n, n_cov)
        Covariate design matrix.
    haplo_variables : array-like, shape (n_hap, n, n_hvars)
        Main haplotype design tensor.
    haplo_variables_alt : array-like, shape (n_hap, n, n_hvars_alt), optional
        Alternative haplotype design tensor.
    batch_size : int, optional
        Number of loci to process at once.
    n_perm : int, optional
        Number of phenotype permutations.
    minP_threshold : float, optional
        Target lower-tail probability used to derive empirical minP thresholds.
    bonf_correction : float, optional
        Bonferroni factor applied to the empirical p-value thresholds, typically
        for correction across phenotypes or genes.
    L_precomp : array-like, optional
        Precomputed Cholesky factors for the main model.
    diag_precomp : array-like, optional
        Precomputed inverse diagonals for the main model.
    L_precomp_alt : array-like, optional
        Precomputed Cholesky factors for the alternative model.
    diag_precomp_alt : array-like, optional
        Precomputed inverse diagonals for the alternative model.
    rng : np.random.Generator, optional
        Random number generator used for phenotype permutations.

    Returns
    -------
    results : np.ndarray
        Per-locus results for the observed phenotype, including omnibus tests,
        parameter-level LOD scores, coefficients and post hoc parameter p-values.
    lrt_pval_threshold : np.ndarray
        Empirical omnibus minP threshold for the main model.
    param_pval_threshold : np.ndarray
        Empirical per-parameter minP threshold(s) for the main model.
    lrt_pval_threshold_alt : np.ndarray or None
        Empirical omnibus minP threshold for the alternative model.
    param_pval_threshold_alt : np.ndarray or None
        Empirical per-parameter minP threshold(s) for the alternative model.
    """
    y, covariates = _as_tensors(y, covariates)

    (n, n_cov, n_hap,
     n_hvars, n_hvars_alt,
     df_resid, df_resid_alt) = _get_params(
        covariates, haplo_variables, haplo_variables_alt
     )

    if bonf_correction is None:
        bonf_correction = 1.0

    beta_nest, _, resid_nest = _fit_lm(covariates, y)

    # shape (n, n_perm + 1)
    y_perms = _generate_permuted_phenotypes(
        covariates, beta_nest, resid_nest, rng=rng, n_perm=n_perm
    )

    del y, beta_nest, resid_nest

    beta_nest, sse_nest, resid_nest, ctc, cty_perms, yty_perms = _fit_lm(
        covariates, y_perms, return_projections=True
    )
    ll_nest = _ll(sse_nest, n).unsqueeze(0)

    del beta_nest, sse_nest, resid_nest

    results = []
    lrt_null_dist = []  # to accumulate minP p val per batch per permutation
    param_null_dist = []
    lrt_null_dist_alt = [] # same but for alternative model
    param_null_dist_alt = []

    for start in range(0, n_hap, batch_size):
        end = min(start + batch_size, n_hap)

        hap_idx = slice(start, end)
        (
            lod_scores_perm, lrt_pvals_perm,
             param_lods_perm, coefs_perm, param_pvals_perm,
             lod_scores_alt_perm, lrt_pvals_alt_perm,
             param_lods_alt_perm, coefs_alt_perm, param_pvals_alt_perm
        ) = _process_batch(
            hap_idx, haplo_variables, haplo_variables_alt,
            y_perms=y_perms, covariates=covariates, ll_nest=ll_nest,
            ctc=ctc, cty_perms=cty_perms, yty_perms=yty_perms,
            L_precomp=L_precomp, diag_precomp=diag_precomp,
            L_precomp_alt=L_precomp_alt, diag_precomp_alt=diag_precomp_alt,
            df_resid=df_resid, df_resid_alt=df_resid_alt
        )
        
        # Separate real data (column 0) from permutations (columns 1+)
        lod_scores = lod_scores_perm[:, 0] # (batch_size,)
        lrt_pvals = lrt_pvals_perm[:, 0] # (batch_size,)
        param_lods = param_lods_perm[:, :, 0]  # (batch_size, n_hvars)
        coefs = coefs_perm[:, :, 0]  # (batch_size, n_hvars)
        param_pvals = param_pvals_perm[:, :, 0]  # (batch_size, n_hvars)

        # direct bonferonni correction of params for number of variables tested
        # other bonferonni corrections are applied to thresholds
        # this should mean that pvals can be used at 5% threshold for posthoc or 
        # at empirically determined threshold for primary tests
        param_pvals = torch.clamp(param_pvals * n_hvars, max=1.0)

        batch_res = torch.cat([
            lod_scores[:, None],
            lrt_pvals[:, None],
            param_lods,
            coefs,
            param_pvals,
        ], dim=1)

        # Compute minP statistic for this batch per permutation

        lrt_null_dist.append(_minP(lrt_pvals_perm[:, 1:]))
        param_null_dist.append(_minP(param_pvals_perm[:, :, 1:]))

        if haplo_variables_alt is not None:

            # Separate real data (column 0) from permutations (columns 1+)
            lod_scores_alt = lod_scores_alt_perm[:, 0] # (batch_size,)
            lrt_pvals_alt = lrt_pvals_alt_perm[:, 0] # (batch_size,)
            param_lods_alt = param_lods_alt_perm[:, :, 0]  # (batch_size, n_hvars_alt)
            coefs_alt = coefs_alt_perm[:, :, 0]  # (batch_size, n_hvars_alt)
            param_pvals_alt = param_pvals_alt_perm[:, :, 0]  # (batch_size, n_hvars_alt)
            param_pvals_alt = torch.clamp(param_pvals_alt * n_hvars_alt, max=1.0)

            batch_res = torch.cat([
                batch_res,
                lod_scores_alt[:, None],
                lrt_pvals_alt[:, None],
                param_lods_alt,
                coefs_alt,
                param_pvals_alt
            ], dim=1)

            lrt_null_dist_alt.append(_minP(lrt_pvals_alt_perm[:, 1:]))
            param_null_dist_alt.append(_minP(param_pvals_alt_perm[:, :, 1:]))

        results.append(batch_res)

        del lod_scores_perm, lrt_pvals_perm
        del param_lods_perm, coefs_perm, param_pvals_perm
        del lod_scores, lrt_pvals, param_lods, coefs, param_pvals

        if haplo_variables_alt is not None:
            del lod_scores_alt_perm, lrt_pvals_alt_perm
            del param_lods_alt_perm, coefs_alt_perm, param_pvals_alt_perm
            del lod_scores_alt, lrt_pvals_alt, param_lods_alt, coefs_alt, param_pvals_alt

    # After all batches, compute overall minP distribution
    lrt_null_dist = _minP(torch.stack(lrt_null_dist))  # (n_perm,)
    lrt_pval_threshold = _calculate_minP_threshold(lrt_null_dist, minP_threshold, bonf_correction).cpu().numpy()
    param_null_dist = _minP(torch.stack(param_null_dist)) # (n_perm, n_hvars)
    param_pval_threshold = _calculate_minP_threshold(param_null_dist, minP_threshold, bonf_correction).cpu().numpy()

    if haplo_variables_alt is not None:
        lrt_null_dist_alt = _minP(torch.stack(lrt_null_dist_alt))  # (n_perm,)
        lrt_pval_threshold_alt = _calculate_minP_threshold(
            lrt_null_dist_alt, minP_threshold, bonf_correction
        ).cpu().numpy()
        param_null_dist_alt = _minP(torch.stack(param_null_dist_alt)) # (n_perm, n_hvars)
        param_pval_threshold_alt = _calculate_minP_threshold(
            param_null_dist_alt, minP_threshold, bonf_correction
        ).cpu().numpy()
    else:
        lrt_pval_threshold_alt = None
        param_pval_threshold_alt = None

    return (
        torch.cat(results, dim=0).cpu().numpy(),
        lrt_pval_threshold,
        param_pval_threshold,
        lrt_pval_threshold_alt,
        param_pval_threshold_alt
    )