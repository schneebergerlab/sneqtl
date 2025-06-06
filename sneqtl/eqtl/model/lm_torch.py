import numpy as np
from scipy import stats

try:
    import torch
    TORCH_AVAILABLE = True
except Exception:  # ImportError or others
    TORCH_AVAILABLE = False
    torch = None

LOD = 2 * np.log(10)


def _linear_model(X, y):
    # X: (n, p), y: (n,)
    Xt = X.T
    XtX = Xt @ X
    Xty = Xt @ y
    L = torch.linalg.cholesky(XtX)
    beta = torch.cholesky_solve(Xty.unsqueeze(1), L).squeeze(1)
    resid = y - X @ beta
    sse = torch.sum(resid ** 2)
    return beta, sse, resid


def permuted_residuals(y, X, *, rng, n_permutations=50):
    dtype = torch.float64
    device = 'cpu'

    y_t = torch.tensor(y, dtype=dtype, device=device)
    X_t = torch.tensor(X, dtype=dtype, device=device)

    beta, sse, resid = _linear_model(X_t, y_t)
    fitted = X_t @ beta
    resid = (y_t - fitted).cpu().numpy()

    for _ in range(n_permutations):
        permuted = rng.permutation(resid)
        y_perm = fitted.cpu().numpy() + permuted
        yield y_perm


def fit_models_torch(y, covariates, haplo_variables, batch_size=500, mode='full'):

    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is not available")

    dtype = torch.float64
    device = "cpu"

    # Use ``torch.tensor`` to avoid warnings from non-writeable memmaps
    y_t = torch.tensor(y, dtype=dtype, device=device)
    cov_t = torch.tensor(covariates, dtype=dtype, device=device)

    n, n_hap, n_hvars = haplo_variables.shape
    n_cov = cov_t.shape[1]

    # Nested model with covariates only
    _, sse_nest, _ = _linear_model(cov_t, y_t)
    ll_nest = -0.5 * n * (np.log(2 * np.pi) + torch.log(sse_nest / n) + 1)

    # Pre-compute quantities shared across all haplotypes
    ct_c = cov_t.T @ cov_t  # (n_cov, n_cov)
    ct_y = cov_t.T @ y_t  # (n_cov,)
    yty = (y_t * y_t).sum()  # scalar

    df_resid = n - (n_cov + n_hvars)

    all_results = []

    for start in range(0, n_hap, batch_size):
        end = min(start + batch_size, n_hap)
        batch_size_actual = end - start

        h = torch.tensor(haplo_variables[:, start:end, :], dtype=dtype, device=device)
        hb_t = h.permute(1, 2, 0)  # (b, n_hvars, n)

        ht_h = torch.bmm(hb_t, h.permute(1, 0, 2))  # (b, n_hvars, n_hvars)
        ht_c = hb_t @ cov_t                         # (b, n_hvars, n_cov)
        ht_y = hb_t @ y_t                           # (b, n_hvars)

        # Assemble XtX blocks
        top = torch.cat([ht_h, ht_c], dim=2)  # (b, n_hvars, p)
        bottom = torch.cat(
            [ht_c.transpose(1, 2), ct_c.repeat(batch_size_actual, 1, 1)], dim=2
        )
        xtx = torch.cat([top, bottom], dim=1)  # (b, p, p)

        # Assemble XtY
        xty = torch.cat([ht_y, ct_y.repeat(batch_size_actual, 1)], dim=1)  # (b, p)

        # Solve via Cholesky
        L = torch.linalg.cholesky(xtx)  # (b, p, p)
        beta = torch.cholesky_solve(xty.unsqueeze(2), L).squeeze(2)  # (b, p)

        sse = yty - (beta * xty).sum(dim=1)  # (b,)
        ll_full = -0.5 * n * (np.log(2 * np.pi) + torch.log(sse / n) + 1)
        chi2_lr = -2 * (ll_nest - ll_full)
        lod_scores = chi2_lr / LOD

        # Standard errors from Cholesky inverse
        xtx_inv = torch.cholesky_inverse(L)  # (b, p, p)
        diag = torch.diagonal(xtx_inv, dim1=1, dim2=2)  # (b, p)
        se = torch.sqrt(diag * (sse / df_resid).unsqueeze(1))  # (b, p)

        t_stats = beta / se  # (b, p)
        param_chi2 = t_stats[:, :n_hvars] ** 2
        param_lods = param_chi2 / LOD

        if mode == 'lod_only':
            batch_res = torch.cat([
                lod_scores[:, None],
                param_lods
            ], dim=1)
        else:
            lr_pvals = stats.chi2.sf(chi2_lr.detach().cpu().numpy(), n_hvars)

            param_pvals = 2 * stats.t.sf(
                torch.abs(t_stats[:, :n_hvars]).detach().cpu().numpy(), df_resid
            )
            param_coefs = beta[:, :n_hvars]

            # Efficient construction of result tensor
            batch_res = torch.cat([
                lod_scores[:, None],
                torch.from_numpy(lr_pvals).to(dtype=dtype, device=device)[:, None],
                param_lods,
                torch.from_numpy(param_pvals).to(dtype=dtype, device=device),
                param_coefs
            ], dim=1)

        all_results.append(batch_res)

    return torch.cat(all_results, dim=0).cpu().numpy()