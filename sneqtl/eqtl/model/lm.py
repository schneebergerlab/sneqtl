import statsmodels.api as sm


def likelihood_ratio(lln, llf):
    return -2 * (lln - llf)


def compare_lr_test(m_full, Ln, dfn):

    Lf = m_full.llf
    dff = m_full.df_model

    chi2_stat = likelihood_ratio(Ln, Lf)
    p = stats.chi2.sf(chi2_stat, dff - dfn)

    return chi2_stat / LOD, p


def parameter_stats(m, n_tests):
    lods = []
    pvals = []

    scale = max(m_full.scale, 1e-8)  # Avoid division by zero
    cov_diag = np.diag(m_full.normalized_cov_params)
    cov_diag = np.where(np.isfinite(cov_diag), cov_diag, 1.0)  # fallback
    invcov = 1 / (cov_diag[:n_tests] * scale)

    coefs = m_full.params[:n_tests]
    chi2 = coefs ** 2 * invcov
    lods = chi2 / LOD
    pvals = m_full.pvalues[:n_tests]
    return lods, pvals, coefs


def fit_model(gene_exprs, variables):
    X = variables.astype(np.float64, copy=True)
    y = gene_exprs.astype(np.float64, copy=True)
    m = sm.OLS(y, X, hasconst=True)
    m = m.fit(disp=0)._results
    return m

