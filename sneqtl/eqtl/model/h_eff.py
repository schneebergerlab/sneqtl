import numpy as np
from sneqtl.variables import arr_utils


def _li_ji_m_eff(h):
    R = arr_utils.corrwith(h._data, h._data, axis=0)
    R = np.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(R, 1.0)
    R = 0.5 * (R + R.T)
    eig = np.linalg.eigvalsh(R)
    eig = np.clip(eig, 0.0, None)
    m_eff = np.minimum(eig, 1.0).sum()
    return float(m_eff)


def _nvars_per_bin(haplotypes, variants, percentile=75):
    counts = {h: 0 for h in haplotypes}
    for chrom, bin_start, _ in variants.snp_ids:
        counts[(chrom, bin_start)] += 1
    counts = np.array(list(counts.values()), dtype=np.float64)
    return float(np.percentile(counts, percentile))


def calculate_effective_haplotypes(haplotype_variables, variant_variables=None, var_perc=75):
    h_eff = 0
    for chrom, h in haplotype_variables.iter_chroms():
        h_eff += _li_ji_m_eff(h)

    if variant_variables is not None:
        # correct by a measure of the number of variants per bin if using variants as well
        h_eff *= _nvars_per_bin(haplotype_variables.haplotypes, variant_variables, var_perc)

    return h_eff