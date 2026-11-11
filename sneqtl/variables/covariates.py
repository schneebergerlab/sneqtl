import numpy as np
from .base import VariableManager
from .haplotypes import HaplotypeVariables
from .genotypes import ParentalGenotypesContainer
from .pca import PrincipalComponents
from .var_utils import check_aligned
from sneqtl.io.gtf import read_gtf_gene_locs


class Covariates(VariableManager):

    @property
    def variable_names(self):
        return self.columns
    
    @property
    def barcodes(self):
        return self.index

    @classmethod
    def from_variables(cls, *, interacting_variables=None, parental_genotypes=None,
                       covariate_haplotypes=None, base_covariates=None,
                       genotype_covariate_mode='founder',
                       genotype_covariates_as_pcs=False, variants=None,
                       name=None, barcodes=None):
        columns = ['const',]
        sections = []

        if interacting_variables is None:
            interacting_variables = []

        sources = [
            *interacting_variables, parental_genotypes,
            covariate_haplotypes, base_covariates
        ]
        index = check_aligned(*sources) if any(s is not None for s in sources) else barcodes

        if index is None:
            raise ValueError('If there are no other covariates, barcodes must be provided')

        for iv in interacting_variables:
            iv = iv.as_covariates()
            sections.append(iv._data)
            columns += iv.columns

        if genotype_covariate_mode != 'none':
            pg = parental_genotypes.as_covariates(
                variants=variants,
                as_pcs=genotype_covariates_as_pcs,
                encoding=genotype_covariate_mode,
            )
            sections.append(pg._data)
            columns += pg.columns

        if covariate_haplotypes is not None:
            ch = covariate_haplotypes.as_covariates()
            sections.append(ch._data)
            columns += ch.columns

        if base_covariates is not None:
            bc = base_covariates.as_covariates()
            sections.append(bc._data)
            columns += bc.columns

        if not sections:
            return cls.null_covariates(barcodes, name=name)

        covariate_matrix = np.column_stack(sections)
        n_obs = covariate_matrix.shape[0]
        intercept = np.ones((n_obs, 1), dtype=covariate_matrix.dtype)
        covariate_matrix = np.column_stack([intercept, covariate_matrix])

        covars = cls(covariate_matrix, index=index, columns=columns, name=name)
        if covars.is_singular:
            raise ValueError(
                f'Covariate matrix is singular'
            )
        return covars

    @classmethod
    def null_covariates(cls, barcodes, dtype=np.float32, name=None):
        intercept = np.ones((len(barcodes), 1), dtype=dtype)
        return cls(intercept, index=barcodes, columns=['const'], name=name)


class CisHaploCovariates(VariableManager):

    index_name = 'cb'
    columns_name = 'gene_id'
    items_name='variable_names'

    @property
    def barcodes(self):
        return self.index

    @property
    def gene_ids(self):
        return self.columns

    @property
    def variable_names(self):
        return self.items

    def __init__(self, data, index, columns, items=None, name=None, dtype=None, frozen=True):
        super().__init__(
            data=data,
            index=index,
            columns=columns,
            items=items,
            name=name,
            dtype=dtype,
            frozen=frozen,
        )
        self.bin_size = None
        self.window_size = None

    @classmethod
    def from_haplotypes(cls, haplotype_variables, gene_locs, parental_genotypes,
                        variants=None, window_size=250_000, max_pcs=10,
                        gene_ids=None, dtype=np.float32, name=None):
        if not isinstance(haplotype_variables, HaplotypeVariables):
            raise TypeError('haplotype_variables must be a HaplotypeVariables object')

        if not isinstance(parental_genotypes, ParentalGenotypesContainer):
            raise TypeError('parental_genotypes must be a ParentalGenotypesContainer object')

        if haplotype_variables.bin_size is None:
            raise ValueError('haplotype_variables.bin_size must be set')

        check_aligned(haplotype_variables, parental_genotypes)

        if gene_ids is not None:
            gene_locs = gene_locs.loc[list(gene_ids)]

        if variants is None:
            items = [f'cis_hap_{h}' for h in parental_genotypes.columns if h != parental_genotypes._ref_geno]
        else:
            n_pcs = min(max_pcs, len(parental_genotypes.columns) - 1)
            items = [f'cis_struct{i}' for i in range(1, n_pcs + 1)]

        # todo: implement window size in centiMorgans rather than bases
        bin_size = haplotype_variables.bin_size
        ws = window_size // bin_size
        h_ws = ws // 2

        columns = []
        data = []
        pos_arrs = {}
        chrom_level_pcs = {}
        
        for gene_id, chrom, pos in gene_locs.itertuples():

            pos_idx = pos // bin_size
            columns.append(gene_id)
            if (chrom, pos_idx) in pos_arrs:
                # already calculated for another gene in same bin
                data.append(pos_arrs[(chrom, pos_idx)])
                continue

            win_start = max(pos_idx - h_ws, 0) * bin_size
            # can be longer than chrom size but doesn't matter due to get_mean_haplotype implementation
            win_end = (pos_idx + h_ws + 1) * bin_size
            win_cis_hap = parental_genotypes.get_haplotype_variables(
                haplotype_variables.get_mean_haplotype(chrom, win_start, win_end),
                variants=None
            )._data

            if variants is not None:
                local_variants = variants.get_bin_variants(chrom, win_start, win_end)

                if len(local_variants) > n_pcs:
                    pcs = PrincipalComponents.from_local_variants(local_variants, n_pcs)
                else:
                    if chrom not in chrom_level_pcs:
                        chrom_variants = variants.get_bin_variants(chrom)
                        if len(chrom_variants) <= n_pcs:
                            raise ValueError(
                                'Insufficient variation to calculate cis-haplotype principal components'
                            )
                        chrom_level_pcs[chrom] = PrincipalComponents.from_local_variants(
                            chrom_variants, n_pcs
                        )
                    pcs = chrom_level_pcs[chrom]

                # Express founder PC scores relative to the reference founder.
                ref = parental_genotypes._ref_geno
                nonref = [f for f in parental_genotypes.columns if f != ref]
                pc_scores = pcs.reindex(index=nonref)._data - pcs.loc[ref]

                win_cis_hap = win_cis_hap @ pc_scores
                        

            pos_arrs[(chrom, pos_idx)] = win_cis_hap
            data.append(win_cis_hap)

        data = np.stack(data, axis=1)

        out = cls(
            data=data,
            index=haplotype_variables.index,
            columns=columns,
            items=items,
            name=name,
            dtype=dtype,
        )
        out.bin_size = haplotype_variables.bin_size
        out.window_size = window_size
        return out

    def get_gene(self, gene_id):
        return VariableManager(
            self.loc[:, gene_id, :],
            index=self.index,
            columns=self.items,
            name=f'{self.name}:{gene_id}' if self.name is not None else str(gene_id),
        )