from collections import defaultdict
from itertools import count

import numpy as np
import pysam

from .base import VariableManager
from .genotypes import ParentalGenotypesContainer


class VariantVariables(VariableManager):

    def __init__(self, data, index, columns, items=None, name=None, dtype=None, frozen=True):
        super().__init__(data, index, columns, items=None, name=name, dtype=dtype, frozen=frozen)
        self.bin_size = None
        self.chroms = set(chrom for chrom, bin_idx, pos in self.index)

    @property
    def snp_ids(self):
        return self.index

    @property
    def haplotypes(self):
        return self.columns

    def get_bin_variants(self, chrom, bidx):
        idx = [
            (c, b, pos)
            for c, b, pos in self.index
            if c == chrom and b == bidx
        ]
        return self.__class__(self.loc[idx], index=idx, columns=self.columns)

    def filter_minor_allele(self, parental_genotypes, haplotype_variables, min_frac=0.05, min_abs=100):
        n_barcodes = parental_genotypes._hap1.shape[0]
        reindex_ids = []
        for (chrom, pos), i in haplotype_variables._column_map.items():
            hap = haplotype_variables[(chrom, pos)]
            w = (
                hap @ parental_genotypes._hap1._data + 
                (1 - hap) @ parental_genotypes._hap2._data
            )
            bin_variants = self.get_bin_variants(chrom, pos)
            minor_counts = bin_variants._data @ w
            minor_counts = np.minimum(minor_counts, n_barcodes - minor_counts)
            maf = minor_counts / n_barcodes
            min_abs_filt = minor_counts >= min_abs
            min_frac_filt = maf >= min_frac
            reindex_ids += [
                pos_id for pos_id, f1, f2 in zip(bin_variants.index, min_abs_filt, min_frac_filt)
                if f1 and f2
            ]
        return self.reindex(index=reindex_ids)

    @classmethod
    def read_vcf(cls, vcf_fn, bin_size, reference_name, use_haplotypes=None):
        if bin_size <= 0:
            raise ValueError("bin_size must be > 0")

        representative_patterns = []
        pos_ids = []
        unique_id = count(1)

        def add_bin_representatives(bin_items, chrom, bin_idx):
            """
            Collapse identical or inverse patterns in bin_items.
            Keep the first encountered representative for each canonical key.
            """
            seen = defaultdict(list)
            n_added = 0
            bin_start = bin_idx * bin_size
            for pos, pattern in bin_items:
                if pattern not in seen:
                    representative_patterns.append(pattern)
                    n_added += 1
                seen[pattern].append(pos)
            for pos_group in seen.values():
                i = next(unique_id)
                pos_group = f'grp{i}:' + ';'.join(str(pos - bin_start) for pos in pos_group)
                pos_ids.append((chrom, bin_start, pos_group))

        vcf = pysam.VariantFile(vcf_fn)
        all_haps = [reference_name,] + list(vcf.header.samples)
        if use_haplotypes is None:
            haplotypes = all_haps
        else:
            missing = [h for h in use_haplotypes if h not in all_haps]
            if missing:
                raise ValueError(f"Samples not found in VCF: {missing}")
            haplotypes = use_haplotypes
        haplotypes.sort()

        n_haps = len(haplotypes)
        current_chrom = None
        current_bin = None
        bin_items = []

        for var in vcf.fetch():
            # only biallelic single-base SNPs
            if len(var.alleles) != 2:
                continue
            var_samples = var.samples
            sample_alleles = []
            n = 0
            for h in haplotypes:
                if h == reference_name:
                    hap_allele = (0, ) # reference allele
                else:
                    hap_allele = var_samples[h].allele_indices
                if len(hap_allele) != 1 or hap_allele[0] is None:
                    break
                sample_alleles.append(hap_allele[0])
                n += 1
            if n != n_haps:
                continue

            chrom = var.chrom
            pos = var.pos
            bin_idx = (pos - 1) // bin_size

            if current_chrom is None:
                current_chrom = chrom
            if current_bin is None:
                current_bin = bin_idx

            if chrom != current_chrom or bin_idx != current_bin:
                if bin_items:
                    add_bin_representatives(bin_items, current_chrom, current_bin)
                bin_items = []
                current_chrom = chrom
                current_bin = bin_idx

            bin_items.append((pos, tuple(sample_alleles)))

        if bin_items:
            add_bin_representatives(bin_items, current_chrom, current_bin)

        # build numpy array of shape (n_variants, n_samples) dtype float32, missing -> np.nan
        if len(representative_patterns) == 0:
            data = np.zeros((0, len(haplotypes)), dtype=np.float32)
        else:
            data = np.array(representative_patterns, dtype=np.float32)

        variant_variables = cls(index=pos_ids, columns=haplotypes, data=data)
        variant_variables.bin_size = bin_size
        return variant_variables
