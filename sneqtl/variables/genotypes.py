import itertools as it
from functools import cached_property
import numpy as np

from coelsch.experiment.genotypes import GenotypeKey
from .base import VariableManager
from .pca import PrincipalComponents
from .arr_utils import get_dummies

DEFAULT_RANDOM_SEED = 101
DEFAULT_RNG = np.random.default_rng(DEFAULT_RANDOM_SEED)


import numpy as np

from coelsch.experiment.genotypes import GenotypeKey

from .base import VariableManager
from .pca import PrincipalComponents


class ParentalGenotypesContainer:
    """
    Represent parental genotypes and their correspondence to native
    coelsch haplotype channels.

    Attributes
    ----------
    _hap1, _hap2 : VariableManager
        Binary founder-membership matrices for the two parents.
        Shape: (n_barcodes, n_founders).

    _unique_genotypes : VariableManager
        One-hot encoding of distinct overall genotype structures,
        ignoring individual genotype names.
        Shape: (n_barcodes, n_genotypes).

    _channel_indices : ndarray
        Maps native haplotype channels to global founder indices.
        Shape: (n_barcodes, n_channels).

    _ref_geno : str
        Reference founder omitted from haplotype regression variables.
    """

    def __init__(self, hap1, hap2, unique_genotypes, channel_indices,
                 reference_genotype=None):
        if (
            hap1.index != hap2.index
            or hap1.columns != hap2.columns
            or hap1.shape != hap2.shape
        ):
            raise ValueError(
                "Parental haplotype matrices must have identical axes"
            )

        if unique_genotypes.index != hap1.index:
            raise ValueError("Genotype and haplotype indexes differ")

        channel_indices = np.asarray(channel_indices, dtype=np.intp)

        if (
            channel_indices.ndim != 2
            or channel_indices.shape[0] != len(hap1.index)
        ):
            raise ValueError("Invalid channel index dimensions")

        if (
            np.any(channel_indices < 0)
            or np.any(channel_indices >= len(hap1.columns))
        ):
            raise ValueError("Channel indices are out of bounds")

        # Native channels must identify distinct founders.
        if any(len(set(row)) != len(row) for row in channel_indices):
            raise ValueError("Duplicate founder assignments within a barcode")

        self._hap1 = hap1
        self._hap2 = hap2
        self._unique_genotypes = unique_genotypes
        self._channel_indices = channel_indices

        if reference_genotype is None:
            reference_genotype = hap1.columns[0]

        if reference_genotype not in hap1.columns:
            raise ValueError(
                f"Unknown reference genotype: {reference_genotype}"
            )

        self._ref_geno = reference_genotype

    @property
    def index(self):
        return self._hap1.index

    @property
    def columns(self):
        return self._hap1.columns

    @property
    def reference_genotype(self):
        return self._ref_geno

    @classmethod
    def from_genotype_dict(cls, parental_genotypes, reference_genotype=None):
        """
        Construct genotype encodings from barcode -> GenotypeKey mappings.

        Supports simple F1 crosses and more complex crossing trees.
        Native founder-channel ordering is preserved.
        """
        barcodes = list(parental_genotypes)

        if not barcodes:
            raise ValueError("No parental genotypes provided")

        genotypes = [
            GenotypeKey.from_any(parental_genotypes[cb])
            for cb in barcodes
        ]

        if any(gt.is_leaf_genotype for gt in genotypes):
            raise ValueError("Expected genotypes with two parents")

        # Shared founder ordering across all barcodes.
        founders = sorted({
            founder
            for gt in genotypes
            for founder in gt.founders
        })

        founder_index = {
            founder: i for i, founder in enumerate(founders)
        }

        n_barcodes = len(barcodes)
        n_founders = len(founders)

        hap1 = np.zeros((n_barcodes, n_founders), dtype=np.float32)
        hap2 = np.zeros((n_barcodes, n_founders), dtype=np.float32)

        for i, gt in enumerate(genotypes):
            for founder in gt.parent1.founders:
                hap1[i, founder_index[founder]] = 1

            for founder in gt.parent2.founders:
                hap2[i, founder_index[founder]] = 1

        hap1 = VariableManager(
            hap1,
            index=barcodes,
            columns=founders,
        )

        hap2 = VariableManager(
            hap2,
            index=barcodes,
            columns=founders,
        )

        genotype_labels = [
            (gt.genotype, gt.parental_roles, gt.sex_roles)
            for gt in genotypes
        ]

        unique_labels = sorted(set(genotype_labels), key=str)

        genotype_index = {
            label: i for i, label in enumerate(unique_labels)
        }

        unique_data = np.zeros(
            (n_barcodes, len(unique_labels)),
            dtype=np.float32,
        )

        for i, label in enumerate(genotype_labels):
            unique_data[i, genotype_index[label]] = 1

        unique_genotypes = VariableManager(
            unique_data,
            index=barcodes,
            columns=unique_labels,
        )

        n_channels = {len(gt.founders) for gt in genotypes}

        channel_indices = np.array([
            [founder_index[f] for f in gt.founders]
            for gt in genotypes
        ], dtype=np.intp)

        return cls(
            hap1=hap1,
            hap2=hap2,
            unique_genotypes=unique_genotypes,
            channel_indices=channel_indices,
            reference_genotype=reference_genotype,
        )

    def as_covariates(self, variants=None, as_pcs=False,
                      min_var_explained=0.01, max_pcs=10,
                      rng=None, encoding='founders'):
        """
        Return position-independent genotype covariates.

        encoding='founders':
            Founder membership summed across the two parents.

        encoding='genotypes':
            One-hot encoding of distinct overall crossing genotypes.
        """
        if encoding not in ('founders', 'genotypes'):
            raise ValueError(
                "encoding must be 'founders' or 'genotypes'"
            )

        if as_pcs:
            if encoding != 'founders':
                raise ValueError(
                    "PCA is only supported with founder encoding"
                )

            kwargs = dict(
                min_var_explained=min_var_explained,
                max_pcs=max_pcs,
            )
            if rng is not None:
                kwargs['rng'] = rng

            return PrincipalComponents.from_variants(
                variants,
                self,
                **kwargs,
            )

        if encoding == 'founders':
            # Existing representation: sum parental founder memberships.
            data = self._hap1._data + self._hap2._data

            columns = [
                f for f in self.columns
                if f != self._ref_geno
            ]
            indices = [self.columns.index(f) for f in columns]

            data = data[:, indices]

        else:
            # One dummy per crossing genotype, dropping the first.
            columns = self._unique_genotypes.columns[1:]
            data = self._unique_genotypes._data[:, 1:]

        covariates = VariableManager(
            data,
            index=self.index,
            columns=columns,
        )

        # Remove constant and linearly dependent covariates.
        if covariates.shape[1]:
            covariates = covariates.drop_uninformative(axis=1)

        if covariates.shape[1] > 1:
            covariates = covariates.drop_rank_dependent()

        return covariates

    def get_haplotype_variables(self, haplotype, variants=None):
        hap_var_data = np.zeros(
            self._hap1.shape,
            dtype=haplotype.dtype,
        )

        hap_var_data[
            np.arange(len(self.index))[:, None],
            self._channel_indices,
        ] = haplotype

        if variants is None:
            hap_vars = VariableManager(
                hap_var_data,
                index=self.index,
                columns=[f'haplo_{g}' for g in self.columns],
            )
            hap_vars = hap_vars.drop(columns=[f'haplo_{self._ref_geno}'])

        else:
            hap_var_data = (variants._data @ hap_var_data.T)[..., None]
            hap_vars = VariableManager(
                hap_var_data,
                index=variants.index,
                columns=self.index,
                items=['alt_snp'],
            )

        return hap_vars

    def reindex(self, index=None, columns=None):
        """
        Reindex barcodes and optionally reorder founder columns.

        Reordering founders updates the native channel mapping.
        Dropping founders is not supported.
        """
        if index is None:
            index = self.index
        if columns is None:
            columns = self.columns

        if set(columns) != set(self.columns):
            raise ValueError(
                "Reindexing must preserve the complete founder set"
            )

        # Map old global founder indices to new ones.
        old_to_new = np.array([
            columns.index(f) for f in self.columns
        ], dtype=np.intp)

        row_indices = [
            self._hap1._index_map[cb] for cb in index
        ]

        channel_indices = old_to_new[
            self._channel_indices[row_indices]
        ]

        return self.__class__(
            hap1=self._hap1.reindex(
                index=index, columns=columns,
            ),
            hap2=self._hap2.reindex(
                index=index, columns=columns,
            ),
            unique_genotypes=self._unique_genotypes.reindex(
                index=index,
            ),
            channel_indices=channel_indices,
            reference_genotype=self._ref_geno,
        )