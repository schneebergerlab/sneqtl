"""Haplotype variables from scalar or multichannel coelsch predictions."""

import numpy as np

from coelsch.records import PredictionRecords
from .base import VariableManager
from .arr_utils import corrwith


class HaplotypeVariables(VariableManager):
    """Per-barcode, per-bin haplotype predictions.

    Scalar coelsch records are stored as ``(barcode, bin)`` arrays.
    Multichannel records are stored as ``(barcode, bin, channel)`` arrays,
    with zero-based channel indices in ``items``.

    Channels are *local to each coelsch genotype*: in a pooled four-way
    population, channel 0 is not necessarily the same named founder for
    every barcode. Use ``PredictionRecords.metadata['genotypes']`` to map
    channels to founders before combining data across crosses.

    Values are not clipped: multichannel data may contain diploid dosages
    greater than one, and clipping would silently corrupt them.
    """

    index_name = 'cb'
    columns_name = 'haplotypes'

    def __init__(self, data, index, columns, items=None, name=None,
                 dtype=None, frozen=True, bin_size=None):
        if data.ndim == 3 and items is None:
            items = list(range(data.shape[2]))
        super().__init__(data, index=index, columns=columns, items=items,
                         name=name, dtype=dtype, frozen=frozen)
        self.bin_size = bin_size
        self.chroms = set(chrom for chrom, _ in self.columns)

    @property
    def barcodes(self):
        return self.index

    @property
    def haplotypes(self):
        return self.columns

    @property
    def channels(self):
        """Channel labels, or None for scalar records."""
        return self.items

    def reindex(self, index=None, columns=None, items=None):
        result = super().reindex(index=index, columns=columns, items=items)
        result.bin_size = self.bin_size
        return result

    def copy(self):
        result = super().copy()
        result.bin_size = self.bin_size
        return result

    def select_channel(self, channel):
        """Select a 2D view of one multichannel dosage/probability track.

        ``channel`` is the numeric *coelsch channel index*, not a global
        founder name. The result retains its genomic bin coordinates.
        """
        if self.ndim != 3:
            raise ValueError('select_channel requires multichannel haplotypes')
        if channel not in self._item_map:
            raise KeyError(f'Unknown haplotype channel: {channel!r}')
        return self.__class__(
            self._data[:, :, self._item_map[channel]],
            index=self.index, columns=self.columns,
            name=self.name, bin_size=self.bin_size,
            frozen=self.frozen,
        )

    def iter_chroms(self):
        for chrom in dict.fromkeys(chrom for chrom, _ in self.columns):
            yield chrom, self.reindex(
                columns=[h for h in self.columns if h[0] == chrom]
            )

    @staticmethod
    def _get_donor_indices(records, diploid_records):
        from coelsch.experiment.genotypes import GenotypeKey

        pollen_genotypes = records.metadata.get('genotypes')
        diploid_genotypes = diploid_records.metadata.get('genotypes')

        if pollen_genotypes is None or diploid_genotypes is None:
            raise ValueError(
                "Both PredictionRecords objects require genotype metadata"
            )

        donor_map = {
            cb: i for i, cb in enumerate(diploid_records.barcodes)
        }

        donor_indices = []

        for cb in records.barcodes:
            try:
                genotype = GenotypeKey.from_any(pollen_genotypes[cb])
            except KeyError:
                raise ValueError(
                    f"Missing genotype metadata for pollen barcode {cb}"
                ) from None

            donor = genotype.name

            if donor not in donor_map:
                raise ValueError(
                    f"No matching diploid parent for pollen barcode "
                    f"{cb}: {donor}"
                )

            try:
                diploid_genotype = GenotypeKey.from_any(
                    diploid_genotypes[donor]
                )
            except KeyError:
                raise ValueError(
                    f"Missing diploid genotype metadata for {donor}"
                ) from None

            if genotype.genotype != diploid_genotype.genotype:
                raise ValueError(
                    f"Genotype mismatch for donor {donor}: "
                    f"{genotype} != {diploid_genotype}"
                )

            donor_indices.append(donor_map[donor])

        return donor_indices


    @staticmethod
    def _map_haploid_predictions_to_founder_channels(pollen, diploid_records,
                                                     donor_indices, chrom):
        strategy = diploid_records.experiment_params.crossing_strategy

        haps = {
            parent: values[donor_indices]
            for parent, values in diploid_records.iter_scalar_haplotypes(chrom)
        }

        p0, p1 = pollen[..., 0], pollen[..., 1]

        if strategy == 'testcross':
            q = haps['parent2']
            return np.stack([
                p0,
                p1 * (1 - q),
                p1 * q,
            ], axis=-1)

        elif strategy == 'three_way':
            q0, q1 = haps['parent1'], haps['parent2']
            return np.stack([
                p0 * (1 - q0) + p1 * (1 - q1),
                p0 * q0,
                p1 * q1,
            ], axis=-1)

        elif strategy == 'four_way':
            q0, q1 = haps['parent1'], haps['parent2']
            return np.stack([
                p0 * (1 - q0),
                p0 * q0,
                p1 * (1 - q1),
                p1 * q1,
            ], axis=-1)

        raise ValueError(f"Unsupported crossing strategy: {strategy}")

    @classmethod
    def from_prediction_records(cls, records, diploid_records=None, cb_whitelist=None,
                                barcode_query=None, dtype=np.float32, name=None):
        """
        Construct HaplotypeVariables from coelsch PredictionRecords.

        If diploid_records is provided, combine haploid recombinant predictions
        with the corresponding diploid founder predictions to reconstruct
        haploid founder dosages. Supports testcross, three-way and four-way
        crossing strategies.

        Channel ordering remains native to each barcode's genotype.

        Parameters
        ----------
        records : PredictionRecords
            Input haplotype predictions.
        diploid_records : PredictionRecords, optional
            Diploid founder predictions for the parents of recombinant pollen.
        cb_whitelist : iterable, optional
            Barcodes to retain.
        barcode_query : str or callable, optional
            Query for filtering barcodes.
        dtype : dtype
            Output data type.
        name : str, optional
            Name of the resulting variable object.

        Returns
        -------
        HaplotypeVariables
            Predictions have shape (n_barcodes, n_bins, n_haplotypes).
        """
        if not isinstance(records, PredictionRecords):
            raise TypeError("records must be a PredictionRecords object")

        if cb_whitelist is not None:
            records = records.filter(cb_whitelist, inplace=False)

        if barcode_query is not None:
            records = records.query(barcode_query)

        barcodes = list(records.barcodes)
        bin_size = records.bin_size

        haplotypes = [
            (chrom, i * bin_size)
            for chrom in records.chrom_sizes
            for i in range(records.nbins[chrom])
        ]

        donor_indices = None

        if diploid_records is not None:
            if not isinstance(diploid_records, PredictionRecords):
                raise TypeError(
                    "diploid_records must be a PredictionRecords object"
                )

            pollen_params = records.experiment_params
            diploid_params = diploid_records.experiment_params
            strategy = diploid_params.crossing_strategy

            if (
                pollen_params.ploidy != 1
                or pollen_params.genotyping_strategy != 'recombinant'
            ):
                raise ValueError(
                    "Pollen predictions must be haploid "
                    "and genotyped in recombinant mode"
                )

            if (
                diploid_params.ploidy != 2
                or diploid_params.genotyping_strategy != 'founder'
                or strategy not in ('testcross', 'three_way', 'four_way')
            ):
                raise ValueError(
                    "Diploid predictions must be diploid, have more than two founder haplotypes "
                    "and have been genotyped in founder mode"
                )

            if (
                records.bin_size != diploid_records.bin_size
                or records.chrom_sizes != diploid_records.chrom_sizes
            ):
                raise ValueError(
                    "Pollen and diploid predictions must have identical "
                    "bin sizes and chromosome coordinates"
                )

            donor_indices = cls._get_donor_indices(
                records, diploid_records
            )

        # Allocate output.
        n_channels = (
            diploid_records.experiment_params.n_haplotypes
            if diploid_records is not None
            else records.experiment_params.n_haplotypes
        )

        shape = (len(barcodes), len(haplotypes), n_channels)
        items = list(range(n_channels))

        result = cls(
            np.empty(shape, dtype=dtype),
            index=barcodes,
            columns=haplotypes,
            items=items,
            name=name,
        )
        result.bin_size = bin_size

        # Populate predictions chromosome by chromosome.
        offset = 0

        for chrom, n_bins in records.nbins.items():

            # Preserve the native coelsch representation.
            chrom_data = records.get_haplotype_dosage(chrom=chrom).stack_values()

            expected_shape = (
                len(barcodes),
                n_bins,
                records.experiment_params.n_haplotypes,
            )

            if chrom_data.shape != expected_shape:
                raise ValueError(
                    f"Unexpected prediction shape for {chrom}: "
                    f"{chrom_data.shape}, expected {expected_shape}"
                )

            if diploid_records is not None:
                chrom_data = cls._map_haploid_predictions_to_founder_channels(
                    chrom_data, diploid_records, donor_indices, chrom
                )

            result._data[:, offset:offset + n_bins, ...] = chrom_data
            offset += n_bins

        return result

    @classmethod
    def read_prediction_records(cls, json_fn, diploid_json_fn=None, cb_whitelist=None,
                                barcode_query=None, dtype=np.float32):
        records = PredictionRecords.read_json(json_fn, subset=cb_whitelist)
        if diploid_json_fn is not None:
            diploid_records = PredictionRecords.read_json(diploid_json_fn)
        else:
            diploid_records = None
        return cls.from_prediction_records(
            records, diploid_records=diploid_records,
            barcode_query=barcode_query, dtype=dtype
        )

    def _get_closest_haplotype(self, chrom, pos):
        positions = [p for c, p in self.columns if c == chrom]
        idx = np.searchsorted(positions, pos, side='right') - 1
        if idx < 0:
            raise KeyError(f'No haplotype bin preceding {chrom}:{pos}')
        return chrom, positions[idx]

    def filter_correlated(self, other, max_r2):
        if self.ndim == 2:
            return super().filter_correlated(other, max_r2)

        # Filter *whole bins* if any of their channels is correlated with
        # any control variable. Preserves the 3D representation.
        n_barcodes, n_bins, n_channels = self.shape
        control = other._data.reshape(n_barcodes, -1)
        candidate = self._data.reshape(n_barcodes, -1)
        r2 = corrwith(control, candidate, axis=0) ** 2
        max_r2_per_channel = np.nan_to_num(r2, nan=0.0).max(axis=0)
        max_r2_per_bin = max_r2_per_channel.reshape(n_bins, n_channels).max(axis=1)
        selected = [
            bin_id for bin_id, corr in zip(self.columns, max_r2_per_bin)
            if corr <= max_r2
        ]
        return self.reindex(columns=selected)

    def get_covariate_haplotypes(self, parental_genotypes, covariate_haplotypes=None,
                                 remove_correlated=True, max_r2=0.95):
        if covariate_haplotypes is None:
            return None, self

        cntrl_hap_columns = [
            self._get_closest_haplotype(*h)
            for h in covariate_haplotypes
        ]

        # Convert control positions to globally aligned founder probabilities.
        cntrl_vars = [
            parental_genotypes.get_haplotype_variables(self[h])
            for h in cntrl_hap_columns
        ]

        control_haplotypes = VariableManager(
            np.concatenate([v._data for v in cntrl_vars], axis=1),
            index=self.index,
            columns=[
                (h, founder)
                for h, v in zip(cntrl_hap_columns, cntrl_vars)
                for founder in v.columns
            ],
        )

        if not remove_correlated or not control_haplotypes.columns:
            return control_haplotypes, self

        # Remove bins with any founder predictor correlated with a control.
        keep = []

        for h in self.columns:
            hap_vars = parental_genotypes.get_haplotype_variables(self[h])
            uncorrelated = hap_vars.filter_correlated(
                control_haplotypes, max_r2
            )

            if len(uncorrelated.columns) == len(hap_vars.columns):
                keep.append(h)

        return control_haplotypes, self.reindex(columns=keep)

    def get_mean_haplotype(self, chrom, start, end):
        """Return mean haplotype dosage across bins in a genomic interval.

        Returns (n_barcodes, n_channels) for multichannel predictions,
        or (n_barcodes,) for scalar predictions.
        """
        indices = [
            i for i, (c, pos) in enumerate(self.columns)
            if c == chrom and start <= pos < end
        ]
        if not indices:
            raise ValueError(f'No haplotype bins in {chrom}:{start}-{end}')

        return self._data[:, indices, ...].mean(axis=1)