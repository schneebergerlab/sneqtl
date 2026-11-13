
import fcntl
import hashlib
import json
import os
import shutil
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from sneqtl.variables.base import VariableManager
from sneqtl.variables.exprs import ExpressionVariables
from sneqtl.variables.covariates import Covariates, CisHaploCovariates
from sneqtl.variables.interactions import HaplotypeInteractionVariables


INPUT_FILES = {
    'exprs_mat_dir',
    'haplo_json_fn',
    'diploid_json_fn',
    'cb_stats_fn',
    'gtf_fn',
    'vcf_fn',
}

# Parameters which define variables should be added here.
# Runtime params that do not affect variable creation should be left out, as they
# would needlessly invalidate previously generated variables.
# TODO: solve so that precompute_hap_factors can be removed
MODEL_PARAMS = {
    'rel_min_cells_exprs', 'abs_min_cells_exprs',            # ExpressionVariables filtering
    'control_principal_components', 'min_pc_var_explained',  # Covariates (PCs)
    'covar_names',                                           # Covariates (extra covariates)
    'genotype_covariate_mode', 'genotype_covariates_as_pcs', # Covariates (parental genotypes)
    'reference_genotype', 'min_cells_per_geno',              # Genotype and barcode selection
    'variant_maf',                                           # VariantVariables filtering
    'celltype_haplotype_interaction', 'celltype_n_clusters', # CellTypes and interaction variables
    'control_cis_haplotypes', 'cis_hap_window_size',         # CisHaploCovariates
    'covariate_haplotypes',                                  # Covariates and haplotype filtering
    'cb_filter_exprs',                                       # Barcode filtering (all variables)
    'precompute_hap_factors',                                # Precomputed Cholesky factors
    'rng_state',                                             # Stochastic PCA and clustering
}


class AnalysisCache:
    VERSION = 1

    def __init__(self, cache_root, *, parameters):

        missing = (MODEL_PARAMS | INPUT_FILES) - parameters.keys()
        if missing:
            raise ValueError(f'Missing cache parameters: {sorted(missing)}')

        self.model_params = {
            name: parameters[name]
            for name in sorted(MODEL_PARAMS)
        }

        input_files = {
            name: parameters[name]
            for name in sorted(INPUT_FILES)
            if parameters[name] is not None
        }

        # GTF only affects model construction when controlling for cis haplotypes.
        if not self.model_params['control_cis_haplotypes']:
            input_files.pop('gtf_fn', None)

        self.datasets = self._expected_datasets()

        self.enabled = cache_root is not None
        if not self.enabled:
            self.root = None
            self.path = None
            self.manifest_path = None
            self.key = None
            return

        self.root = Path(cache_root).resolve()
        signature = {
            'version': self.VERSION,
            'inputs': {
                name: self._file_signature(path)
                for name, path in input_files.items()
            },
            'model_params': self.model_params,
        }
        # roundtrip to json to encode encode/decode does not alter comparison
        self.signature = json.loads(json.dumps(signature, sort_keys=True))

        payload = json.dumps(
            self.signature, sort_keys=True, separators=(',', ':')
        ).encode()

        self.key = hashlib.sha256(payload).hexdigest()[:20]
        self.path = self.root / f'sneqtl_{self.key}'
        self.manifest_path = self.path / 'manifest.json'

    def _expected_datasets(self):
        params = self.model_params

        datasets = {
            "expression_variables": ExpressionVariables,
            "covariates": Covariates,
            "haplotype_variables": HaplotypeInteractionVariables,
        }

        if params["control_cis_haplotypes"]:
            datasets["cis_haplo_covariates"] = CisHaploCovariates

        if (
            params["precompute_hap_factors"]
            and not params["control_cis_haplotypes"]
        ):
            datasets["L_precomp"] = VariableManager
            datasets["diag_precomp"] = VariableManager

        return datasets

    @staticmethod
    def _file_signature(path):
        path = Path(path).resolve()

        def metadata(p):
            stat = p.stat()
            return [stat.st_size, stat.st_mtime_ns]

        if path.is_dir():
            files = {
                str(p.relative_to(path)): metadata(p)
                for p in sorted(path.rglob("*"))
                if p.is_file()
            }
        else:
            files = {path.name: metadata(path)}

        return {"path": str(path), "files": files}

    @property
    def dataset_classes(self):
        return {
            name: f"{cls.__module__}.{cls.__qualname__}"
            for name, cls in self.datasets.items()
        }

    def _dataset_file_sizes(self):
        """Validate required files and return their sizes."""
        files = {}

        for name, cls in self.datasets.items():
            data_path = self.path / f"{name}.npy"
            data = np.load(data_path, mmap_mode="r")

            suffixes = [".npy", "_index.pickle", "_columns.pickle"]
            if data.ndim == 3:
                suffixes.append("_items.pickle")
            del data

            if cls.memmap_attrs:
                suffixes.append("_attrs.pickle")

            for suffix in suffixes:
                path = self.path / f"{name}{suffix}"
                files[path.name] = path.stat().st_size

        return files

    def is_complete(self):
        """Return True if the cache is complete and valid."""
        if not self.enabled:
            return False
        try:
            manifest = json.loads(self.manifest_path.read_text())

            return (
                manifest.get("signature") == self.signature
                and manifest.get("datasets") == self.dataset_classes
                and manifest.get("files") == self._dataset_file_sizes()
            )
        except (OSError, ValueError, TypeError, EOFError):
            return False

    @contextmanager
    def lock(self):
        """Exclusive lock for cache construction."""
        if not self.enabled:
            yield
            return
        self.root.mkdir(parents=True, exist_ok=True)
        lock_path = self.root / f"sneqtl_{self.key}.lock"

        with lock_path.open("a+b") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

    def reset(self):
        """Remove an incomplete cache. Must be called under lock."""
        if not self.enabled:
            return
        if self.path.exists():
            shutil.rmtree(self.path)
        self.path.mkdir(parents=True)

    def mark_complete(self):
        """Atomically commit a completed cache. Call under lock."""
        if not self.enabled:
            return
        manifest = {
            "signature": self.signature,
            "datasets": self.dataset_classes,
            "files": self._dataset_file_sizes(),
        }

        temporary = self.path / "manifest.json.tmp"
        temporary.write_text(json.dumps(manifest, indent=2))
        os.replace(temporary, self.manifest_path)
