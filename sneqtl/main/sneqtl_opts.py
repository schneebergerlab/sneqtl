from logging import getLogger
import re
import click

from coelsch.main.opts.file_opts import _input_dir_type, _input_file_type, _output_dir_type, _output_file_type
from coelsch.main.opts.util_opts import _get_rng
from coelsch.main.registry import OptionRegistry
from coelsch.main.logger import click_logger
from coelsch.main.opts.callbacks import log_parameters
from coelsch.defaults import DEFAULT_RANDOM_SEED


log = getLogger('sneqtl')


sneqtl_opts = OptionRegistry(subcommands=['loadexprs', 'concatexprs', 'exprsstats', 'eqtl', 'peakcall'])
sneqtl_opts.register_callback(log_parameters('sneqtl'))


sneqtl_opts.argument(
    'exprs-mat-dir',
    subcommands=['exprsstats', 'eqtl'],
    required=True,
    nargs=1,
    type=_input_dir_type,
)


sneqtl_opts.argument(
    'exprs-mat-dir',
    subcommands=['concatexprs'],
    required=True,
    nargs=-1,
    type=_input_dir_type
)


sneqtl_opts.argument(
    'haplo-json-fn',
    subcommands=['eqtl'],
    required=True,
    nargs=1,
    type=_input_file_type,
)


sneqtl_opts.argument(
    'eqtl-dir',
    subcommands=['peakcall'],
    required=True,
    nargs=1,
    type=_input_dir_type,
)


sneqtl_opts.option(
    '-o', '--output-mat-dir',
    subcommands=['loadexprs', 'concatexprs'],
    required=True,
    nargs=1,
    type=_output_dir_type,
    help='Output directory for matrix market format expression data'
)


sneqtl_opts.option(
    '-o', '--output-prefix',
    subcommands=['eqtl'],
    required=True,
    type=_output_file_type,
    help='Output prefix'
)


sneqtl_opts.option(
    '-o', '--output-tsv-fn',
    subcommands=['exprsstats', 'peakcall'],
    required=True,
    type=_output_file_type,
    help='Output prefix'
)


sneqtl_opts.option(
    '-b', '--bam-fn',
    subcommands=['loadexprs'],
    required=True,
    type=_input_file_type,
    help='BAM file name'
)


sneqtl_opts.option(
    '-c', '--cb-stats-fn',
    subcommands=['loadexprs', 'exprsstats', 'eqtl'],
    required=False,
    type=_input_file_type,
    help='cell barcode stats (output of coelsch predict/stats)'
)


sneqtl_opts.option(
    '-g', '--gtf-fn',
    subcommands=['eqtl'],
    required=False,
    type=_input_file_type,
    help='GTF file name'
)


sneqtl_opts.option(
    '-g', '--gtf-fn',
    subcommands=['loadexprs', 'peakcall'],
    required=True,
    type=_input_file_type,
    help='GTF file name'
)


sneqtl_opts.option(
    '-V', '--vcf-fn',
    subcommands=['eqtl'],
    required=False,
    type=_input_file_type,
    help='VCF file name (activates GWAS mode!!)'
)

sneqtl_opts.option(
    '--diploid-json-fn',
    subcommands=['eqtl'],
    required=False,
    type=_input_file_type,
    help='A coelsch prediction records json containing parental recombinant genotypes' # TODO clearer docstring
)

def _validate_bam_tag(ctx, param, value):
    if not len(value) == 2:
        raise ValueError(f'{param} is not of length 2')
    return value


sneqtl_opts.option(
    '--cb-tag',
    subcommands=['loadexprs'],
    required=False,
    type=str,
    default='CB',
    callback=_validate_bam_tag,
    help='bam file tag representing cell barcode'
)


sneqtl_opts.option(
    '--umi-collapse-method',
    subcommands=['loadexprs'],
    is_eager=True,
    required=False,
    type=click.Choice(['exact', 'directional', 'none'], case_sensitive=False),
    default='exact',
    help='Method for deduplicating/collapsing UMIs'
)


sneqtl_opts.option(
    '--umi-tag',
    subcommands=['loadexprs'],
    required=False,
    type=str,
    default='UB',
    help='bam file tag representing UMI'
)


sneqtl_opts.option(
    '--min-mapq',
    subcommands=['loadexprs'],
    required=False,
    type=click.IntRange(0, 255),
    default=0,
    help='only reads with a mapq greater than this value are used'
)



sneqtl_opts.option(
    '--strand-type',
    subcommands=['loadexprs'],
    required=False,
    type=click.Choice(['fwd', 'rev'], case_sensitive=False),
    default='fwd',
    help='Which strand to take reads from in relation to transcripts'
)


sneqtl_opts.option(
    '--quant-type',
    subcommands=['loadexprs'],
    required=False,
    type=click.Choice(['gene', 'gene_full'], case_sensitive=False),
    default='gene',
    help='how to quantify gene expression - "gene" == exons, "gene_full" == exons+introns'
)


def _parse_merge_suffixes(ctx, param, value):
    if value is not None:
        value = value.strip(',')
        if not re.match('^[a-zA-Z0-9_,]+$', value):
            log.error('merge suffixes must be alphanumeric only')
        value = value.split(',')
    return value


sneqtl_opts.option(
    '-M', '--merge-suffixes',
    subcommands=['concatexprs'],
    required=False,
    type=str,
    default=None,
    callback=_parse_merge_suffixes,
    help=('comma separated list of suffixes to append to cell barcodes from '
          'each file. Must be alphanumeric with same length as no. mtx dirs')
)


sneqtl_opts.option(
    '--rel-min-cells-exprs',
    subcommands=['exprsstats', 'eqtl'],
    required=False,
    type=click.FloatRange(0.0, 0.5),
    default=0.02,
    help='the minimum proportion of cell barcodes that must express a gene for it to be tested'
)



sneqtl_opts.option(
    '--abs-min-cells-exprs',
    subcommands=['exprsstats', 'eqtl'],
    required=False,
    type=click.IntRange(0, 5000),
    default=100,
    help='the minimum number of cell barcodes that must express a gene for it to be tested'
)



sneqtl_opts.option(
    '--control-pcs/--no-pcs', 'control_principal_components',
    subcommands=['eqtl'],
    required=False,
    default=True,
    help='whether to use principal components of the expression matrix as covariates'
)


sneqtl_opts.option(
    '--min-pc-var-explained',
    subcommands=['exprsstats', 'eqtl'],
    required=False,
    type=click.FloatRange(0, 1.0),
    default=0.01,
    help='the minimum variance explained required to keep a principal component for modelling'
)


def _parse_whitelist_covar_names(ctx, param, value):
    if value is not None:
        value = value.split(',')
    return value


sneqtl_opts.option(
    '--covar-names',
    subcommands=['eqtl'],
    required=False,
    type=str,
    default=None,
    callback=_parse_whitelist_covar_names,
    help=('comma separated list of columns from --cb-stats-fn that should be used as '
          'covariates in the model')
)


sneqtl_opts.option(
    '--control-genotype/--no-genotype', 'control_parental_genotype',
    subcommands=['eqtl'],
    required=False,
    default=False,
    help='whether to control for parental (diploid) genotype'
)


sneqtl_opts.option(
    '--ref-genotype', 'reference_genotype',
    subcommands=['eqtl'],
    required=False,
    default=None,
    help='which founder genotype to use as a reference (to be dropped in the model)'
)


sneqtl_opts.option(
    '--min-cells-per-geno',
    subcommands=['eqtl'],
    required=False,
    type=click.IntRange(1, None),
    default=1,
    help='the minimum number of barcodes each genotype needs to have to be modelled'
)


sneqtl_opts.option(
    '-f', '--variant-min-allele-fraction', 'variant_maf',
    subcommands=['eqtl'],
    required=False,
    type=click.FloatRange(1e-2, 1.0),
    default=0.05,
    help=('the minimum allele fraction (accounting for number of barcodes per genotype) required '
          'to model a variant')
)


sneqtl_opts.option(
    '--geno-as-pcs/--no-geno-as-pcs', 'genotype_covariates_as_pcs',
    subcommands=['eqtl'],
    required=False,
    default=False,
    help='when generating covariates, whether to use parental genotypes directly or to create PCs. Requires --vcf-fn'
)


sneqtl_opts.option(
    '--control-cis-hap/--no-cis-hap', 'control_cis_haplotypes',
    subcommands=['eqtl'],
    required=False,
    default=False,
    help='whether to control for cis-haplotype as a covariate. When a vcf is provided, PCs will be used'
)


sneqtl_opts.option(
    '--cis-hap-window-size',
    subcommands=['eqtl'],
    required=False,
    type=click.IntRange(0, 100_000_000),
    default=250_000,
    help=('the size of window to use to generaet cis haplotype covariates')
)


sneqtl_opts.option(
    '--celltype/--no-celltype', 'celltype_haplotype_interaction',
    subcommands=['eqtl'],
    required=False,
    default=False,
    help='whether to cluster barcodes into celltypes and model them as interactors with haplotype'
)


sneqtl_opts.option(
    '--celltype-n-clusters',
    subcommands=['exprsstats', 'eqtl'],
    required=False,
    type=click.IntRange(2, 10),
    default=None,
    help=('number of clusters to use for celltype clustering. '
          'default is to automatically choose best number using silhouette scoring')
)


def _parse_covar_haplotypes(ctx, param, value):
    if value is not None:
        parsed = []
        for locus in value.strip().split(','):
            chrom, pos = locus.split(':')
            pos = int(pos)
            parsed.append((chrom, pos))
    return parsed


sneqtl_opts.option(
    '--covariate-haplotypes',
    subcommands=['eqtl'],
    required=False,
    default=None,
    help=('comma separated list of positions (in format Chr1:100000), the haplotypes of which '
          'should be controlled for as covariates in the model')
)


sneqtl_opts.option(
    '--minP-threshold', 'minP_threshold',
    subcommands=['eqtl'],
    required=False,
    type=click.FloatRange(0, 1),
    default=0.01,
    help=('The target family-wise error rate used to derive empirical p-value thresholds '
          'from the permutation-based minP distribution')
)


sneqtl_opts.option(
    '--n-perm',
    subcommands=['eqtl'],
    required=False,
    type=click.IntRange(10, 1_000_000),
    default=1000,
    help=('The number of permutations to use to generate null minP distributions')
)


sneqtl_opts.option(
    '--cb-filter-exprs',
    subcommands=['loadexprs', 'exprsstats', 'eqtl'],
    required=False,
    type=str,
    default=None,
    help=('expression used to filter barcodes by column in --cb-stats-fn '
          'e.g. "doublet_probability < 0.5"')
)

sneqtl_opts.option(
    '--peak-variable',
    subcommands=['peakcall'],
    required=False,
    type=str,
    default='overall',
    help='the variable used for lod peak calling'
)

sneqtl_opts.option(
    '--lod-threshold',
    subcommands=['eqtl', 'peakcall'],
    required=False,
    type=click.FloatRange(2, 20),
    default=5,
    help='Absolute LOD score threshold used for peak calling'
)


sneqtl_opts.option(
    '--rel-lod-threshold',
    subcommands=['eqtl', 'peakcall'],
    required=False,
    type=click.FloatRange(0, 1),
    default=0.1,
    help=('Relative LOD score threshold used for secondary peak calling '
          'compared to highest LOD on chromosome')
)


sneqtl_opts.option(
    '--rel-prominence',
    subcommands=['eqtl', 'peakcall'],
    required=False,
    type=click.FloatRange(0, 1),
    default=0.25,
    help=('Relative prominence of LOD score used for secondary peak calling '
          'compared to highest LOD on chromosome')
)


sneqtl_opts.option(
    '--ci-lod-drop',
    subcommands=['eqtl', 'peakcall'],
    required=False,
    type=click.FloatRange(1, 10),
    default=1.5,
    help='Drop in LOD score from peak used to calculate eQTL confidence intervals'
)


sneqtl_opts.option(
    '--min-dist-between-eqtls',
    subcommands=['eqtl', 'peakcall'],
    required=False,
    type=click.IntRange(0, 1e10),
    default=3e6,
    help='Minimum allowed distance in bases between two eQTL peaks'
)


sneqtl_opts.option(
    '--cis-eqtl-range',
    subcommands=['eqtl', 'peakcall'],
    required=False,
    type=click.IntRange(0, 5e6),
    default=5e5,
    help='Maximum distance from edge of eQTL CI boundary to gene for it to be considered a cis-eQTL'
)


sneqtl_opts.option(
    '-p', '--processes',
    subcommands=['loadexprs', 'eqtl',],
    required=False,
    type=click.IntRange(min=1),
    default=1,
    help='number of cpu processes to use'
)

sneqtl_opts.option(
    '-t', '--pytorch-threads', 'torch_threads_per_process',
    subcommands=['eqtl',],
    required=False,
    type=click.IntRange(min=1),
    default=1,
    help='number of pytorch threads to use per process'
)


sneqtl_opts.option(
    '--resume',
    subcommands=['eqtl'],
    is_flag=True,
    help='resume an existing run - only test genes not already existing'
)


sneqtl_opts.option(
    '-r', '--random-seed', 'rng',
    subcommands=['eqtl'],
    required=False,
    type=int,
    default=DEFAULT_RANDOM_SEED,
    callback=_get_rng,
    help='seed for random number generator'
)


sneqtl_opts.option(
    '-T', '--tmpdir', 'tmpdir',
    subcommands=['eqtl'],
    required=False,
    default=None,
    help='temporary directory for memmapped variables'
)



sneqtl_opts.option(
    '-v', '--verbosity',
    subcommands=['loadexprs', 'concatexprs', 'exprsstats', 'eqtl', 'peakcall'],
    required=False,
    expose_value=False,
    metavar='LVL',
    type=click.Choice(
        ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        case_sensitive=False
    ),
    default='info',
    callback=click_logger('sneqtl'),
    help='Logging level, either debug, info, warning, error or critical'
)
