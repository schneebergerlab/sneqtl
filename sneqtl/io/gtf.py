import re
import pandas as pd

def get_gtf_attribute(gtf_record, attribute):
    try:
        attr = re.search(f'{attribute} "(.+?)";', gtf_record[8]).group(1)
    except AttributeError:
        raise ValueError(
            f'Could not parse attribute {attribute} '
            f'from GTF with feature type {gtf_record[2]}'
        )
    return attr


def flatten_intervals(invs):
    flattened = []
    inv_it = iter(invs)
    inv_start, inv_end = next(inv_it)
    for start, end in inv_it:
        if start <= inv_end:
            inv_end = max(inv_end, end)
        else:
            flattened.append([inv_start, inv_end])
            inv_start, inv_end = start, end
    if not flattened or flattened[-1] != [inv_start, inv_end]:
        flattened.append([inv_start, inv_end])
    return flattened


def parse_gtf(gtf_fn, model_type='gene'):
    gtf_records = {}
    with open(gtf_fn) as gtf:
        for i, record in enumerate(gtf):
            if record.startswith('#'):
                continue
            record = record.split('\t')
            chrom, _, feat_type, start, end, _, strand = record[:7]
            start = int(start) - 1
            end = int(end)
            if feat_type == 'exon':
                gene_id = get_gtf_attribute(record, 'gene_id')
                idx = (chrom, strand, gene_id)
                if idx not in gtf_records:
                    gtf_records[idx] = []
                gtf_records[idx].append([start, end])
    gene_records = []
    for (chrom, strand, gene_id), invs in gtf_records.items():
        invs.sort()
        if model_type == 'gene':
            invs = flatten_intervals(invs)
        elif model_type == 'gene_full':
            invs = [[invs[0][0], invs[-1][1]]]
        else:
            raise ValueError('unrecognised quant_type')
        gene_records.append((gene_id, chrom, strand, invs))
    return gene_records


def read_gtf_gene_locs(gtf_fn):
    gene_locs = {}
    for gene_id, chrom, strand, invs in  parse_gtf(gtf_fn, model_type='gene_full'):
        gene_locs[gene_id] = (chrom, (invs[0][0] + invs[-1][1]) // 2)
    return pd.DataFrame.from_dict(gene_locs, orient='index', columns=['chrom', 'pos'])

