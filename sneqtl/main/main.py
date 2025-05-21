import click

from .sneqtl_opts import sneqtl_opts

@click.group()
@click.version_option()
def sneqtl():
    '''
    sneqtl: a toolkit for performing single nucleus eQTL mapping with snco results
    '''
    pass


@sneqtl.command('eqtl')
@sneqtl_opts('eqtl')
def eqtl_subcommand(**kwargs):
    from .eqtl import run_eqtl
    run_eqtl(**kwargs)


@sneqtl.command('peakcall')
@sneqtl_opts('peakcall')
def peakcall_subcommand(**kwargs):
    from .peaks import run_peakcall
    run_peakcall(**kwargs)