import click

from .sneqtl_opts import sneqtl_opts

@click.group()
@click.version_option()
def sneqtl():
    '''
    sneqtl: a toolkit for performing single nucleus eQTL mapping with snco results
    '''
    pass


@sneqtl.command('loadexprs')
@sneqtl_opts('loadexprs')
def loadexprs_subcommand(**kwargs):
    from ..exprs import run_loadexprs
    run_loadexprs(**kwargs)


@sneqtl.command('concatexprs')
@sneqtl_opts('concatexprs')
def concatexprs_subcommand(**kwargs):
    from ..exprs import run_concatexprs
    run_concatexprs(**kwargs)


@sneqtl.command('doubletexprs')
@sneqtl_opts('doubletexprs')
def doubletexprs_subcommand(**kwargs):
    from ..exprs import run_doubletexprs
    run_doubletexprs(**kwargs)


@sneqtl.command('eqtl')
@sneqtl_opts('eqtl')
def eqtl_subcommand(**kwargs):
    from ..eqtl import run_eqtl
    run_eqtl(**kwargs)


@sneqtl.command('peakcall')
@sneqtl_opts('peakcall')
def peakcall_subcommand(**kwargs):
    from ..peaks import run_peakcall
    run_peakcall(**kwargs)