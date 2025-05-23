from .qtl import SinglePhenoQTLAnalysis
from ..variables.exprs import ExpressionVariables
from ..variables.haplotypes import HaplotypeVariables
from ..variables.covariates import Covariates
from ..variables.cishaplotype import CisHaplotypeCovariates


class eQTLAnalysisManager:

    def __init__(self, exprs_mat, haplotypes, parental_genotypes, )