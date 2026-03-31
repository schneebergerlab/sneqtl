from setuptools import setup

setup(
    name='sneqtl',
    version='1.0',
    description=(
        'eQTL mapping for single cell/nucleus sequencing data'
    ),
    author='Matthew Parker',
    entry_points={
        'console_scripts': [
            'sneqtl = sneqtl.main:sneqtl'
        ]
    },
    packages=[
        'sneqtl',
    ],
    scripts=['scripts/single_cell_gene_counts.py'],
    install_requires=[
        'coelsch',
        'numpy',
        'scipy',
        'pandas',
        'torch',
        'pomegranate>=1.0',
        'click',
        'joblib',
        'pysam',
        'matplotlib',
        'scikit-learn',
        'seaborn', # only for sneqtl plotting, may remove in future
    ],
)
