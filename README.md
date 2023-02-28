# hpi-neg

Repository for [On the choice of negative examples for prediction of host-pathogen protein interactions](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9798088/)

## Requirements
* Python 3
* torch >= 1.8.1
* numpy
* sklearn
* biopython

### Parameters for model.py
-m model (pipr, denovo, deepviral, deeptrio)\
-n name (experiment name, required for log files)\
-t file (training file eg a csv, required)\
-l length (maximum sequence length, deepviral: -l 1000, deeptrio: -l 1500)\
-e epochs (epochs, deepviral: -e 30, deeptrio: -e 50)\
-f fasta file (for denovo datasets, use denovo.fasta, for hpidb-denovo, use hpidb.fasta)\

NOTE: DeepTrio requires 15+ gigs of gpu memory.
