# hpi-neg

## Requirements
* Python 3
* torch >= 1.8.1
* numpy
* sklearn
* biopython

### Parameters
-m model (pipr, denovo, deepviral, deeptrio)\
-n name (experiment name, required for log files)\
-t file (training file eg a csv, required)\
-l length (maximum sequence length, deepviral: -l 1000, deeptrio: -l 1500)\
-e epochs (epochs, deepviral: -e 30, deeptrio: -e 50)\
-f fasta (for denovo datasets, use denovo.fasta, for hpidb-denovo, use hpidb.fasta)\
-B (required only for hpidb-ratio*.csv to use balanced pos/neg batches)

NOTE: DeepTrio requires 15+ gigs of gpu memory.\
