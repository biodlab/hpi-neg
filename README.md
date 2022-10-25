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
-l length (maximum sequence length, deepviral: -l 1000, deeptrio: 1500)\
-e epochs (epochs, deepviral: -e 30, deeptrio: -e 50)
