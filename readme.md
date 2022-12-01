# scripts for "Systematic Error in Flood Hazard Aggregation"

This repository is a copy of my [private repository](https://github.com/cefect/2112_Agg) which is no longer used. This is still quite messy and contains some old scripts. 

For the QGIS script, see the [FloodRescaler repo](https://github.com/cefect/FloodRescaler)

## Installation
see conda\conda_env.yml

## Submodules
git submodule add -b 2112_Agg https://github.com/cefect/coms.git

## Use
raw input data is specified in the definitions.py
working data files are constructed with haz/run.py and expo/run.py
    resulting data filepaths are entered into the corresponding 'res_fp_lib'
    
    
plots are generated with a few standalone scripts 'run_da...'

    maps_SJ_r11_direct_diffXR_1201.svg: agg2/run_da_maps


## Tests
using pytests

some old tests are in /tests
new tests are in /tests2, but many of these seem to be broken