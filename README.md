# OscInA: Oscillations in autocorrelated time-series

Tools to test for periodic rhythms in aperiodic autocorrelated signals.


## Citation

> [Brookshire, G. (2022) Putative rhythms in attentional switching can be explained by aperiodic temporal structure. *Nature Human Behaviour.*](https://www.nature.com/articles/s41562-022-01364-0)


## Installation

- `mtspec` should be installed using conda: `$ conda install -c conda-forge mtspec`
- This code relies on specific versions of libraries that are not stable yet, so I strongly recommend installing everything in a dedicated conda environment.

On macOS or Linux, here's how you'd do that:
```bash
conda create -n oscina
conda activate oscina
conda install -c conda-forge mtspec
pip3 install git+https://github.com/gbrookshire/oscina
```


### Requirements

- Python 3
- External Python dependencies are listed in `requirements.txt`.


## Examples

See the notebook `examples/Example.ipynb` for illustrations of how to use `oscina`.
