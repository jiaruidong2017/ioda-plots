# ioda-plots
Tools to perform spatial and temporal binning of observations in JEDI IODA files, manipulate those
binned statistics, and then plot those diagnostics.

The goal of this package is to be able to very quickly produce useful diagnostics with minimal
or no configuration required, but at the same time allow for optional customization of plotting
through yaml configuration files.

## Installation
Assuming you have python3 already installed, `ioda-plots` can be installed by running
```
pip install --user .
```

Alternatively, if you don't feel like doing that, you can simply point to the source directory by
setting the following (and noting that you'll have to run `iodaplots.py` instead of `iodaplots`):
```
PATH=<ioda-plots-path>/src/iodaplots/bin:$PATH
PYTHONPATH=<ioda-plots-path>/src:$PYTHONPATH
```

## Use

(NOTE: Running `iodaplots --help` will provide documentation of the commands available, and running
`iodaplots <command> --help` will provide further documentation about those commands.)

Using ioda-plots consists of three steps:

1. **Spatial binning** - use `iodaplots bin` to take input IODA files and spatially bin them.
  Unless specific configuration is given in a yaml file, this will by default be done on a regular
  latlon grid, and globally, for observation value, O-B, O-A, and A-B statistics. This step should
  usually be done once separately for each cycle of a DA experiment. (this is the slow part, the
  rest of the steps are a lot faster.)

2. **Temporal merging/concatenation (optional)** - use `iodaplots merge` to combine output from the
  previous step, as if you had run `iodaplots bin` on the entire length of the experiment.
  Or, use `iodaplots cat` to combine the output from the previous step, but instead by adding a new `datetime` dimension to the binning. `cat` will then allow for generation of timeseries plots.

3. **Plotting** - use `iodaplots plot` on the output of the previous step to generate a whole bunch
  of plots. By default every metric that could be plotted is done, which is probably way more
  plots than you want. A yaml configuration can select which plots are generate, and allows for
  fine tuning of those plots.


(More detailed documentation and examples will be added later!)