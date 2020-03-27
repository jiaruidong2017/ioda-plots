#!/usr/bin/env python3
import binnedStats
import plot
import xarray
import os

# yaml config
config="binning.yaml"

# input files
prefix="../data/hofx3d_gfs_c12_gsi_"
postfix="_PT6H_20190630_2100Z"
ext=".nc4"
platforms= [
    "aircraft",
#    "airs-aqua",
    "amsua-n19",
    "gnssro",
    "hirs4-metopa",
#    "iasi-metopa",
    "mhs-n19",
    "radiosonde",
    "satwind",
    "seviri-m08",
    "sst"

]

# output plots
plotdir="../plots"

for platform in platforms:
    # input hofx filename
    filename=prefix + platform + postfix + ext
    print(filename)

    # get list of all variables in that file
    d = xarray.open_dataset(filename)
    variables = binnedStats.get_valid_obsvars(d)
    d.close()

    args={
        'thumbnail':False,
        'diff': False,
    }

    # for each var, bin, save binning, and plot
    for v in sorted(variables):
        args['prefix']=plotdir+'/'+platform + "/"
        args['title'] = platform+" "+v

        # create output directories
        d=os.path.dirname(args['prefix'])
        if not os.path.exists(d):
            os.makedirs(d)

        # bin data
        bins = binnedStats.BinnedStatsCollection.create(
            filename, config, v)

        # # save bins
        # # (skip for now)
        # bin_filename = bin_dir + platform + "_" + v + postfix + ".p"
        # bins.save(bin_filename)

        # save plot
        for k,v in bins.binned_stats.items():
            if len(v.bin_dims) == 2: #only do 2d plots for now
                plot.plot_2d(v, **args, daterange=bins.daterange)


