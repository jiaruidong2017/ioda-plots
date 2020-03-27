#!/usr/bin/env python3
from binnedStats import BinnedStats, BinnedStatsCollection
import os, sys
import numpy
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.dates as mdates

import cartopy.crs as ccrs
import matplotlib.ticker as mticker
from cartopy.mpl import gridliner

zdims = ('height','depth')
cmap_div="RdBu_r"
cmap_seq="inferno"

# # ------------------------------------------------------------------------------
# # ------------------------------------------------------------------------------
# def plot_3d_xy(data):
#     #######################################
#     # TODO implement this!
#     #######################################
#     print("Plot 3D (latlon) ", data)
#     # which dimension is the non lat/lon dim
#     # zdim = (set(data.bin_dims) - set(('latitude','longitude'))).pop()


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
def plot_2d(data, daterange, **kwargs):

    title = kwargs['title']
    title = data.obsvar if title is None else title

    # figure out what type of plot
    mesh_opt = {}
    xy_type = ""
    transpose = False

    if set(('latitude','longitude')) == set(data.bin_dims):
        # lat/lon
        xy_type = "latlon"
        proj = ccrs.PlateCarree(central_longitude=-155)
        mesh_opt['transform'] = ccrs.PlateCarree()
        x_dim = data.bin_dims.index('longitude')
        y_dim = data.bin_dims.index('latitude')
        def plot_type_pre(ax):
            # lat/lon axes
            gl = ax.gridlines(zorder=1, alpha=0.5, color='k',
                              linestyle='--', draw_labels=True)
            gl.xformatter = gridliner.LONGITUDE_FORMATTER
            gl.xlocator = mticker.FixedLocator([-90, 0, 90, 180, 270])
            gl.xlabels_top = False
            gl_label_style = {'color': 'gray', 'size': 8}
            gl.xlabel_style = gl_label_style
            gl.yformatter = gridliner.LATITUDE_FORMATTER
            gl.ylocator = mticker.FixedLocator([-90, -60, -30, 0, 30, 60, 90])
            gl.ylabels_right = False
            gl.ylabel_style = gl_label_style
            # coastline, bkg color
            ax.coastlines(color='k', alpha=0.5)
            ax.background_patch.set_facecolor('lightgray')

    elif 'time' in data.bin_dims:
        # Hovmoller
        xy_type = "Hovmoller"
        proj = None
        time_dim = data.bin_dims.index('time')
        other_dim = 1 - time_dim
        if data.bin_dims[other_dim] in ('longitude'):
            # time should be on the y axis
            x_dim = other_dim
            y_dim = time_dim
        else:
            # time should be on the x axis
            transpose = True
            x_dim = time_dim
            y_dim = other_dim

        def plot_type_pre(ax):
            pass
    else:
        # generic coordinates
        proj = None
        x_dim = 0
        y_dim = 1
        def plot_type_pre(ax):
            pass

    print("Plot 2D ",xy_type, data)

    # generate x/y coordinates
    mesh_i, mesh_j = numpy.meshgrid(data.bin_edges[x_dim], data.bin_edges[y_dim])


    def plot_common_pre(title_add="", text=[]):
        plt.figure(figsize=(8.0, 4.0))
        ax = plt.axes(projection=proj)
        ax.set_facecolor('lightgray')
        #plt.axvline(x=0.0, color='k', alpha=0.5)
        ax.set_xlabel(data.bin_dims[x_dim])
        ax.set_ylabel(data.bin_dims[y_dim])
        # invert y axis if it is depth
        if data.bin_dims[y_dim] in ('depth',):
            plt.gca().invert_yaxis()
        #dstr = [d.strftime("%Y-%m-%d") for d in dates]
        #dstr = dstr[0] if dstr[0] == dstr[1] else dstr[0] + ' to ' + dstr[1]
        #plt.annotate(dstr, ha='right', xycoords='axes points', xy=(420, -24.0))
        t=title+" "+title_add
        exp = kwargs['exp_name'] if "exp_name" in kwargs else ""
        if exp != "":
          t = t +" (" + exp +")"
        plt.title(t)
        i = -24.0
        for t in text:
            plt.annotate(t, xycoords='axes points', xy=(0.0, i))
            i -= 12.0
        dstr = [d.strftime("%Y-%m-%d") for d in daterange]
        dstr = dstr[0] if dstr[0] == dstr[1] else dstr[0] + ' to '+dstr[1]
        plt.annotate(dstr, ha='right', xycoords='axes points', xy=(420, -24.0))
        plot_type_pre(ax)
        return ax

    def plot_common_post(type_):
        bbox_inches='tight' if kwargs['thumbnail'] else None
        plt.colorbar(orientation='vertical', shrink=0.7, fraction=0.02)
        plt.savefig("{}{}.{}{}.{}.png".format(
            kwargs['prefix'], title.lower().replace(' ','_'), data.name, "",type_),
            bbox_inches = bbox_inches)
        plt.close()

    # counts
    for p in ('count', 'count_qc'):
        d = data.count(qc=p=='count_qc')
        dMax = numpy.max(d)
        dSum = numpy.sum(d)
        text = ["min: {:0.0f}".format(dSum),
                "max: {:0.0f}".format(dMax),]
        if p == 'count':
            dRange = numpy.percentile(d[d>0], [99])[0]
        plot_common_pre(title_add=p, text=text)
        d = numpy.transpose(d) if transpose else d
        plt.pcolormesh(mesh_i, mesh_j, d, **mesh_opt,
            cmap=cmap_seq, norm=colors.LogNorm(vmin=1.0, vmax=dRange))
        plot_common_post(p)

    # pct bad
    count = data.count(qc=False)
    count_qc = data.count(qc=True)
    d = (count - count_qc)*100
    d[count > 0] /= count[count > 0]
    d = numpy.ma.masked_where(count == 0, d)
    dMin = numpy.min(d[count > 0])
    dMax = numpy.max(d)
    dAvg = numpy.mean(d[count > 0])
    text = ["min: {:0.2f} max: {:0.2f}".format(dMin, dMax),
            "avg: {:0.2f}".format(dAvg)]
    d = numpy.transpose(d) if transpose else d
    plot_common_pre(title_add=" count_pctbad", text=text)
    plt.pcolormesh(mesh_i, mesh_j, d, cmap=cmap_seq, **mesh_opt)
    plot_common_post('count_pctbad')

    # rmsd
    for p in ('ombg', ):
        d = data.rmsd(mode=p)
        dMax = numpy.max(d)
        dAvg = numpy.mean(d)
        text = ['max: {:0.2e}'.format(dMax),
                'avg: {:0.2e}'.format(dAvg),]

        if p == 'ombg':
            if kwargs['diff']:
                dRange = numpy.max(numpy.abs(
                    numpy.percentile(d[count_qc > 0], [1, 99])))
                norm=colors.Normalize(vmin=-dRange, vmax=dRange)
                cmap=cmap_div
            else:
                dRange = numpy.percentile(d[count_qc > 0], [1, 99])
                norm=colors.LogNorm(*dRange)
                cmap=cmap_seq
        d = numpy.transpose(d) if transpose else d
        plot_common_pre(title_add=p+" rmsd", text=text)
        plt.pcolormesh(mesh_i, mesh_j, d, cmap=cmap, norm=norm, **mesh_opt)
        plot_common_post(p+'_rmsd')


    # bias
    for p in ('ombg', ):
        d = data.mean(mode=p)
        dMax = max(numpy.max(d), abs(numpy.min(d)))
        dAvg = numpy.mean(d[count_qc > 0])
        text = ['max: {:0.2e}'.format(dMax),
                'avg: {:0.2e}'.format(dAvg)]

        if p == 'ombg':
            dRange = numpy.max(numpy.abs(
                numpy.percentile(d[count_qc > 0], [1,99])))
        d = numpy.transpose(d) if transpose else d
        plot_common_pre(title_add=p+' bias', text=text)
        plt.pcolormesh(mesh_i, mesh_j, d, cmap=cmap_div, vmin=-dRange, vmax=dRange,
            **mesh_opt)
        plot_common_post(p+'_bias')


# # ------------------------------------------------------------------------------
# # ------------------------------------------------------------------------------
# def plot_2d_z(data, **kwargs):
#     print("Plot 2D (? x depth) ", data)

#     # TODO plot variable size based on dimensions
#     # TODO thumbnail mode
#     # TODO add real date annotation
#     # TODO shrink the colorbar padding
#     dates=[datetime.now(), datetime.now()]

#     idx_z = 0 if data.bin_dims[0] in zdims else 1
#     idx_x = 1 if idx_z is 0 else 0
#     mesh_x, mesh_z = numpy.meshgrid(data.bin_edges[idx_x], data.bin_edges[idx_z])

#     def plot_common_pre(title=""):
#         plt.figure(figsize=(8.0, 4.0))
#         ax = plt.axes()
#         ax.set_facecolor('lightgray')
#         plt.title(data.obsvar+" "+title)
#         plt.axvline(x=0.0, color='k', alpha=0.5)
#         ax.set_xlabel(data.bin_dims[idx_x])
#         ax.set_ylabel(data.bin_dims[idx_z])
#         if data.bin_dims[idx_z] in ('depth',):
#             plt.gca().invert_yaxis()
#         dstr = [d.strftime("%Y-%m-%d") for d in dates]
#         dstr = dstr[0] if dstr[0] == dstr[1] else dstr[0] + ' to ' + dstr[1]
#         plt.annotate(dstr, ha='right', xycoords='axes points', xy=(420, -24.0))
#         return ax

#     def plot_common_post(type_):
#         plt.colorbar(orientation='vertical', shrink=0.7, fraction=0.02)
#         plt.savefig("{}{}_{}_{}.png".format(
#             kwargs['prefix'], data.obsvar, data.name, type_))
#         plt.close()

#     # counts
#     for p in ('count', 'count_qc'):
#         d = data.count(qc=p=='count_qc')
#         # dMax = numpy.max(d)
#         # dSum = numpy.sum(d)
#         if p == 'count':
#             dRange = numpy.percentile(d[d>0], [99])[0]
#         plot_common_pre(title=p)
#         plt.pcolormesh(mesh_x, mesh_z, d,
#                        cmap=cmap_seq, norm=colors.LogNorm(vmin=1.0, vmax=dRange))
#         plot_common_post(p)

#     # pct bad
#     count = data.count(qc=False)
#     count_qc = data.count(qc=True)
#     d = count - count_qc
#     d[count > 0] /= count[count > 0]
#     d = numpy.ma.masked_where(count == 0, d)
#     # dMin = numpy.min(d[count > 0])
#     # dMax = numpy.max(d)
#     # dAvg = numpy.mean(d[count > 0])
#     plot_common_pre(title=" count_pctbad")
#     plt.pcolormesh(mesh_x, mesh_z, d, cmap=cmap_seq)
#     plot_common_post('count_pctbad')

#     # rmsd
#     for p in ('ombg', 'oman'):
#         d = data.rmsd(mode=p)
#         # dMax = numpy.max(d)
#         # dAvg = numpy.mean(d)
#         if p == 'ombg':
#             dRange = numpy.percentile(d[count_qc > 0], [1,99])

#         plot_common_pre(title=p+" rmsd")
#         plt.pcolormesh(mesh_x, mesh_z, d, cmap=cmap_seq, norm=colors.LogNorm(*dRange))
#         plot_common_post(p+'_rmsd')

#     # bias
#     for p in ('ombg', 'oman'):
#         d = data.mean(mode=p)
#         # dMax = max(numpy.max(d), abs(numpy.min(d)))
#         # dAvg = numpy.mean(d[count_qc > 0])
#         if p == 'ombg':
#             dRange = numpy.max(numpy.abs(
#                 numpy.percentile(d[count_qc > 0], [1,99])))
#         plot_common_pre(title=p+' bias')
#         plt.pcolormesh(mesh_x, mesh_z, d, cmap=cmap_div, vmin=-dRange, vmax=dRange)
#         plot_common_post(p+'_bias')


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
def plot_1d(exps, daterange, **kwargs):
    # TODO merge logic with plot_1d_z

    # handle either a list or a single exp being passed in
    if type(exps) is list:
        data = exps[0]
    else:
        data = exps
        exps = [data,]

    print("Plot 1D", exps[0])

    # TODO make sure meta data is same across exps
    # TODO, these are wrong, did this for quick fix to dates
    # put it back at some point
    bin_centers = [e.bin_edges[0][:-1] for e in exps]

    #bin_centers = (data.bin_edges[0][:-1] + data.bin_edges[0][1:]) / 2.0
    #bin_widths = (data.bin_edges[0][1:] - data.bin_edges[0][:-1])

    transpose = False
    if 'depth' in data.bin_dims:
        transpose = True

    exp = kwargs['exp_name'] if "exp_name" in kwargs else None
    title = kwargs['title']
    title = data.obsvar if title is None else title

    def plot_common_pre(title_add=""):
        plt.figure(figsize=(8.0, 4.0))
        ax = plt.axes()

        if exp is None:
            exp_str = ""
        else:
            exp_str= " (" + exp +")"
        dstr = [d.strftime("%Y-%m-%d") for d in daterange]
        dstr = dstr[0] if dstr[0] == dstr[1] else dstr[0] + ' to '+dstr[1]
        plt.annotate(dstr, ha='right', xycoords='axes points', xy=(480, -28.0))
        plt.title(title+" "+title_add + exp_str)

        if transpose:
            ax.set_ylabel(data.bin_dims[0])
        else:
            ax.set_xlabel(data.bin_dims[0])

        # if x axis is lat or lon, and 0 deg is in the range,
        # draw vertical line
        if data.bin_dims[0] in ('latitude','longitude') and \
            ( numpy.min(bin_centers[0]) < 0 < numpy.max(bin_centers[0]) ):
            plt.axvline(x=0.0, color='black', alpha=0.5)
        plt.grid(True, alpha=0.5)
        if 'depth' in data.bin_dims:
            plt.gca().invert_yaxis()

        # fix dates
        if "time" in exps[0].bin_dims:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%d'))


    def plot_common_post(type_):
        bbox_inches ='tight' if kwargs['thumbnail'] else None
        plt.savefig("{}{}.{}{}.{}.png".format(
            kwargs['prefix'], title.lower().replace(' ','_'), data.name, "",type_),
            bbox_inches = bbox_inches)
        plt.close()

    # # counts
    # bar_widths = bin_widths - numpy.min(bin_widths)*0.1
    # plot_common_pre(title_add="counts")
    # plt.bar(bin_centers, data.count(qc=False)/bar_widths, color='C0', alpha=0.4,
    #          width=bar_widths)
    # plt.bar(bin_centers, data.count(qc=True)/bar_widths, color='C0',
    #          width=bar_widths)
    # plot_common_post('counts')

    # rmsd
    plot_common_pre(title_add="rmsd")
    for e in enumerate(exps):
        for p in ('ombg', ):
            rmsd = e[1].rmsd(mode=p)
            d1 =  rmsd if transpose else bin_centers[e[0]]
            d2 =  bin_centers[e[0]] if transpose else rmsd
            plt.plot(d1, d2, color='C{}'.format(e[0]), alpha=1.0, lw=2.0,
                marker='.', markevery=[0,-1], ms=15,
                label = kwargs['label'][e[0]] if p == 'ombg' else None,
                ls='--' if p == 'oman' else None)
    plt.legend()
    if kwargs['diff']:
        # only plot a dark horizontal line at 0.0 if doing a diff
        plt.axhline(y=0.0, color='black', alpha=0.5)
    plot_common_post('rmsd')

    # bias
    plot_common_pre(title_add="bias")
    for e in enumerate(exps):
        for p in ('ombg',):
            bias = e[1].mean(mode=p)
            d1 =  bias if transpose else bin_centers[e[0]]
            d2 =  bin_centers[e[0]] if transpose else bias
            plt.plot(d1, d2, color='C{}'.format(e[0]), alpha=1.0, lw=2.0,
                marker='.', markevery=[0,-1], ms=15,
                label = kwargs['label'][e[0]] if p == 'ombg' else None,
                ls='--' if p == 'oman' else None)
    plt.legend()
    if transpose:
        plt.axvline(x=0.0, color='black', alpha=0.5)
    else:
        plt.axhline(y=0.0, color='black', alpha=0.5)
    plot_common_post('bias')


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
def main():
    import argparse
    import os

    parser = argparse.ArgumentParser(add_help=False)

    parser_req = parser.add_argument_group("required arguments")
    parser_req.add_argument('-e','--exp', required=True, action="append", nargs="+",
        help="one or more files to load. This argument can be repeated to handle"+
            " multiple experimetns (2D difference, or multiple line plots)")

    parser_opt = parser.add_argument_group("optional arguments")
    parser_opt.add_argument('-d','--diff', default=False, action="store_true",
        help = "The first '--exp' is subtracted from each subsequent '--exp', and then" +
            " is removed from the list of exps to plot." +
            " This option can only be used if more than 1 '--exp' is given. "+
            " If more than 2 '--exp' are given, 2D plots will not be generated, only 1D line plots.")
    parser_opt.add_argument("-h", "--help", action="help", help="show this help message and exit")
    parser_opt.add_argument('--label', type=str, nargs="+", required=False,
        help = "Names for each experiment to use in the plot legends," +
            " coresponding to the order given by the '-exp' arguments. The" +
            " number of names passed in must equal the number of '--exp'" +
            " arguments. If '-label' is not given, the labels default to 'exp<n>'")
    parser_opt.add_argument('--prefix', default="",
        help = "The directory/filename to prefex each generated output image file.")
    parser_opt.add_argument('--thumbnail', action="store_true", default=False,
        help='Create smaller images without labels and colorbars. '+
            "(Currently not working right, don't use it!)")
    parser_opt.add_argument('-t','--timeseries', default = False, action="store_true",
        help = "Normally, all files for a given '--exp' are merged into a single " +
            "set of binned stats before plotting. If '--timeseries' is present " +
            "the separarate files are not merged, but instead are used to add a " +
            "'time' dimension to the plots. I.e 0-D stats become a 1-D line plot, "+
            "1-D stats become a 2D Hovmoller, and 2D stats are not plotted, because "+
            "how do you expect me to plot in 3D?")
    parser_opt.add_argument('--title', type=str,
        help = "by default the variable name will be used as the beginning of "+
            "each plot title. This will override that.")

    args = parser.parse_args()

    # create output directories
    d = os.path.dirname(args.prefix)
    if d != "" and not os.path.exists(d):
        os.makedirs(d)

    # determine the name of the experiments, if not already given
    if args.label is None:
        args.label = [ "exp{}".format(i+1) for i in range(len(args.exp))]
    if len(args.label) != len(args.exp):
        print("ERROR: number of '-label' arguments must equal number of '-exp' given.")
        sys.exit(1)

    # read and merge data. If a timeseries is specified, stats are not merged
    exps=[]
    for files in enumerate(args.exp):
        data = []
        for f in sorted(files[1]):
            bs = BinnedStatsCollection.load(filename=f, exp=args.label[files[0]])
            data.append(bs)
        if args.timeseries:
            data = BinnedStatsCollection.timeseries(data)
        else:
            data = BinnedStatsCollection.merge(data)
        exps.append(data)

    # calculate the difference between the first experiment, if doing a comparison
    if args.diff and len(exps) == 1:
        raise Exception("cannot use '--diff' with only one '-exp' given")
    elif args.diff:
        exp0_name = exps[0].exp()
        args.label = args.label[1:]
        for i in enumerate(exps[1:]):
            exps[i[0]+1] -= exps[0]
        exps = exps[1:]
        data = exps[0]

    print("")
    print("Generating plots for: ", data)

    # TODO check to make sure timeseries data is same format
    for k, v in data.binned_stats.items():

        # determine what kind of plot this is
        s = set(v.bin_dims)

        # get all data frames for the experiments
        exps_v = [ e.binned_stats[k] for e in exps ]
        daterange = exps[0].daterange


        # The following can only be done for a single experiment
        #----------------------------------------------------------------------

        #  # Multiple 2D lat/lon plots
        # if ( len(exps) == 1 and len(v.bin_dims) == 3
        #      and set(('latitude', 'longitude')).issubset(s)):
        #     plot_3d_xy(exps_v[0])

        # 2D plot
        if len(exps) == 1 and len(v.bin_dims) == 2:
            plot_2d(exps_v[0], **vars(args), exp_name=exps[0].exp(), daterange=daterange)


        # The following can only be done for any number of experiments
        #----------------------------------------------------------------------

        # probably a 1D line plot over some other dimension
        elif len(v.bin_dims) == 1:
            exp_name = None if not args.diff else \
                "<exp> - " + exp0_name
            plot_1d(exps_v, **vars(args), exp_name=exp_name, daterange=daterange)

        else:
            # TODO 0-D data, can't think of anything to do with this
            print("Skipping ", v)

if __name__ == "__main__":
    main()