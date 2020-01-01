#!/usr/bin/env python3
from binnedStats import BinnedStats, BinnedStatsCollection
import os, sys
import numpy
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors

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
def plot_2d(data, **kwargs):
    
    # figure out what type of plot
    mesh_opt = {}
    xy_type = ""
    if set(('latitude','longitude')) == set(data.bin_dims):
        # lat/lon
        xy_type = "latlon"
        proj = ccrs.PlateCarree(central_longitude=-155)
        mesh_opt['transform'] = ccrs.PlateCarree()
        i_dim = data.bin_dims.index('longitude')
        j_dim = data.bin_dims.index('latitude')
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
            pass
    else:
        # generic coordinates
        proj = None
        i_dim = 0
        j_dim = 1
        def plot_type_pre(ax):
            pass        

    print("Plot 2D ",xy_type, data)

    # generate x/y coordinates
    mesh_i, mesh_j = numpy.meshgrid(data.bin_edges[i_dim], data.bin_edges[j_dim])
    
    #TODO, use the right dates
    dates=[datetime.now(), datetime.now()]

    def plot_common_pre(title="", text=[]):
        plt.figure(figsize=(8.0, 4.0))
        ax = plt.axes(projection=proj)
        ax.set_facecolor('lightgray')
        #plt.axvline(x=0.0, color='k', alpha=0.5)
        ax.set_xlabel(data.bin_dims[0])
        ax.set_ylabel(data.bin_dims[1])
        #if data.bin_dims[idx_z] in ('depth',):
         #   plt.gca().invert_yaxis()
        #dstr = [d.strftime("%Y-%m-%d") for d in dates]
        #dstr = dstr[0] if dstr[0] == dstr[1] else dstr[0] + ' to ' + dstr[1]
        #plt.annotate(dstr, ha='right', xycoords='axes points', xy=(420, -24.0))
        exp = kwargs['exp_name'] if "exp_name" in kwargs else ""
        plt.title(data.obsvar+" "+title+ " (" + exp +")")
        i = -24.0
        for t in text:
            plt.annotate(t, xycoords='axes points', xy=(0.0, i))
            i -= 12.0
        dstr = [d.strftime("%Y-%m-%d") for d in dates]
        dstr = dstr[0] if dstr[0] == dstr[1] else dstr[0] + ' to '+dstr[1]
        plt.annotate(dstr, ha='right', xycoords='axes points', xy=(420, -24.0))
        plot_type_pre(ax)
        return ax
    
    def plot_common_post(type_):
        bbox_inches='tight' if kwargs['thumbnail'] else None
        plt.colorbar(orientation='vertical', shrink=0.7, fraction=0.02)
        plt.savefig("{}{}.{}.{}.png".format(
            kwargs['prefix'], data.obsvar, data.name, type_),
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
        plot_common_pre(title=p, text=text)
        plt.pcolormesh(mesh_i, mesh_j, d, **mesh_opt,
            cmap=cmap_seq, norm=colors.LogNorm(vmin=1.0, vmax=dRange))
        plot_common_post(p)

    # pct bad
    count = data.count(qc=False)
    count_qc = data.count(qc=True)
    d = count - count_qc
    d[count > 0] /= count[count > 0]
    d = numpy.ma.masked_where(count == 0, d)
    dMin = numpy.min(d[count > 0])
    dMax = numpy.max(d)
    dAvg = numpy.mean(d[count > 0])
    text = ["min: {:0.2f} max: {:0.2f}".format(dMin, dMax),
            "avg: {:0.2f}".format(dAvg)]
    plot_common_pre(title=" count_pctbad", text=text)
    plt.pcolormesh(mesh_i, mesh_j, d, cmap=cmap_seq, **mesh_opt)
    plot_common_post('count_pctbad')
    
    # rmsd
    for p in ('ombg', 'oman'):        
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
        plot_common_pre(title=p+" rmsd", text=text)
        plt.pcolormesh(mesh_i, mesh_j, d, cmap=cmap, norm=norm, **mesh_opt)
        plot_common_post(p+'_rmsd')
        
    
    # bias
    for p in ('ombg', 'oman'):
        d = data.mean(mode=p)
        dMax = max(numpy.max(d), abs(numpy.min(d)))
        dAvg = numpy.mean(d[count_qc > 0])        
        text = ['max: {:0.2e}'.format(dMax),
                'avg: {:0.2e}'.format(dAvg)]            

        if p == 'ombg':
            dRange = numpy.max(numpy.abs(
                numpy.percentile(d[count_qc > 0], [1,99])))
        plot_common_pre(title=p+' bias', text=text)
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
def plot_1d(exps, **kwargs):
    print("Plot 1D", exps)

    # handle either a list or a single exp being passed in
    if type(exps) is list:
        data = exps[0]
    else:
        data = exps
        exps = [data,]

    # TODO make sure meta data is same across exps
    bin_centers = (data.bin_edges[0][:-1] + data.bin_edges[0][1:]) / 2.0
    bin_widths = (data.bin_edges[0][1:] - data.bin_edges[0][:-1])

    def plot_common_pre(title=""):
        plt.figure(figsize=(8.0, 4.0))
        ax = plt.axes()
        plt.title(data.obsvar+" "+title)
        ax.set_xlabel(data.bin_dims[0])
    
    def plot_common_post(type_):
        bbox_inches ='tight' if kwargs['thumbnail'] else None
        plt.savefig("{}{}.{}.{}.png".format(
            kwargs['prefix'], data.obsvar, data.name, type_),
            bbox_inches = bbox_inches)
        plt.close()

    # # counts
    # bar_widths = bin_widths - numpy.min(bin_widths)*0.1
    # plot_common_pre(title="counts")
    # plt.bar(bin_centers, data.count(qc=False)/bar_widths, color='C0', alpha=0.4,
    #          width=bar_widths)
    # plt.bar(bin_centers, data.count(qc=True)/bar_widths, color='C0',
    #          width=bar_widths)
    # plot_common_post('counts')

    # rmsd
    plot_common_pre(title="rmsd")
    for e in enumerate(exps):
        for p in ('ombg', 'oman'):
            rmsd = e[1].rmsd(mode=p)
            plt.plot(bin_centers, rmsd, color='C{}'.format(e[0]), alpha=1.0, lw=2.0,
                label = kwargs['label'][e[0]] if p == 'ombg' else None,
                ls='--' if p == 'oman' else None)
    plt.legend()
    plt.axvline(x=0.0, color='black', alpha=0.5)
    plot_common_post('rmsd')

    # bias
    plot_common_pre(title="bias")
    for e in enumerate(exps):
        for p in ('ombg', 'oman'):
            bias = e[1].mean(mode=p)
            plt.plot(bin_centers, bias, color='C{}'.format(e[0]), alpha=1.0, lw=2.0,
                label = kwargs['label'][e[0]] if p == 'ombg' else None,
                ls='--' if p == 'oman' else None)
    plt.legend()
    plt.axvline(x=0.0, color='black', alpha=0.5)
    plot_common_post('bias')


# # ------------------------------------------------------------------------------
# # ------------------------------------------------------------------------------
# def plot_1d_z(data, daterange, **kwargs):
#     # TODO handle counts correctly for timeseries data
#     if kwargs['timeseries']:
#         data_timeseries = data
#         data = data_timeseries[-1]
#     else:
#         data_timeseries = [data,]

#     print("Plot 1D (timeseries={})".format(kwargs['timeseries']), data)
            
#     bin_centers = (data.bin_edges[0][:-1] + data.bin_edges[0][1:]) / 2.0
#     bin_widths = (data.bin_edges[0][1:] - data.bin_edges[0][:-1])
#     # count_qc = data.count(qc=True)

#     def plot_common_pre(title=""):
#         plt.figure(figsize=(4.0, 8.0))
#         ax = plt.axes()
#         if data.bin_dims[0] == 'depth':
#             plt.gca().invert_yaxis()
#         if not kwargs['thumbnail']:
#             plt.title(data.obsvar+" "+title)
#             dstr = [d.strftime("%Y-%m-%d") for d in daterange]
#             dstr = dstr[0] if dstr[0] == dstr[1] else dstr[0] + " to " + dstr[1]        
#             plt.annotate(dstr, xycoords="axes points", xy=(0, -30.0))
#             ax.set_ylabel(data.bin_dims[0])
#         return ax

#     def plot_common_post(type_):
#         bbox_inches ='tight' if kwargs['thumbnail'] else None
#         plt.savefig("{}{}_{}_{}.png".format(
#             kwargs['prefix'], data.obsvar, data.name, type_),
#             bbox_inches = bbox_inches)
#         plt.close()

#     # counts
#     bar_widths = bin_widths - numpy.min(bin_widths)*0.1
#     plot_common_pre(title="counts")
#     plt.barh(bin_centers, data.count(qc=False)/bar_widths, color='C0', alpha=0.4,
#              height=bar_widths)
#     plt.barh(bin_centers, data.count(qc=True)/bar_widths, color='C0',
#              height=bar_widths)
#     plot_common_post('counts')

#     # rmsd
#     plot_common_pre(title="rmsd")
#     for p in ('ombg', 'oman'):
#         for data in data_timeseries:
#             rmsd = data.rmsd(mode=p)
#             plt.plot(rmsd, bin_centers, color='C0', alpha=0.2,
#                      ls='--' if p == 'oman' else None)
#         plt.plot(rmsd, bin_centers, color='C0', alpha=1.0, lw=2.0,
#              ls='--' if p == 'oman' else None)
#     plt.axvline(x=0.0, color='black', alpha=0.5)
#     plot_common_post('rmsd')

#     # bias
#     plot_common_pre(title="bias")
#     for p in ('ombg', 'oman'):
#         for data in data_timeseries:
#             bias = data.mean(mode=p)
#             plt.plot(bias, bin_centers, color='C0', alpha=0.2,
#                      ls='--' if p == 'oman' else None)
#         plt.plot(rmsd, bin_centers, color='C0', alpha=1.0, lw=2.0,
#              ls='--' if p == 'oman' else None)
#     plt.axvline(x=0.0, color='black', alpha=0.5)
#     plot_common_post('bias')
            


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
def main():
    import argparse
    import os

    parser = argparse.ArgumentParser()
    
    parser_req = parser.add_argument_group("required arguments")
    parser_req.add_argument('-exp', required=True, action="append", nargs="+",
        help="one or more files to load. This argument can be repeated to handle multiple experimetns (diff or multiple line plots)")

    parser_opt = parser
    parser_opt.add_argument('--diff', default=False, action="store_true")
    parser_opt.add_argument('--prefix', default="")    
    parser_opt.add_argument('--thumbnail', action="store_true", default=False,
                        help='Create smaller images without labels and colorbars')
    parser_opt.add_argument('--timeseries', default = False, action="store_true",
                        help='')
    parser_opt.add_argument('-label', nargs="+", required=False)
    
    args = parser.parse_args()

    # create output directories
    d = os.path.dirname(args.prefix)
    if d is not "" and not os.path.exists(d):
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
            print('Loading: ', f)
            bs = BinnedStatsCollection.load(filename=f, exp=args.label[files[0]])
            data.append(bs)
        if args.timeseries:
            data = BinnedStatsCollection.timeseries(data)
        else:
            data = BinnedStatsCollection.merge(data)       
        exps.append(data)

    # calculate the difference between the first experiment, if doing a comparison
    if args.diff and len(exps) == 1:
        raise Exception("cannot use '--compare' with only one '-exp' given")
    elif args.diff:
        for i in enumerate(exps[1:]):
            exps[i[0]+1] -= exps[0]
        exps = exps[1:]


    print("Generating plots for: ", data)
    print("")

    # TODO check to make sure timeseries data is same format
    for k, v in data.binned_stats.items():
        
        # determine what kind of plot this is
        s = set(v.bin_dims)

        # get all data frames for the experiments
        exps_v = [ e.binned_stats[k] for e in exps ]


        # The following can only be done for a single experiment
        #----------------------------------------------------------------------

        #  # Multiple 2D lat/lon plots
        # if ( len(exps) == 1 and len(v.bin_dims) == 3 
        #      and set(('latitude', 'longitude')).issubset(s)):
        #     plot_3d_xy(exps_v[0])
        
        # 2D plot
        if len(exps) == 1 and len(v.bin_dims) == 2:
            plot_2d(exps_v[0], **vars(args), exp_name=data.exp())

        # # 2D cross section plots wrt depth ( or some other variable)  
        # # TODO check this          
        # elif len(v.bin_dims) == 2 and \
        #         len(set(('latitude', 'longitude')).intersection(s)) == 1:            
        #     plot_2d_z(v, **vars(args))
        
        # # 1D line plots with height
        # elif len(v.bin_dims) == 1 and  v.bin_dims[0] in zdims:
        #     if args.timeseries:
        #         v = [d.binned_stats[k] for d in data]            
        #     plot_1d_z(v, data.daterange, **vars(args))
        
        # The following can only be done for any number of experiments
        #----------------------------------------------------------------------

        # probably a 1D line plot over some other dimension
        elif len(v.bin_dims) == 1:
            plot_1d(exps_v, **vars(args))
        
        else:
            # TODO handle 0-D data with a timeseries            
            print("Skipping ", v)

if __name__ == "__main__":
    main()