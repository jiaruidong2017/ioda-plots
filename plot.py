#!/usr/bin/env python3
from binnedStats import BinnedStats, BinnedStatsCollection
import os, sys
import numpy
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors

# TODO add support for multiple experiments
# TODO add "timeseries" line plots
# TODO finish 3d_xy
# TODO finish 1d_z

zdims = ('height','depth')
cmap_div="RdBu_r"
cmap_seq="inferno"

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
def plot_3d_xy(data):
    #######################################
    # TODO implement this!
    #######################################
    print("Plot 3D (latlon) ", data)
    # which dimension is the non lat/lon dim
    # zdim = (set(data.bin_dims) - set(('latitude','longitude'))).pop()


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
def plot_2d_xy(data, **kwargs):
    # TODO add real date annotation
    # TODO shrink the colorbar padding
    # TODO variable size based on dimension clipping           
    import cartopy.crs as ccrs
    import matplotlib.ticker as mticker
    from cartopy.mpl import gridliner

    print("Plot 2D-latlon ", data)
    dates=[datetime.now(), datetime.now()]
    proj = ccrs.PlateCarree(central_longitude=-155)
    trans = ccrs.PlateCarree()
    meshLons, meshLats = numpy.meshgrid(data.bin_edges[1], data.bin_edges[0])
    
    def plot_common_pre(title="", text=[]):
        plt.figure(figsize= (4.0,2.0) if kwargs['thumbnail'] else (8.0, 4.0) )
        ax = plt.axes(projection=proj)
        ax.coastlines(color='k', alpha=0.5)
        ax.background_patch.set_facecolor('lightgray')
        if not kwargs['thumbnail']:
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
            exp = kwargs['exp_name'] if "exp_name" in kwargs else ""
            plt.title(data.obsvar+" "+title+ " (" + exp +")")
            i = -24.0
            for t in text:
                plt.annotate(t, xycoords='axes points', xy=(0.0, i))
                i -= 12.0
            dstr = [d.strftime("%Y-%m-%d") for d in dates]
            dstr = dstr[0] if dstr[0] == dstr[1] else dstr[0] + ' to '+dstr[1]
            plt.annotate(dstr, ha='right', xycoords='axes points', xy=(420, -24.0))
        return ax

    def plot_common_post(type_):
        bbox_inches='tight' if kwargs['thumbnail'] else None
        if not kwargs['thumbnail']:
            plt.colorbar(orientation='vertical', shrink=0.8, fraction=0.02)
        plt.savefig("{}{}.{}.{}.png".format(
            kwargs['prefix'], data.obsvar, data.name, type_),
                    bbox_inches = bbox_inches)
        plt.close()

    # counts
    # NOTE dont plot these if doing an exp comparision
    if data.exps() == 1:
        for p in ('count', 'count_qc'):
            d = data.count(qc=p=='count_qc')
            dMax = numpy.max(d)
            dSum = numpy.sum(d)
            if p == 'count':
                dRange = numpy.percentile(d[d > 0], [99])[0]
            text = ["min: {:0.0f}".format(dSum),
                    "max: {:0.0f}".format(dMax),]
            plot_common_pre(title=p, text=text)
            plt.pcolormesh(meshLons, meshLats, d,
                        transform=trans, cmap=cmap_seq,
                        norm=colors.LogNorm(vmin=1.0, vmax=dRange))
            plot_common_post(p)

        # pct bad obs
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
        plt.pcolormesh(meshLons, meshLats, d, transform=trans, cmap=cmap_seq)
        plot_common_post('count_pctbad')

    count_qc = data.count(qc=True)

    # rmsd
    for p in ('ombg', 'oman'):
        d = data.rmsd(mode=p)
        dMax = numpy.max(d)
        dAvg = numpy.mean(d)
        if p == 'ombg':
            if data.exps() == 1:
                dRange = numpy.percentile(d[count_qc > 0], [1, 99])
                norm=colors.LogNorm(vmin=dRange[0], vmax=dRange[1])
                cmap=cmap_seq
            else:
                dRange = numpy.max(numpy.abs(
                    numpy.percentile(d[count_qc > 0], [1, 99])))
                norm=colors.Normalize(vmin=-dRange, vmax=dRange)
                cmap=cmap_div
                    
        text = ['max: {:0.2e}'.format(dMax),
                'avg: {:0.2e}'.format(dAvg),]
        plot_common_pre(title=p+" rmsd", text=text)
        plt.pcolormesh(meshLons, meshLats, d, transform=trans, cmap=cmap, norm=norm)
        plot_common_post(p+'_rmsd')

    # bias
    for p in ('ombg', 'oman'):
        d = data.mean(mode=p)
        dMax = max(numpy.max(d), abs(numpy.min(d)))
        dAvg = numpy.mean(d[count_qc > 0])
        if p == 'ombg':
            dRange = numpy.max(numpy.abs(
                numpy.percentile(d[count_qc > 0], [1, 99])))
        text = ['max: {:0.2e}'.format(dMax),
                'avg: {:0.2e}'.format(dAvg)]            
        plot_common_pre(title=p+" bias", text=text)
        plt.pcolormesh(meshLons, meshLats, d, transform=trans, cmap=cmap_div,
                       vmin=-dRange, vmax=dRange)
        plot_common_post(p+'_bias')


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
def plot_2d(data, **kwargs):
    print("Plot 2D ", data)

    #TODO figure out the correct order for the axes

    mesh_i, mesh_j = numpy.meshgrid(data.bin_edges[0], data.bin_edges[1])
    
    def plot_common_pre(title=""):
        plt.figure(figsize=(8.0, 4.0))
        ax = plt.axes()
        ax.set_facecolor('lightgray')
        plt.title(data.obsvar+" "+title)
        #plt.axvline(x=0.0, color='k', alpha=0.5)
        ax.set_xlabel(data.bin_dims[0])
        ax.set_ylabel(data.bin_dims[1])
        #if data.bin_dims[idx_z] in ('depth',):
         #   plt.gca().invert_yaxis()
        #dstr = [d.strftime("%Y-%m-%d") for d in dates]
        #dstr = dstr[0] if dstr[0] == dstr[1] else dstr[0] + ' to ' + dstr[1]
        #plt.annotate(dstr, ha='right', xycoords='axes points', xy=(420, -24.0))
        return ax
    
    def plot_common_post(type_):
        plt.colorbar(orientation='vertical', shrink=0.7, fraction=0.02)
        plt.savefig("{}{}.{}.{}.png".format(
            kwargs['prefix'], data.obsvar, data.name, type_))
        plt.close()

    # counts
    for p in ('count', 'count_qc'):
        d = data.count(qc=p=='count_qc')
        if p == 'count':
            dRange = numpy.percentile(d[d>0], [99])[0]
        plot_common_pre(title=p)
        plt.pcolormesh(mesh_i, mesh_j, d,
                       cmap=cmap_seq, norm=colors.LogNorm(vmin=1.0, vmax=dRange))
        plot_common_post(p)

    # pct bad
    count = data.count(qc=False)
    count_qc = data.count(qc=True)
    d = count - count_qc
    d[count > 0] /= count[count > 0]
    d = numpy.ma.masked_where(count == 0, d)
    # dMin = numpy.min(d[count > 0])
    # dMax = numpy.max(d)
    # dAvg = numpy.mean(d[count > 0])
    plot_common_pre(title=" count_pctbad")
    plt.pcolormesh(mesh_i, mesh_j, d, cmap=cmap_seq)
    plot_common_post('count_pctbad')
    
    # rmsd
    for p in ('ombg', 'oman'):        
        d = data.rmsd(mode=p)
        if p == 'ombg':
            dRange = numpy.percentile(d[count_qc > 0], [1,99])
        plot_common_pre(title=p+" rmsd")
        plt.pcolormesh(mesh_i, mesh_j, d, cmap=cmap_seq, norm=colors.LogNorm(*dRange))
        plot_common_post(p+'_rmsd')
        
    
    # bias
    for p in ('ombg', 'oman'):
        d = data.mean(mode=p)
        if p == 'ombg':
            dRange = numpy.max(numpy.abs(
                numpy.percentile(d[count_qc > 0], [1,99])))
        plot_common_pre(title=p+' bias')
        plt.pcolormesh(mesh_i, mesh_j, d, cmap=cmap_div, vmin=-dRange, vmax=dRange)
        plot_common_post(p+'_bias')
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
def plot_2d_z(data, **kwargs):
    print("Plot 2D (? x depth) ", data)

    # TODO plot variable size based on dimensions
    # TODO thumbnail mode
    # TODO add real date annotation
    # TODO shrink the colorbar padding
    dates=[datetime.now(), datetime.now()]

    idx_z = 0 if data.bin_dims[0] in zdims else 1
    idx_x = 1 if idx_z is 0 else 0
    mesh_x, mesh_z = numpy.meshgrid(data.bin_edges[idx_x], data.bin_edges[idx_z])

    def plot_common_pre(title=""):
        plt.figure(figsize=(8.0, 4.0))
        ax = plt.axes()
        ax.set_facecolor('lightgray')
        plt.title(data.obsvar+" "+title)
        plt.axvline(x=0.0, color='k', alpha=0.5)
        ax.set_xlabel(data.bin_dims[idx_x])
        ax.set_ylabel(data.bin_dims[idx_z])
        if data.bin_dims[idx_z] in ('depth',):
            plt.gca().invert_yaxis()
        dstr = [d.strftime("%Y-%m-%d") for d in dates]
        dstr = dstr[0] if dstr[0] == dstr[1] else dstr[0] + ' to ' + dstr[1]
        plt.annotate(dstr, ha='right', xycoords='axes points', xy=(420, -24.0))
        return ax

    def plot_common_post(type_):
        plt.colorbar(orientation='vertical', shrink=0.7, fraction=0.02)
        plt.savefig("{}{}_{}_{}.png".format(
            kwargs['prefix'], data.obsvar, data.name, type_))
        plt.close()

    # counts
    for p in ('count', 'count_qc'):
        d = data.count(qc=p=='count_qc')
        # dMax = numpy.max(d)
        # dSum = numpy.sum(d)
        if p == 'count':
            dRange = numpy.percentile(d[d>0], [99])[0]
        plot_common_pre(title=p)
        plt.pcolormesh(mesh_x, mesh_z, d,
                       cmap=cmap_seq, norm=colors.LogNorm(vmin=1.0, vmax=dRange))
        plot_common_post(p)

    # pct bad
    count = data.count(qc=False)
    count_qc = data.count(qc=True)
    d = count - count_qc
    d[count > 0] /= count[count > 0]
    d = numpy.ma.masked_where(count == 0, d)
    # dMin = numpy.min(d[count > 0])
    # dMax = numpy.max(d)
    # dAvg = numpy.mean(d[count > 0])
    plot_common_pre(title=" count_pctbad")
    plt.pcolormesh(mesh_x, mesh_z, d, cmap=cmap_seq)
    plot_common_post('count_pctbad')
    
    # rmsd
    for p in ('ombg', 'oman'):        
        d = data.rmsd(mode=p)
        # dMax = numpy.max(d)
        # dAvg = numpy.mean(d)
        if p == 'ombg':
            dRange = numpy.percentile(d[count_qc > 0], [1,99])

        plot_common_pre(title=p+" rmsd")
        plt.pcolormesh(mesh_x, mesh_z, d, cmap=cmap_seq, norm=colors.LogNorm(*dRange))
        plot_common_post(p+'_rmsd')

    # bias
    for p in ('ombg', 'oman'):
        d = data.mean(mode=p)
        # dMax = max(numpy.max(d), abs(numpy.min(d)))
        # dAvg = numpy.mean(d[count_qc > 0])
        if p == 'ombg':
            dRange = numpy.max(numpy.abs(
                numpy.percentile(d[count_qc > 0], [1,99])))
        plot_common_pre(title=p+' bias')
        plt.pcolormesh(mesh_x, mesh_z, d, cmap=cmap_div, vmin=-dRange, vmax=dRange)
        plot_common_post(p+'_bias')


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
def plot_1d(data, **kwargs):
    print("Plot 1D", data)

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

    # counts
    bar_widths = bin_widths - numpy.min(bin_widths)*0.1
    plot_common_pre(title="counts")
    plt.bar(bin_centers, data.count(qc=False)/bar_widths, color='C0', alpha=0.4,
             width=bar_widths)
    plt.bar(bin_centers, data.count(qc=True)/bar_widths, color='C0',
             width=bar_widths)
    plot_common_post('counts')

    # rmsd
    plot_common_pre(title="rmsd")
    for p in ('ombg', 'oman'):
        rmsd = data.rmsd(mode=p)
        plt.plot(bin_centers, rmsd, color='C0', alpha=1.0, lw=2.0,
             ls='--' if p == 'oman' else None)
    plt.axvline(x=0.0, color='black', alpha=0.5)
    plot_common_post('rmsd')

    # bias
    plot_common_pre(title="bias")
    for p in ('ombg', 'oman'):
        bias = data.mean(mode=p)
        plt.plot(bin_centers, bias, color='C0', alpha=1.0, lw=2.0,
             ls='--' if p == 'oman' else None)
    plt.axvline(x=0.0, color='black', alpha=0.5)
    plot_common_post('bias')

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
def plot_1d_z(data, daterange, **kwargs):
    # TODO handle counts correctly for timeseries data
    if kwargs['timeseries']:
        data_timeseries = data
        data = data_timeseries[-1]
    else:
        data_timeseries = [data,]

    print("Plot 1D (timeseries={})".format(kwargs['timeseries']), data)
            
    bin_centers = (data.bin_edges[0][:-1] + data.bin_edges[0][1:]) / 2.0
    bin_widths = (data.bin_edges[0][1:] - data.bin_edges[0][:-1])
    # count_qc = data.count(qc=True)

    def plot_common_pre(title=""):
        plt.figure(figsize=(4.0, 8.0))
        ax = plt.axes()
        if data.bin_dims[0] == 'depth':
            plt.gca().invert_yaxis()
        if not kwargs['thumbnail']:
            plt.title(data.obsvar+" "+title)
            dstr = [d.strftime("%Y-%m-%d") for d in daterange]
            dstr = dstr[0] if dstr[0] == dstr[1] else dstr[0] + " to " + dstr[1]        
            plt.annotate(dstr, xycoords="axes points", xy=(0, -30.0))
            ax.set_ylabel(data.bin_dims[0])
        return ax

    def plot_common_post(type_):
        bbox_inches ='tight' if kwargs['thumbnail'] else None
        plt.savefig("{}{}_{}_{}.png".format(
            kwargs['prefix'], data.obsvar, data.name, type_),
            bbox_inches = bbox_inches)
        plt.close()

    # counts
    bar_widths = bin_widths - numpy.min(bin_widths)*0.1
    plot_common_pre(title="counts")
    plt.barh(bin_centers, data.count(qc=False)/bar_widths, color='C0', alpha=0.4,
             height=bar_widths)
    plt.barh(bin_centers, data.count(qc=True)/bar_widths, color='C0',
             height=bar_widths)
    plot_common_post('counts')

    # rmsd
    plot_common_pre(title="rmsd")
    for p in ('ombg', 'oman'):
        for data in data_timeseries:
            rmsd = data.rmsd(mode=p)
            plt.plot(rmsd, bin_centers, color='C0', alpha=0.2,
                     ls='--' if p == 'oman' else None)
        plt.plot(rmsd, bin_centers, color='C0', alpha=1.0, lw=2.0,
             ls='--' if p == 'oman' else None)
    plt.axvline(x=0.0, color='black', alpha=0.5)
    plot_common_post('rmsd')

    # bias
    plot_common_pre(title="bias")
    for p in ('ombg', 'oman'):
        for data in data_timeseries:
            bias = data.mean(mode=p)
            plt.plot(bias, bin_centers, color='C0', alpha=0.2,
                     ls='--' if p == 'oman' else None)
        plt.plot(rmsd, bin_centers, color='C0', alpha=1.0, lw=2.0,
             ls='--' if p == 'oman' else None)
    plt.axvline(x=0.0, color='black', alpha=0.5)
    plot_common_post('bias')
            


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

    # read and merge data
    # TODO I don't like the whole "timeseries" business
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

    if len(exps) == 1:
        data = exps[0]
    elif len(exps) == 2:
        data = exps[0] - exps[1]
    else:
        print("ERROR: unable to handle >2 exps currently.")
        sys.exit(1)

    print("Generating plots for: ", data)
    print("")

    # TODO check to make sure timeseries data is same format
    for k, v in data.binned_stats.items():
        
        # determine what kind of plot this is
        s = set(v.bin_dims)

         # Multiple 2D lat/lon plots
        if len(v.bin_dims) == 3 and set(('latitude', 'longitude')).issubset(s):           
            plot_3d_xy(v)
        
        # 2D lat/lon plot
        elif set(('latitude', 'longitude')) == s:            
            plot_2d_xy(v, **vars(args), exp_name=data.exp())

        # 2D plot that is NOT lat/lon
        elif len(v.bin_dims) == 2:
            plot_2d(v, **vars(args), exp_name=data.exp())

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
        
        # probably a 1D line plot over some other dimension
        elif len(v.bin_dims) == 1:
            plot_1d(v, **vars(args))
        
        else:
            # TODO handle 0-D data with a timeseries            
            print("whoa, you shouldn't get here.")

if __name__ == "__main__":
    main()