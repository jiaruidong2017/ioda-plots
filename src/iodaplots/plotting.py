# (C) Copyright 2019-2020 UCAR
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

from ruamel.yaml import YAML
import abc
import cartopy.crs as ccrs
from cartopy.mpl import gridliner
import collections
import dateutil
import fnmatch
import iodaplots
import logging
import matplotlib
import numpy
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import sys
import traceback

_logger = logging.getLogger(__name__)

PlottingGroup = collections.namedtuple('PlottingGroup',
                                       ('name','metric','stat','args'))

class PlotType(abc.ABC):
  @classmethod
  @abc.abstractmethod
  def create(cls, dim_names, exp_cnt):
    """
    return an instance of this class, or one of its subclasses,
    if the given dimensions are appropriate for this class' plotting type.
    Otherwise return None

    Args:
      dim_names: list(str)
      exp_cnt: (int) number of experiments that will be simultaneously plotted
    """
    # call create() on any subclasses
    # NOTE: subclasses will in turn call super().create(), winding up back here
    #  IF they self determine they are an appropriate type
    for c in cls.__subclasses__():
      c2 = c.create(dim_names, exp_cnt)
      if c2 is not None:
        return c2

    if cls == PlotType:
      return None
    else:
      return cls

  def __init__(self, data, dims, **kwargs):
    """
    Args:
      data: dict[('metric','exp')] = array
      dims: dict['exp'] = Dimension()
    """
    default_args = {
      'thumbnail': False,
    }
    self._args = {**default_args, **kwargs}
    self._data = data
    self._dims = list(dims.values())[0] # TODO calculate this
    self._dims_exp = dims
    self._title = kwargs['title']

    # annotations based on clipping dims
    # TODO do some smarter logic to figure out the precision of the datetime needed
    # TODO handle case where clip dims are different between experiments (datetime?)
    self._annotations = []
    # dims = stats[0].clip_dims
    # dims += [d for d in stats[0].bin_dims if d.name == 'datetime']
    # for d in dims:
    #   if d.name == 'datetime':
    #     bs = [dateutil.parser.parse(str(b)).strftime("%Y-%m-%d %HZ")
    #           for b in d.bounds ]
    #     self._annotations.append(f'{bs[0]} to {bs[1]}')
    #   else:
    #     self._annotations.append(f'{d.name}:  {d.bounds[0]} to {d.bounds[1]}')

  def plot_before(self):
    plt.figure()
    self._ax = plt.axes()

  @abc.abstractmethod
  def plot(self, data):
    if not self._args['thumbnail']:
      # annotations at the bottom
      y = 0.0
      for a in self._annotations:
        y+= 12
        plt.annotate(a, xy=(0.0, y), xycoords='figure points', weight='light')

      # title
      plt.title(self._title)

  def set_datetime_label(self, axis):
    # set labels on the time axis depending on the date range
    idx = [d.name for d in self._dims].index('datetime')
    dateLen = self._dims[idx].bounds[1] - self._dims[idx].bounds[0]
    dateLen = dateLen.astype('timedelta64[D]') / numpy.timedelta64(1, 'D')
    # TODO sub-daily resolution ?
    if dateLen <= 15: # show xaxis with daily resolution
      major = mdates.DayLocator()
      majorFmt = mdates.DateFormatter('%d')
      minor = major
    elif dateLen <= 31: # show xaxis with weekly resolution
      major = mdates.WeekdayLocator(byweekday=mdates.MO)
      majorFmt = mdates.DateFormatter('%m-%d')
      minor = mdates.DayLocator()
    elif dateLen <= 395: #show xaxis with monthly resolution
      major = mdates.MonthLocator()
      majorFmt = mdates.DateFormatter('%b')
      minor = major
    else: # otherwise, yearly resolution
      major = mdates.YearLocator()
      majorFmt = mdates.DateFormatter('%Y')
      minor = mdates.MonthLocator()

    if axis == 'x':
      ax = plt.gca().xaxis
    elif axis == 'y':
      ax = plt.gca().yaxis
    else:
      raise Exception('"axis" must be "x" or "y"')

    ax.set_major_locator(major)
    ax.set_minor_locator(minor)
    ax.set_major_formatter(majorFmt)


class PlotType1D(PlotType):
  @classmethod
  def create(cls, dim_names, exp_cnt):
    if exp_cnt >= 1 and len(dim_names) == 1:
      return super().create(dim_names, exp_cnt)

  def __init__(self, data, dims, **kwargs):
    super().__init__(data, dims, **kwargs)
    self._invert_y = False
    self._transpose = False

  def plot(self, data):
    super().plot(data)
    if self._invert_y:
      plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.5)

    if self._dims[0].name != 'datetime':
      self._ax.set_xlabel(self._dims[0].name)

    # get the data to plot, transpose it if required
    for k, d in data.items():
      metric, exp = k
      dims = self._dims_exp[exp]
      d2 = (dims[0].bin_centers, d)

      if self._transpose:
        d2 = reversed(d2)

      plt.plot(*d2)


class PlotType1DLat(PlotType1D):
  @classmethod
  def create(cls, dim_names, exp_cnt):
    if dim_names[0] == 'latitude':
      return super().create(dim_names, exp_cnt)

  def plot(self, data):
    super().plot(data)

    # 0 line if latitudes are in range
    b = self._dims[0].bounds
    if numpy.min(b) < 0 < numpy.max(b):
      plt.axvline(x=0.0, color='black', alpha=0.5)


class PlotType1DTimeseries(PlotType1D):
  @classmethod
  def create(cls, dim_names, exp_cnt):
    if dim_names[0] == 'datetime':
      return super().create(dim_names, exp_cnt)

  def plot(self, data):
    super().plot(data)
    self.set_datetime_label(axis='x')


class PlotType1DProfile(PlotType1D):

  _profile_dims_inverted = (
    'depth',
    'air_pressure')

  _profile_dims = _profile_dims_inverted + (
    'height',)

  @classmethod
  def create(cls, dim_names, exp_cnt):
    if dim_names[0] in cls._profile_dims:
      return super().create(dim_names, exp_cnt)

  def __init__(self, data, dims, **kwargs):
    super().__init__(data, dims, **kwargs)
    self._transpose = True
    self._invert_y = self._dims[0].name in self._profile_dims_inverted



class PlotType2D(PlotType):
  @classmethod
  def create(cls, dim_names, exp_cnt):
    if exp_cnt == 1 and len(dim_names) == 2:
      return super().create(dim_names, exp_cnt)

  def __init__(self, data, dims, **kwargs):
    default_args={
      'scale': 'linear',
      'vmin': None,
      'vmax': None,
    }
    super().__init__(data, dims, **{**default_args, **kwargs})

    self._projection = None
    self._transform = None
    self._transpose = False

  def plot(self, data):
    if len(data) > 1:
      raise NotImplementedError('PlotType2D.plot(): cannot plot more than '+
        'one metric at a time')

    idx = 1 if self._transpose else 0
    data = next(iter(data.values()))
    d = data.T if not self._transpose else data

    plt_args={}
    if self._transform is not None:
      plt_args['transform'] = self._transform

    # calculate adaptive range
    vmin = self._args['vmin']
    vmax = self._args['vmax']
    if vmin is None or vmax is None:
      _percentile = 99.0
      rng = numpy.percentile(numpy.ma.array(data).compressed(),
                             [(100.0-_percentile)/2.0,
                              (100.0-_percentile)/2.0 + _percentile ])
      vmin, vmax = tuple(rng)

    # figure out the scale type
    cmap='rainbow'
    if self._args['scale'] == 'div':
      cmap="RdBu_r"
      vmax = numpy.max(numpy.abs( (vmin, vmax) ))
      vmin = - vmax
    if 'cmap' in self._args:
      cmap = self._args['cmap']


    plt.pcolormesh(
      self._dims[idx].bin_edges, self._dims[1-idx].bin_edges,
      d, cmap=cmap, antialiased=True, vmin=vmin, vmax=vmax,
                   **plt_args)

    # TODO allow configurable use of contour instead of mesh
    # plt.contourf(
    #   self._dimensions[idx].bin_centers, self._dimensions[1-idx].bin_centers,
    #   d, cmap=cmap, antialiased=True, vmin=vmin, vmax=vmax, **args)

    if not self._args['thumbnail']:
      plt.colorbar(orientation='vertical', shrink=0.7, fraction=0.02)

    super().plot(data)


class PlotType2DHovmoller(PlotType2D):
  @classmethod
  def create(cls, dim_names, exp_cnt):
    if 'datetime' in dim_names:
      return super().create(dim_names, exp_cnt)

  def __init__(self, data, dims, **kwargs):
    default_args = {
    }
    super().__init__(data, dims, **{**default_args, **kwargs})
    # TODO cleanup the designation of time axis
    self._transpose = self._dims[0].name != 'datetime'
    self._flip_y = False

    # longitude hovmollers are special
    if 'longitude' in [d.name for d in self._dims]:
      self._transpose = not self._transpose
      self._flip_y = True

  def plot(self, data):

    idx = 1 if self._transpose else 0
    plt.xlabel(self._dims[idx].name)
    plt.ylabel(self._dims[1-idx].name)

    if self._flip_y:
      plt.gca().invert_yaxis()

    self.set_datetime_label('x' if self._transpose else 'y')

    super().plot(data)

    # line at 0N if latitude is a dimension
    if 'latitude' in [d.name for d in self._dims]:
      idx = 0 if self._transpose else 1
      b = self._dims[idx].bounds
      if numpy.min(b) < 0 < numpy.max(b):
        plt.axhline(y=0.0, color='black', alpha=0.5)

    plt.grid(True, alpha=0.5)


class PlotType2DLatlon(PlotType2D):
  @classmethod
  def create(cls, dim_names, exp_cnt):
    if set(dim_names) == set(['latitude','longitude']):
      return super().create(dim_names, exp_cnt)

  def __init__(self, data, dims, **kwargs):
    default_args = {
      'projection' : 'PlateCarree',
      'projection_args' : {
        'central_longitude': 205,
      }
    }
    super().__init__(data, dims, **{**default_args, **kwargs})

    # if latitude is not first in dimensions,
    # swap dimensions and indicate data should transposed
    self._transpose = self._dims[0].name != 'longitude'

    # TODO get projection from kwargs
    self._projection = getattr(
        sys.modules['cartopy.crs'],
        self._args['projection'])(**self._args['projection_args'])
    self._transform=ccrs.PlateCarree()

  def plot_before(self):
    plt.figure(figsize=(8.0, 4.0))
    self._ax = plt.axes(projection=self._projection)

  def plot(self, data):
    self._ax.coastlines(color='k', alpha=0.5)
    self._ax.background_patch.set_facecolor('lightgray')
    gl = self._ax.gridlines(alpha=0.5, color='k', linestyle='--')
    gl.xformatter = gridliner.LONGITUDE_FORMATTER
    gl.xlocator = mticker.FixedLocator([-90, 0, 90, 180, 270])
    gl.yformatter = gridliner.LATITUDE_FORMATTER
    gl.ylocator = mticker.FixedLocator([-90, -60, -30, 0, 30, 60, 90])

    super().plot(data)


# class PlotType2DVertical(PlotType2D):
#   pass

def plot(exps, names, config_file, **kwargs):

  # load/generate configuration
  if config_file is None:
    config = _make_default_config()
  else:
    cfg = YAML().load(config_file)
    config = cfg['plotting'] if 'plotting' in cfg else _make_default_config()

  # load all the experiment data
  exp_data = [iodaplots.BinnedStatsCollection.load(e) for e in exps]

  # take difference from first exp, if doing a diff
  if kwargs['diff']:
    if len(exps) < 2:
      raise Exception(f"must have >= 2 experiments when using --diff flag")

    _logger.info(f'Plotting difference from "{names[0]}"')
    exp_data = [ d.diff(exp_data[0]) for d in exp_data[1:]]
    names = [ f'{n}-{names[0]}' for n in names[1:]]

  # TODO pass to create() what is really needed, not "stats"
  # for each variable, and each binning of that variable, do plots
  for vk in exp_data[0].stats:
    exp_v = [ e.stats[vk] for e in exp_data ]
    for bk in exp_v[0]:
      # make sure clipping dims are the same across exps, otherwise send warning
      stats = [ e[bk] for e in exp_v]
      for s in stats[1:]:
        # TODO, test and warn
        pass

      outfile=kwargs['output']+f'{vk}_{bk}'

      # Get a plotting class capable of these dimensions
      # TOOD get a proper union of dimensions, when using multiple exps
      dim_names = [d.name for d in stats[0].bin_dims]
      plot_class = PlotType.create(dim_names=dim_names, exp_cnt=len(stats))
      if plot_class is None:
        _logger.warning(f"Can't find a method to plot {len(stats)} {stats[0]}")
        continue
      _logger.info(f"plotting {len(stats)} {stats[0]}")
      _logger.info(f" with {plot_class}")

      # determine which metrics/stats should be plotted
      match_args ={
        'variable': vk,
        'binning': bk,
        'class': plot_class.__name__,
        'metric': stats[0].variables,
        'diff': kwargs['diff']
      }
      groups = _generate_plot_groups(config, match_args)

      # global arguments
      # TODO allow for yaml overrides
      default_args = {
        'scale': 'linear' if not kwargs['diff'] else 'div',
      }

      # for each plot
      for g in groups:
        data = {}
        dims = {}
        name = ""
        args={} # TODO redo args to be per-group

        # pack the data to be plotted
        for i,n in enumerate(names):
          dims[n] = stats[i].bin_dims
          for g2 in g:
            f=getattr(stats[i], g2.stat)
            a = [] if g2.stat in ('count',) else [ g2.metric, ]
            data[(g2.name, n)] = f(*a)
            args = g2.args
            name = g2.name

        # set the args
        args = {
          **default_args,
          **args,
          'title': f'{vk} {name}', }

        # TODO set correct filename
        fn = outfile + f'_{name}.png'

        # plot!
        plotter = plot_class(data, dims, **args)
        plotter.plot_before()
        plotter.plot(data)
        _logger.info(f'saving {fn}')
        plt.savefig(fn)
        plt.close()


def _generate_plot_groups(config, match_args):
  # TODO combine the common 'basic' and 'composite' logic
  groups=[]

  if 'basic' in config:
    cfgs=[]
    # expand the "any" keywords
    for c in config['basic']:
      if 'any' in c:
        c2 = c.copy()
        del c2['any']
        for a in c['any']:
          cfgs.append({**c2,**a})
      else:
        cfgs.append(dict(c))

    # find a configuration that matches what we are goign to plot
    # TODO handle "variable","binning","class" that are lists
    for c in cfgs:
      valid=True
      for k, v in c.items():
        if k in ('stat', 'metric'):
          continue
        if not fnmatch.fnmatch(match_args[k], v):
          valid=False
          break

      if valid:
        metric=set(match_args['metric'])
        if 'metric' in c:
          m = c['metric']
          if type(m) is str:
            m = [m,]
          m = set(m)
          metric = metric.intersection(m)

        stat_orig = c['stat']
        stat = c['stat'].copy()
        # TODO read in real args from config file
        args = [{} for i in c['stat']]

        # change 'bias' to 'mean' with specific args
        for i, s in enumerate(stat):
          if s == 'bias':
            stat[i] = 'mean'
            args[i]['scale'] = 'div'


        for m in metric:
          names = [f'{m}_{s}' for s in stat_orig]
          groups += [ [PlottingGroup(n,m,s,a),] for n,s,a in
                         zip(names, stat, args)]

  if 'composite' in config:
    raise NotImplementedError()
    cfgs = []
    for c in config['composite']:
      # TODO split based on "any" keyword
      cfgs.append(c)

    for c in cfgs:
      valid=True
      for k, v in c.items():
        if k in ('stat','metric', 'plot', 'name'):
          continue
        if not fnmatch.fnmatch(match_args[k], v):
          valid = False
          break

      if valid:
        gs = []
        for i, p in enumerate(c['plot']):
          # TODO pass the correct args
          gs.append(PlottingGroup(f"{p['metric']}_{p['stat']}",
                                  p['metric'],p['stat'],
                                  {'ls':'--'}))
        groups += [ gs, ]

  # TODO set the name correctly, instead of passing back list, pass back a dict

  return groups

def _make_default_config():
  # TODO make this smarter. Only return "count" for one of the metrics
  return {
    'basic': [{
      'class':'*',
      'stat': ['rmsd','mean','stddev','count']} ]}