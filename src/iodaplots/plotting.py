# (C) Copyright 2019-2020 UCAR
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

import abc
import cartopy.crs as ccrs
from cartopy.mpl import gridliner
import dateutil
import iodaplots
import logging
import numpy
import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import sys
import traceback

_logger = logging.getLogger(__name__)

class PlotType(abc.ABC):
  @classmethod
  @abc.abstractmethod
  def create(cls, stats, **kwargs):
    """
    return an instance of this class, or one of its subclasses,
    if the given dimensions are appropriate for this class' plotting type.
    Otherwise return None
    """
    # call create() on any subclasses
    # NOTE: subclasses will in turn call super().create(), winding up back here
    #  IF they self determine they are an appropriate type
    for c in cls.__subclasses__():
      c2 = c.create(stats)
      if c2 is not None:
        if cls is PlotType:
          try:
            c2 = c2(stats, **kwargs)
          except Exception:
            _logger.error(f'error creating {c2}')
            _logger.error(traceback.format_exc())
            c2 = None
        return c2

    if cls == PlotType:
      return None
    else:
      return cls

  def __init__(self, stats, **kwargs):
    self.stats = stats # list(BinnedStats)
    # TODO set _dimensions to the union of dimensions
    self._dimensions = stats[0].bin_dims
    self._title = f"{stats[0].variable}"
    self._thumbnail = False if 'thumbnail' not in kwargs else kwargs['thumbnail']

    # annotations based on clipping dims
    # TODO do some smarter logic to figure out the precision of the datetime needed
    # TODO handle case where clip dims are different between experiments (datetime?)
    self._annotations = []
    dims = stats[0].clip_dims
    dims += [d for d in stats[0].bin_dims if d.name == 'datetime']
    for d in dims:
      if d.name == 'datetime':
        bs = [dateutil.parser.parse(str(b)).strftime("%Y-%m-%d %HZ")
              for b in d.bounds ]
        self._annotations.append(f'{bs[0]} to {bs[1]}')
      else:
        self._annotations.append(f'{d.name}:  {d.bounds[0]} to {d.bounds[1]}')

  def plot_before(self):
    plt.figure()
    self._ax = plt.axes()

  @abc.abstractmethod
  def plot(self, data):
    if not self._thumbnail:
      # annotations at the bottom
      y = 0.0
      for a in self._annotations:
        y+= 12
        plt.annotate(a, xy=(0.0, y), xycoords='figure points', weight='light')

      # title
      plt.title(self._title)


  def set_datetime_label(self, axis):
    # set labels on the time axis depending on the date range
    dim_names = [d.name for d in self._dimensions]
    idx = dim_names.index('datetime')
    dateLen = self._dimensions[idx].bounds[1] - self._dimensions[idx].bounds[0]
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
  def create(cls, stats):
    if len(stats) >= 1 and len(stats[0].bin_dims) == 1:
      return super().create(stats)

  def __init__(self, stats, **kwargs):
    super().__init__(stats, **kwargs)
    self._invert_y = False
    self._transpose = False

  def plot(self, data):
    super().plot(data)
    if self._invert_y:
      plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.5)

    if self._dimensions[0].name != 'datetime':
      self._ax.set_xlabel(self._dimensions[0].name)

    # get the data to plot, transpose it if required
    for i, d in enumerate(data):
      dims = self.stats[i].bin_dims
      d2 = (dims[0].bin_centers, d)

      if self._transpose:
        d2 = reversed(d2)

      plt.plot(*d2)


class PlotType1DLat(PlotType1D):
  @classmethod
  def create(cls, stats):
    if stats[0].bin_names[0] in ('latitude',):
      return super().create(stats)

  def plot(self, data):
    super().plot(data)

    # 0 line if latitudes are in range
    b = self.stats[0].bin_dims[0].bounds
    if numpy.min(b) < 0 < numpy.max(b):
      plt.axvline(x=0.0, color='black', alpha=0.5)


class PlotType1DTimeseries(PlotType1D):
  @classmethod
  def create(cls, stats):
    if stats[0].bin_names[0] in ('datetime',):
      return super().create(stats)

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
  def create(cls, stats):
    if stats[0].bin_names[0] in cls._profile_dims:
      return super().create(stats)

  def __init__(self, stats, **kwargs):
    super().__init__(stats, **kwargs)
    self._transpose = True
    self._invert_y = stats.bin_dims[0].name in self._profile_dims_inverted



class PlotType2D(PlotType):
  @classmethod
  def create(cls, stats):
    if len(stats)==1 and len(stats[0].bin_dims) == 2:
      return super().create(stats)

  def __init__(self, stats, **kwargs):
    super().__init__(stats, **kwargs)
    self._vmin = None
    self._vmax = None
    self._transpose = False

  def plot(self, data):
    if not self._thumbnail:
      plt.colorbar(orientation='vertical', shrink=0.7, fraction=0.02)

    super().plot(data)


class PlotType2DHovmoller(PlotType2D):
  @classmethod
  def create(cls, stats):
    if 'datetime' in stats[0].bin_names:
      return super().create(stats)

  def __init__(self, stats, **kwargs):
    super().__init__(stats, **kwargs)
    # TODO cleanup the designation of time axis
    self._transpose = self._dimensions[0].name == 'datetime'
    self._flip = False

    # longitude hovmollers are special
    if 'longitude' in [d.name for d in self._dimensions]:
      self._transpose = not self._transpose
      self._flip = True

  def plot(self, data):
    dims = self._dimensions

    # transpose data?
    if self._transpose:
      data[0] = data[0].T
      dims = list(reversed(dims))

    # draw the axis labels
    if self._flip:
      plt.xlabel(dims[1].name)
      plt.gca().invert_yaxis()
      self.set_datetime_label('y')
    else:
      plt.ylabel(dims[0].name)
      self.set_datetime_label('x')

    # TODO test if dimensions are equal distant
    #plt.imshow(data)
    plt.pcolormesh(dims[1].bin_edges, dims[0].bin_edges, data[0],
      cmap='rainbow', antialiased=True)

    # line at 0N if latitude is a dimension
    if dims[0].name == 'latitude':
      b = dims[0].bounds
      if numpy.min(b) < 0 < numpy.max(b):
        plt.axhline(y=0.0, color='black', alpha=0.5)

    plt.grid(True, alpha=0.5)

    super().plot(data)


class PlotType2DLatlon(PlotType2D):
  @classmethod
  def create(cls, stats):
    if set(stats[0].bin_names) == set(['latitude','longitude']):
      return super().create(stats)

  def __init__(self, stats, **kwargs):
    super().__init__(stats, **kwargs)

    # if latitude is not first in dimensions,
    # swap dimensions and indicate data should transposed
    self._data_T = self._dimensions[0].name == 'longitude'
    if not self._data_T:
      self._dimensions.reverse()

    # TODO get projection from kwargs
    projection_args = {'central_longitude': 205,}
    self._projection = getattr(sys.modules['cartopy.crs'],
                               'PlateCarree')(**projection_args)

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

    #transpose data, if input dimensions were flipped
    if self._data_T:
      data = data.T

    # calculate range
    vmin = self._vmin
    vmax = self._vmax
    if vmin is None or vmax is None:
      _percentile = 99.0
      rng = numpy.percentile(numpy.ma.array(data).compressed(),
                             [(100.0-_percentile)/2.0,
                              (100.0-_percentile)/2.0 + _percentile ])
      vmin, vmax = tuple(rng)

    # TODO test if dimensions are regular, if not, do pcolormesh
    # plt.imshow( data, transform=ccrs.PlateCarree(), interpolation='nearest',
    #     extent=( *self._dimensions[0].bounds, *self._dimensions[1].bounds),
    #     origin="lower", cmap='rainbow', vmin=vmin, vmax=vmax)
    plt.pcolormesh(self._dimensions[0].bin_edges, self._dimensions[1].bin_edges,
                   data[0], cmap='rainbow', transform=ccrs.PlateCarree(),
                   vmin=vmin, vmax=vmax, antialiased=True)

    super().plot(data)


# class PlotType2DVertical(PlotType2D):
#   pass

def plot(exps, names, **kwargs):

  # load all the experiment data
  exp_data = [iodaplots.BinnedStatsCollection.load(e) for e in exps]

  # take difference from first exp, if doing a diff
  if kwargs['diff']:
    if len(exps) >= 2:
      _logger.info(f'Plotting difference from "{names[0]}"')
      exp_data = [ d.diff(exp_data[0]) for d in exp_data[1:]]
    else:
      raise Exception(f"must have >= 2 experiments when using --diff flag")

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

      pt = PlotType.create(stats)
      if pt is None:
        _logger.warning(f"Can't find a method to plot {len(stats)} {stats[0]}")
        continue

      _logger.info(f"plotting {len(stats)} {stats[0]}")
      _logger.info(f" with {pt}")

      methods = [
        ('stddev', [s.stddev for s in stats]),
        ('rmsd', [s.rmsd for s in stats]),
        ('mean', [s.mean for s in stats])
        ]

      fn = outfile + f'_count.png'
      pt.plot_before()
      data = [s.count for s in stats]

      pt.plot(data)
      _logger.info(f'saving {fn}')
      plt.savefig(fn)
      plt.close()

      for v in stats[0].variables:
        for method in methods:

          # get data
          data = [m(v) for m in method[1]]

          #determine output filename
          fn = outfile + f'_{v}_{method[0]}.png'

          # plot the figure
          pt.plot_before()
          pt.plot(data)

          # save figure and cleanup
          _logger.info(f'saving {fn}')
          plt.savefig(fn)
          plt.close()