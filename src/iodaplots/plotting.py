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
    self._stats = stats
    self._dimensions = stats.bin_dims
    self._title = f"{stats.variable} {stats.name}"
    self._thumbnail = False if 'thumbnail' not in kwargs else kwargs['thumbnail']

    # annotations based on clipping dims
    # TODO do some smarter logic to figure out the precision of the datetime needed
    self._annotations = []
    for d in stats.clip_dims:
      if d.name == 'datetime':
        bs = [dateutil.parser.parse(str(b)).strftime("%Y-%m-%d %HZ") for b in d.bounds ]
        self._annotations.append(f'date range:  {bs[0]} to {bs[1]}')
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

      plt.title(self._title)


class PlotType1D(PlotType):
  @classmethod
  def create(cls, stats):
    if len(stats.bin_dims) == 1:
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
    self._ax.set_xlabel(self._dimensions[0].name)

    # get the data to plot, transpose it if required
    d = (self._dimensions[0].bin_centers, data)
    if self._transpose:
      d = reversed(d)

    plt.plot(*d)


class PlotType1DLat(PlotType1D):
  @classmethod
  def create(cls, stats):
    if stats.bin_dims[0].name in ('latitude',):
      return super().create(stats)

  def plot(self, data):
    super().plot(data)

    # 0 line if latitudes are in range
    b = self._stats.bin_dims[0].bounds
    if numpy.min(b) < 0 < numpy.max(b):
      plt.axvline(x=0.0, color='black', alpha=0.5)


class PlotType1DTimeseries(PlotType1D):
  @classmethod
  def create(cls, stats):
    if stats.bin_dims[0].name in ('datetime',):
      return super().create(stats)

  def plot(self, data):
    super().plot(data)


class PlotType1DProfile(PlotType1D):

  _profile_dims_inverted = (
    'depth',
    'air_pressure')

  _profile_dims = _profile_dims_inverted + (
    'height',)

  @classmethod
  def create(cls, stats):
    if stats.bin_dims[0].name in cls._profile_dims:
      return super().create(stats)

  def __init__(self, stats, **kwargs):
    super().__init__(stats, **kwargs)
    self._transpose = True
    self._invert_y = stats.bin_dims[0].name in self._profile_dims_inverted


# class PlotType1DTimeseries(PlotType1D):
#   pass


class PlotType2D(PlotType):
  @classmethod
  def create(cls, stats):
    if len(stats.bin_dims) == 2:
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
    if 'datetime' in [d.name for d in stats.bin_dims]:
      return super().create(stats)

  def __init__(self, stats, **kwargs):
    super().__init__(stats, **kwargs)
    self._transpose = self._dimensions[0].name == 'datetime'
    self._flip_y = False

    # longitude hovmollers are special
    if 'longitude' in [d.name for d in self._dimensions]:
      self._transpose = not self._transpose
      self._flip_y = True
 
  def plot(self, data):
    dims = self._dimensions

    if self._transpose:
      data = data.T
      dims = list(reversed(dims))

    plt.xlabel(dims[1].name)
    plt.ylabel(dims[0].name)
    
    if self._flip_y:
      plt.gca().invert_yaxis()

    # TODO test if dimensions are equal distant
    #plt.imshow(data)
    plt.pcolormesh(dims[1].bin_edges, dims[0].bin_edges, data, 
      cmap='rainbow', antialiased=True)

    # line at 0N if latitude is a dimension
    if dims[0].name == 'latitude':
      b = dims[0].bounds
      if numpy.min(b) < 0 < numpy.max(b):
        plt.axhline(y=0.0, color='black', alpha=0.5)

    super().plot(data)


class PlotType2DLatlon(PlotType2D):
  @classmethod
  def create(cls, stats):
    if set([d.name for d in stats.bin_dims]) == set(['latitude','longitude']):
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
    self._projection = getattr(sys.modules['cartopy.crs'], 'PlateCarree')(**projection_args)

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
    plt.pcolormesh(self._dimensions[0].bin_edges, self._dimensions[1].bin_edges, data, 
      cmap='rainbow', transform=ccrs.PlateCarree(), vmin=vmin, vmax=vmax, antialiased=True)

    super().plot(data)


# class PlotType2DVertical(PlotType2D):
#   pass

def plot(exps, names, **kwargs):

  s = iodaplots.BinnedStatsCollection.load(exps[0])
  for vk, v in s.variables.items():
    for bk, b in v.items():
      stats = b
      outfile=kwargs['output']+f'{vk}_{bk}'
     
      pt = PlotType.create(stats)
      if pt is None:
        _logger.error(f"Can't find a method to plot {stats}")
        continue

      _logger.info(f"plotting {stats}")
      _logger.info(f" with {pt}")

      methods = [
        ('stddev', stats.stddev),
        ('rmsd',   stats.rmsd),
        ('mean',  stats.mean),
        ('bias',  stats.mean)]

      fn = outfile + f'_count.png'
      pt.plot_before()
      pt.plot(stats.count)
      _logger.info(f'saving {fn}')
      plt.savefig(fn)
      plt.close()

      for v in stats.variables:
        for method in methods:
          # get data, determine output filename
          data = method[1](v)
          fn = outfile + f'_{v}_{method[0]}.png'

          # plot the figure
          pt.plot_before()
          pt.plot(data)

          # save figure and cleanup
          _logger.info(f'saving {fn}')
          plt.savefig(fn)
          plt.close()
