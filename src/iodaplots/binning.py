# (C) Copyright 2019-2020 UCAR
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

from ruamel.yaml import YAML
import collections
import logging
import numpy
import pandas
import re
import xarray

_logger = logging.getLogger(__name__)

__all__ = [
  'Dimension',
  'BinnedStats',
  'BinnedStatsCollection'
]

class Dimension:
  """
  Contains information about a single dimension.

  This is used for both defining how binning is performed as well as
  regional cropping (which is essentially a bin of size 1). This class
  can be initilized with either a fixed resolution, or specific list
  of edge values for the bins.
  """

  @classmethod
  def from_xarray(cls, name, binning, data):
    """ Recreate a Dimension class based on data serialized to an xarray """
    # TODO clean this up, just pass in the appropriate section of the xarray
    d = data[f'{name}%edges@{binning}']
    return cls(name, bins=d)

  def __init__(self, name, resolution=None, bins=None, bounds=None):
    self.bin_edges = bins # note, bins are the edges
    self.name = name

    # make sure exactly one from (resolution, bins) is defined
    if sum( x is not None for x in (resolution, bins)) == 2:
      raise Exception(('error creating Dimension for "{0}":'+
                       '"resolution" and "bins" cannot both be defined.').format(name))

    # if bounds is given, make sure min value is first
    if bounds is not None:
      if bounds[0] > bounds[1]:
        raise Exception(f'bounds for "{name}" must be in increasing order')

    # if defining the dimension based on regular resolution
    if resolution is not None:
      if resolution <= 0 :
        raise Exception(f'resolution for "{name}" must be > 0.0')

      # set the default bounds, if none were given
      if bounds is None:
        if self.name == "latitude":
          bounds = [-90, 90]
        elif self.name == "longitude":
          bounds = [-180, 180]
        else:
          raise Exception('Bounds need to be given for non lat/lon '
                           + 'dimensions if "resolution" is specified')

      self.bin_edges = numpy.arange(bounds[0], bounds[1] + resolution/2.0, resolution)

      if not numpy.array_equal(self.bounds, numpy.array(bounds)):
        _logger.warning(f'bounds were changed for dimension "{self.name}" to stay a fixed resolution')

    # if defining the dimension based on specified bin edges
    elif bins is not None:
      if bounds is not None:
        raise Exception (f'Error for dimension "{name}": cannot specify both "bins" and "bounds"')

      # make sure that the input bin edges are monotonic
      bins = numpy.array(bins)
      if not numpy.all( bins[1:] >= bins[:-1]):
        raise Exception(f'binning for "{name}" is not monotonically increasing')

      self.bin_edges = bins

    # otherwise, not a dimension used for binning, bound selection only
    elif bounds is not None:
      self.bin_edges = numpy.array(bounds)

    else:
      raise Exception(f'No specs were given for "{name}", try again.')

  def merge(self, other):
    if self.name != other.name:
      raise Exception(f'Trying to merge two different dimensions "{self.name}" and "{other.name}"')

    if self.bin_count > 1:
      if ( numpy.sum(other.bin_edges - self.bin_edges)) > 1e-5:
        raise Exception(f'Trying to merge dimension "{self.name}" with different bins')
    else:
      # only a single bin, we are using this dimension for datetime bounds?
      self.bin_edges[0] = min(other.bin_edges[0], self.bin_edges[0])
      self.bin_edges[1] = max(other.bin_edges[1], self.bin_edges[1])

  def cat(self, other):
    # modify value at the joining edges
    self.bin_edges[-1] += (other.bin_edges[0] - self.bin_edges[-1])/2.0

    # add rest of the edges from the other set
    self.bin_edges = numpy.concatenate( (self.bin_edges, other.bin_edges[1:]))

    # ensure the result is monotonic
    if not numpy.all( self.bin_edges[1:] >= self.bin_edges[:-1]):
        raise Exception(f'unable to concatenate dimension "{self.name}", result is not monotonic.')

  @property
  def bin_centers(self):
    dt = (self.bin_edges[1:] - self.bin_edges[:-1]) / 2.0
    return  self.bin_edges[:-1] + dt

  @property
  def bin_count(self):
    return len(self.bin_edges) - 1

  @property
  def bounds(self):
    return self.bin_edges[ [0,-1] ]

  def __repr__(self):
    return f'<Dimension("{self.name}", bounds={self.bounds}, bins={len(self.bin_edges)-1}>'


class BinnedStats:

  @classmethod
  def from_ioda(cls, data, variable, name, dimensions, metrics, qc_metavar=None):
    _logger.debug(f'Creating BinnedStats "{name}" for "{variable}""')

    args = {
      'variable': variable,
      'name': name
    }

    dimensions  = [Dimension(**d) for d in dimensions]
    args['dimensions'] = dimensions
    _logger.debug(f' with dimensions: {[d.name for d in dimensions]}')

    # make sure all the desired dimensions actually exist
    # TODO handle more gracefully, and print more info
    for d in dimensions:
      if d.name+"@MetaData" not in data.variables:
        raise Exception(f'Dimension "{d.name}" specified in config but not found in data')

    # Make sure dimensions are only listed once
    for k,v in collections.Counter([d.name for d in dimensions]).items():
      if v > 1:
        raise Exception(f'Dimension "{k}" is listed more than once')

    # generate clipping mask
    # using dimensions that are only bin size 1
    mask_clip = None
    for d in [d for d in dimensions if d.bin_count == 1]:
      #ensure clipping for longitude is in range -180 to 180
      # TODO allow for cyclic longitude in clipping
      if d.name == "longitude":
        if d.bounds[0] < -180 or d.bounds[1] > 180:
          NotImplementedError("unable to handle longitude clipping outside -180,180")

      # add this dimension to the mask
      dv = numpy.array(data[d.name+'@MetaData'])
      dv = numpy.logical_and( dv >= d.bounds[0], dv <= d.bounds[1])
      mask_clip = dv if mask_clip is None else numpy.logical_and(mask_clip, dv)

    # create QC mask
    if qc_metavar is not None:
      mask_qc = numpy.array(data[f'{variable}@{qc_metavar}'] == 0)
      if mask_clip is not None:
        mask_qc = numpy.logical_and(mask_qc, mask_clip)
    else:
      mask_qc = mask_clip

    # get the coordinate values for each dimension.
    dim_bins = []
    dim_vals = []
    dim_vals_qc = []
    for d in [d for d in dimensions if d.bin_count > 1]:
      dv = numpy.array(data[d.name+'@MetaData'])
      # make sure longitude between -180 and 180
      if d.name == "longitude":
        dv[dv >= 180.0] -= 360.0
        dv[dv < -180.0] += 360.0
      dim_bins.append(d.bin_edges)
      dim_vals.append(dv if mask_clip is None else dv[mask_clip])
      dim_vals_qc.append(dv if mask_qc is None else dv[mask_qc])

    # if there are no dimensions, quick hack to force it to use all obs
    if len(dim_bins) == 0:
      dv = numpy.array(data['latitude@MetaData'])
      dim_bins.append( numpy.array([-100, 100]))
      dim_vals.append(dv if mask_clip is None else dv[mask_clip])
      dim_vals_qc.append(dv if mask_qc is None else dv[mask_qc])

    # simple counts
    count, _ = numpy.histogramdd(dim_vals, dim_bins)
    count_qc, _ = numpy.histogramdd(dim_vals_qc, dim_bins)
    args['count'] = count
    args['count_qc'] = count_qc
    #self.count_qc_x   *worry about later *

    # second order statistics for all the metrics defined in "metrics"
    stat_m = {}
    stat_m2 = {}
    for m_name, m_eq in metrics.items():
      # prepare the metric definition to be evaluated
      # TODO add error checking here
      m_eq = re.sub('@([a-zA-Z0-9]*)', r"data.variables[variable+'@\g<1>']", m_eq)
      d = numpy.array(eval(m_eq))

      # apply QC masking and/or clipping
      if mask_qc is not None:
        d = d[mask_qc]

      # calculate the mean
      stat_m[m_name], _ = numpy.histogramdd(dim_vals_qc, dim_bins, weights=d)

      # sum of square difference from the mean
      # calculated using ofsset data (subtract global mean) to help numerical stability, maybe
      offset = 0.0 if len(d) == 0 else numpy.mean(d)
      stat_m2[m_name], _ = numpy.histogramdd(dim_vals_qc, dim_bins, weights=(d-offset)**2)
      m = count_qc > 0
      stat_m2[m_name][m] = stat_m2[m_name][m]
      stat_m2[m_name][m] -= (stat_m[m_name][m]-count_qc[m]*offset)**2/count_qc[m]
    args['m'] = stat_m
    args['m2'] = stat_m2

    return cls(**args)

  @classmethod
  def from_xarray(cls, variable, name, data):
    _logger.debug(f'creating BinnedStats from xarray: variable="{variable}", binning="{name}"')

    # arguments to pass to __init__(), append onto this
    # as we create the variables
    args = {
      'variable': variable,
      'name': name }

    # get the dimensions
    dimensions = [re.match(f'([a-zA-Z0-9_]+)%edges@{name}$',v) for v in data.variables]
    dimensions = [d[1] for d in dimensions if d is not None]
    args['dimensions'] = [Dimension.from_xarray(d, name, data) for d in dimensions]

    # counts
    args['count'] = numpy.array(data[f'{variable}%count@{name}'])
    args['count_qc'] = numpy.array(data[f'{variable}%count_qc@{name}'])

    # get the metrics
    stat_m = {}
    stat_m2 = {}
    metrics = [re.match(f'{variable}%m%([a-zA-Z0_9_]+)@{name}', m) for m in data.variables]
    metrics = [m[1] for m in metrics if m is not None]
    for m in metrics:
      stat_m[m] = numpy.array(data[f'{variable}%m%{m}@{name}'])
      stat_m2[m] = numpy.array(data[f'{variable}%m2%{m}@{name}'])
    args['m'] = stat_m
    args['m2'] = stat_m2

    return cls(**args)

  def __init__(self, variable, name, dimensions, count, count_qc, m, m2):
    """ why are you calling this? Don't call this. Use from_xarry() or from_ioda()"""
    self.variable = variable
    self.name = name
    self.count = count
    self.count_qc = count_qc
    self._dimensions = collections.OrderedDict([(d.name, d) for d in dimensions])
    self._m = m
    self._m2 = m2

  @property
  def variables(self):
    return list(self._m.keys())

  @property
  def bin_dims(self):
    return [d for d in self._dimensions.values() if d.bin_count > 1]

  @property
  def bin_names(self):
    return [d.name for d in self._dimensions.values() if d.bin_count > 1]

  @property
  def clip_dims(self):
    return [d for d in self._dimensions.values() if d.bin_count == 1]

  def mean(self, variable):
    d = self._m[variable].copy()
    mask = self.count_qc > 0
    d[mask] /= self.count_qc[mask]
    d = numpy.ma.masked_where(numpy.logical_not(mask), d)
    return d

  def variance(self, variable):
    mask = self.count_qc > 1
    d = self._m2[variable].copy()
    d[mask] /= (self.count_qc[mask] - 1)
    d[d < 0] = 0.0
    d = numpy.ma.masked_where(numpy.logical_not(mask), d)
    return d

  def stddev(self, variable):
    return numpy.sqrt( self.variance(variable))

  def rmsd(self, variable):
    # indirectly calculated using the binned 1st and 2nd moments
    d = self._m2[variable].copy()
    m = self.count_qc > 0
    d[m] /= self.count_qc[m]
    d = d + self.mean(variable)**2
    d[d < 0] = 0.0
    return numpy.sqrt( d )

  def cat(self, other, dim):
    """Concatenate statistics along the time dimensions for two sets of statistics"""

    # does the binning alreay have an appropriate time dimension?
    s_expand = dim not in [d.name for d in self.bin_dims] and len(self.bin_dims) > 0
    o_expand = dim not in [d.name for d in other.bin_dims] and len(other.bin_dims) > 0

    self._dimensions[dim].cat(other._dimensions[dim])

    # concatenate the counts
    for v in ('count','count_qc'):
      s = getattr(self, v)
      o = getattr(other, v)
      s = numpy.expand_dims(s, -1) if s_expand else s
      o = numpy.expand_dims(o, -1) if o_expand else o
      setattr(self, v, numpy.concatenate((s,o),-1))

    # and concatentate the other 1st & 2nd moments
    for v in ('_m','_m2'):
      s = getattr(self, v)
      o = getattr(other, v)
      for m in s:
        if s_expand:
          s[m] = numpy.expand_dims(s[m], -1)
        if o_expand:
          o[m] = numpy.expand_dims(o[m], -1)
        s[m] = numpy.concatenate( (s[m], o[m]), -1)

  def merge(self, other):
    """ Merge the statistics for two separate sets of binned statistics"""
    # TODO make sure the two sets are congruent
    new_count_qc = self.count_qc + other.count_qc
    for k in self._m:
      # difference in mean, needed for variance update
      delta = self.mean(k) - other.mean(k)
      delta[self.count_qc == 0] = 0.0
      delta[other.count_qc == 0] = 0.0

      # update sums
      self._m[k] += other._m[k]

      # update sum of square from mean
      m = new_count_qc > 0
      self._m2[k] += other._m2[k]
      self._m2[k][m] += (delta[m]**2)*self.count_qc[m]*other.count_qc[m]/new_count_qc[m]

    # update fincal counts
    self.count_qc = new_count_qc
    self.count += other.count

    # update the clipping dimensions (should just be datetime?)
    for d in self._dimensions.keys():
      self._dimensions[d].merge(other._dimensions[d])


  def serialize(self):
    xr = xarray.Dataset()

    # add all the dimensions
    for d in self._dimensions.values():
      # # ignore bin centers?
      # n = f'{d.name}@{self.name}'
      # xr.update({n: (n, d.bin_centers)})
      n = f'{d.name}%edges@{self.name}'
      xr.update({n: (n, d.bin_edges)})

    # generate list of dimensions
    # if there are NO dimensions (i.e. global binning), use a dummpy
    # "none" dimension, because we need something
    if len(self.bin_dims) > 0:
      dims = (tuple([f'{d.name}@{self.name}' for d in self.bin_dims]))
    else:
      dims = (f'none@{self.name}',)

    # counts
    vbase = f'{self.variable}%'
    xr.update({vbase+f'count@{self.name}': ( dims, self.count)})
    xr.update({vbase+f'count_qc@{self.name}': ( dims, self.count_qc)})

    # first and second moments
    for v in self.variables:
      xr.update({vbase+f'm%{v}@{self.name}': (dims, self._m[v])})
      xr.update({vbase+f'm2%{v}@{self.name}': (dims, self._m2[v])})

    return xr

  def __str__(self):
    return f'<BinnedStats(variable="{self.variable}",name="{self.name}", dims={[d.name for d in self.bin_dims]} clipping={[d.name for d in self.clip_dims]})>'


class BinnedStatsCollection:

  @classmethod
  def from_ioda(cls, ioda_file, config_file):
    """ Using a yaml configuration file, and input ioda NetCDF file, generate binned statistics"""    # open the
    _logger.debug(f'binning {ioda_file}')

    # open the data
    data = xarray.open_dataset(ioda_file)

    # open/create the binning configuration
    if config_file is None:
      # if no config given, invent one
      config = cls.make_default_config(data)
    else:
      # otherwise read in a yaml file
      config=YAML().load(config_file)

    return cls.generate(data, **config)

  @classmethod
  def generate(cls, data, binning, metrics, variables=None):
    """ generate binned statistics using the input data and parameters """
    # determine the list of meta varaiable that are required for each variables
    # currently "ObsValue", the QC variable, and anything else defined
    # in the "metrics" section of the config
    required_cols=set(["ObsValue",])
    qc_metavar = cls._get_qc_metavar(data)
    if qc_metavar is not None:
      required_cols.add(qc_metavar)
    for v in metrics.values():
      required_cols |= set(re.findall('@([a-zA-Z0-9]+)', v))

    # get a list of valid variables
    valid_variables = cls._get_valid_variables(data, required_cols)
    if variables is None:
      variables = valid_variables
    else:
      variables = set(variables)
      if not valid_variables > variables:
        _logger.warning("Some variables listed in binning configuration are not present in the data file:")
        _logger.warning(f' unused variables: {list(variables - valid_variables)}')
      variables &= valid_variables
    variables = sorted(list(variables))
    _logger.debug(f'binning variables: {variables}')

    # convert datetime to correct format
    # need this working before being able to do binning along the time dimension
    time = pandas.to_datetime(numpy.array(data.variables['datetime@MetaData'].astype(str)))
    time = numpy.array([ d.replace(tzinfo=None) for d in time])
    data.update({'datetime@MetaData': ('nlocs', time)})

    # add datetime as a dimensions, if not already in the lists
    # as a simple cropping dimensions that includes all observations
    for b, _ in enumerate(binning):
      if 'datetime' not in [d['name'] for d in binning[b]['dimensions']]:
        v = data.variables['datetime@MetaData']
        binning[b]['dimensions'].append(
          {'name': 'datetime',
           'bounds': (
             numpy.array(numpy.min(v)),
             numpy.array(numpy.max(v)))
          })

    # generate the binning for each variable
    # TODO threadding pool here?
    stats = collections.defaultdict(dict)
    for v in list(variables):
      for b in binning:
        stats[v][b['name']] = BinnedStats.from_ioda(data, v, **b, metrics=metrics, qc_metavar=qc_metavar )

    return cls(stats)

  @classmethod
  def load(cls, bin_file):
    """ Read a prevously precomputed binned statistics file """
    _logger.debug(f'loading bin file: {bin_file}')
    data = xarray.open_dataset(bin_file)

    # generate list of variables
    # TODO read this in from a nc attribute
    variables = [re.match('([a-zA-Z0-9_]+)%count@',v) for v in data.variables]
    variables = list(collections.OrderedDict.fromkeys([v[1] for v in variables if v is not None]))

    # generate a list of binning names
    # TODO read this in from a nc attribute
    bins = [re.match('.+@(.+)', v) for v in data.variables]
    bins = list(collections.OrderedDict.fromkeys([b[1] for b in bins if b is not None]))

    # read it all in!
    stats = collections.defaultdict(dict)
    for v in list(variables):
      for b in list(bins):
        stats[v][b] = BinnedStats.from_xarray(variable=v, name=b, data=data)

    return cls(stats)

  def __init__(self, stats):
    """ Why are you calling this? Don't do that. You probably want load() or from_ioda()"""
    self.stats = stats

  @property
  def variables(self):
    return self.stats

  def merge(self, other):
    """ merge two sets of binned stats together"""
    for var_k, var_v in self.stats.items():
      for bin_k, bin_v in var_v.items():
        bin_v.merge(other.stats[var_k][bin_k])

  def cat(self, other, dim='datetime'):
    """ Concatenate binned stats along their time dimension"""
    for var_k, var_v in self.stats.items():
      for bin_k in list(var_v.keys()):
        bin_v = var_v[bin_k]
        # TODO flip bin/var ordering to make this less annoying
        if len(set([d.name for d in bin_v.bin_dims])-set( (dim,) )) >= 2:
          # We can't (yet!) plot in 3D, so there is no reason keeping 3+ dimensions
          _logger.warning(f'unable to cat "{var_k}" "{bin_k}", too many dimensions')
          var_v.pop(bin_k)
        else:
          bin_v.cat(other.stats[var_k][bin_k], dim)

  def save(self, output):
    # TODO this is messy, swap the "variable" "BinnedStats" order??
    # TODO save each BinnedStats to a separate netcdf group

    # get serialization of each BinnedStats
    xr = xarray.Dataset()
    for v in self.variables.values():
      for b in v.values():
        xr.update(b.serialize())

    # add compression to variables
    encoding={}
    for v in xr.variables:
      encoding[v] = {'zlib': True, 'complevel': 4}

    # save!
    xr.to_netcdf(output, format='NETCDF4', engine='netcdf4',
                 encoding=encoding)

  def diff(self, other):
    # I'm lazy, not creating a special class for collection of BinnedStatsDiff
    c=collections.namedtuple('BinnedStatsDiffCollection', ('stats',))

    bs={}
    for vk, v in self.stats.items():
      bs[vk]={}
      for bk, b in v.items():
        ob = other.stats[vk][bk]
        bs[vk][bk] = BinnedStatsDiff(b, ob)

    return c(bs)

  @staticmethod
  def _get_qc_metavar(data):
    """
    find the meta variable responsible for the qc flag
    (currently assumes the "EffectiveQC" metavar with the highest number after it)
    """
    all_metavars = list(set([k.split('@')[-1] for k in data.variables.keys()]))
    match = [ re.match("^EffectiveQC([0-9]*)$", v) for v in all_metavars]
    match = [m for m in match if m is not None]
    if len(match) == 0:
      return None
    suffix = numpy.max([ int(m[1]) if m[1].isdigit() else None for m in match])
    suffix = "" if suffix is None else suffix
    return f"EffectiveQC{suffix}"

  @staticmethod
  def _get_valid_variables(data, required_cols=None):
    # get the list of all variables, and columns for each var
    all_obsvars = collections.defaultdict(set)
    for v in data.variables.keys():
      token = v.split('@')
      all_obsvars[token[0]].add(token[1])

    # check which variables have the required columns
    valid_obsvars = set()
    for v in all_obsvars:
      if all_obsvars[v].issuperset(required_cols):
        valid_obsvars.add(v)
    return valid_obsvars

  @staticmethod
  def make_default_config(data):
    """
    Try to make an educated guess about what the binning configuration
    should be based on some of the metadata in the input files.
    Obviously won't be as good as a manually created config yaml, but
    should be a good starting point for someone.
    """

    binning=[]

    # 2D lat lon
    if set(('latitude@MetaData', 'longitude@MetaData')) < set(data.variables):
      binning.append({
        'name': 'latlon',
        'dimensions': [
          {'name': 'latitude',
           'resolution': 1.0},
          {'name': 'longitude',
           'resolution': 1.0},]
      })

      # and a global binning
      binning.append({
        'name':'global',
        'dimensions':[]
      })

    # vertical profiles
    # TODO this doesn't work when merging files
    #  due to the likely different bounds
    for v in ('depth','height','air_pressure'):
      v2=f'{v}@MetaData'
      if v2 in data.variables:
        vmin = numpy.min(numpy.array(data[v2]))
        vmax = numpy.max(numpy.array(data[v2]))
        #TODO do better error checking than this
        if numpy.max(numpy.abs(numpy.array([vmin,vmax]))) > 1e12:
          continue
        binning.append({
          'name': 'profile',
          'dimensions': [
            {'name': v,
             'resolution': 20.0,
             'bounds': [vmin, vmax]}
          ]
        })

    # take a guess at what metrics to look at
    metrics = {}
    metavars = set([k.split('@')[-1] for k in data.variables])
    if 'ObsValue' in metavars:
      metrics['obs'] = '@ObsValue'

    if 'ombg' in metavars:
      metrics['ombg'] = '@ombg'
    elif set(('hofx','ObsValue')) < metavars:
      metrics['ombg'] = '@ObsValue - @hofx'

    if 'oman' in metavars:
      metrics['oman'] = '@oman'
    else:
      pass
      # TODO, find last hofx[0-9]+ and use that for 'oman'

    config={
      'binning': binning,
      'metrics': metrics,
    }

    return config


class BinnedStatsDiff:

  def __init__(self, stats1, stats2):
    # sanity checks
    if stats1.variable != stats2.variable:
      raise Exception(f'BinnedStatsDiff(): variable names do not match: "{stats1.variable}" and "{stats2.variable}"')
    if stats1.variables != stats2.variables:
      raise Exception(f'BinnedStatsDiff(): variables do not match: "{stats1.variables}" and "{stats2.variables}"')
    if stats1.name != stats2.name:
      raise Exception(f'BinnedStatsDiff(): binning names do not match: "{stats1.name}" and "{stats2.name}"')

    self.stats1 = stats1
    self.stats2 = stats2

    # make sure clipping dimensions are similar enough
    for d1, d2 in zip(stats1.clip_dims, stats2.clip_dims):
      if d1.name != d2.name:
        raise Exception(f'BinnedStatsDiff(): clipping dimension names do not match: {d1.name} and {d2.name}')

      if d1.name == 'datetime':
        # calculate percentage overlap in the bounds
        overlap_range = numpy.min((d1.bounds[1], d2.bounds[1])) - numpy.max((d1.bounds[0], d2.bounds[0]))
        total_range = numpy.max( (d1.bounds, d2.bounds)) - numpy.min( (d1.bounds, d2.bounds))
        overlap = overlap_range/total_range
        if overlap < 0.95:
          _logger.warn(f'binning "{stats1.name}": bounds of clipping dimension "{d1.name}" only overlaps by {int(overlap*100)} %' )
          _logger.warn(f'  stats1 {d1.name} bounds: {d1.bounds}')
          _logger.warn(f'  stats1 {d2.name} bounds: {d2.bounds}')
          _logger.warn('  continuing, but the difference plots may not show the results you expect.')
      else:
        if numpy.sqrt(numpy.mean( (d1.bounds-d2.bounds)**2)) > 1e-5:
          _logger.warn(f'BinnedStatsDiff(): Clipping dimension {d1.name} has different bounds for input stats')
          _logger.warn(f'  stats1={d1.bounds}')
          _logger.warn(f'  stats2={d2.bounds}')

    # calculate dimension masks, in the case of non-overlapping sections of dimensions
    for d1, d2 in zip(stats1.bin_dims, stats2.bin_dims):
      print("TODO")
      pass

  @property
  def variable(self):
    return self.stats1.variable

  @property
  def variables(self):
    return self.stats1.variables

  @property
  def bin_dims(self):
    d1 = self.stats1.bin_dims
    d2 = self.stats2.bin_dims

    # TODO calculate the overlap
    return d1

  @property
  def bin_names(self):
    return self.stats1.bin_names

  @property
  def clip_dims(self):
    # TODO handle overlap issues
    return self.stats1.clip_dims

  @property
  def count(self):
    return self.stats1.count - self.stats2.count

  @property
  def count_qc(self):
    raise NotImplementedError()

  def mean(self, variable):
    return self.stats1.mean(variable) - self.stats2.mean(variable)

  def rmsd(self, variable):
    return self.stats1.rmsd(variable) - self.stats2.rmsd(variable)

  def stddev(self, variable):
    return self.stats1.stddev(variable) - self.stats2.stddev(variable)

  def variance(self, variable):
   return self.stats1.variance(variable) - self.stats2.variance(variable)

  def __str__(self):
    s=f'<BinnedStatsDiff(variable="{self.stats1.variable}", name="{self.stats1.name}", '
    s+=f'dims={[d.name for d in self.stats1.bin_dims]}, clipping={[d.name for d in self.stats1.clip_dims]})>'
    return s
