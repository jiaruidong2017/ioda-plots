#!/usr/bin/env python3
import collections
import gzip
import numpy
import pickle
import xarray
import yaml
import copy
import dateutil

class BinnedStatsCollectionDiff:
    def __init__(self, collection1, collection2):
        # ensure the two collections are congruent
        assert collection1.obsvar == collection2.obsvar
        assert collection1.bin_config == collection2.bin_config
        self.obsvar = collection1.obsvar
        self.bin_config = collection1.bin_config
        self.binned_stats = {}
        self.exps=2

        self._collection1 = collection1
        self._collection2 = collection2

        # TODO warning if the date ranges are different?
        self.daterange = collection1.daterange

        for k in collection1.binned_stats:
            self.binned_stats[k] = collection1.binned_stats[k] - collection2.binned_stats[k]

    def __str__(self):
        return ('<BinnedStatsCollectionDiff exp="({})" variable="{}" bins="{}" dates="{} to {}">'.format(
            self.exp(), self.obsvar, len(self.binned_stats),
            self.daterange[0].strftime("%Y%m%dT%X"), self.daterange[1].strftime("%Y%m%dT%X")))

    def exp(self):
        return "{}-{}".format(self._collection1.exp(), self._collection2.exp())


class BinnedStatsCollectionTimeseries:
    def __init__(self, stats):
        # TODO ensure the two collections are congruent
        self.obsvar = stats[0].obsvar
        self.bin_config = stats[0].bin_config
        self.binned_stats = {}        

        # WRONG        
        self.daterange = stats[0].daterange
        for k in stats[0].binned_stats:
            self.binned_stats[k] = BinnedStatsTimeseries( [stats[i].binned_stats[k] for i in range(len(stats))])

    def __str__(self):
        return ('<BinnedStatsCollectionTimeseries exp="({})" variable="{}" bins="{}" dates="{} to {}">'.format(
            self.exp(), self.obsvar, len(self.binned_stats),
            self.daterange[0].strftime("%Y%m%dT%X"), self.daterange[1].strftime("%Y%m%dT%X")))

    def exp(self):
        return "TBI"


# TODO make histogram calls go faster
# TODO handle region subselection
# TODO handle binning extents 
class BinnedStatsCollection:
    def __init__(self):
        self.exps = 1
        self._exp = ""
        self.obsvar = None
        self.bin_config = {}
        self.binned_stats = {}
        self.daterange = []

    def exp(self):
        return self._exp

    @staticmethod
    def load(filename, exp):
        # TODO, load/save are having pickle problems
        cls = BinnedStatsCollection()
        f = gzip.GzipFile(filename, 'rb')
        cls.__dict__.update(pickle.loads(f.read()))
        f.close()
        cls._exp = exp
        return cls

    def save(self, filename):
        print('writing binned stats to: ', filename)
        f = gzip.GzipFile(filename, 'wb')
        f.write(pickle.dumps(self.__dict__, 1))
        f.close()

    @staticmethod
    def create(ioda_file, yaml_file):
        print('Reading: ', ioda_file)
        cls = BinnedStatsCollection()

        # read in the ioda data
        ioda_data = xarray.open_dataset(ioda_file)
        cls.obsvar = get_valid_obsvars(ioda_data)
        assert len(cls.obsvar) == 1
        cls.obsvar = cls.obsvar[0]

        # save the min/max datetimes
        d = ioda_data['datetime@MetaData']
        cls.daterange = [dateutil.parser.parse(str(s.data.astype(str))) for s in
             [numpy.min(d), numpy.max(d)] ]
        
        # read the yaml file
        with open(yaml_file, 'r') as stream:
            cls.bin_config = yaml.safe_load(stream)

        # for each binning type, do the binning
        for binning_spec in cls.bin_config['binning']:
            bs = BinnedStats(ioda_data, binning_spec)
            if bs.obsvar is None:
                continue
            cls.binned_stats[bs.name] = bs
        return cls

    @staticmethod
    def timeseries(stats):
        return BinnedStatsCollectionTimeseries(stats)

    @staticmethod
    def merge(stats):
        cls = copy.deepcopy(stats[0])
        # make sure each input field has same properties
        for s in stats:
            assert s.obsvar == cls.obsvar
            assert s.bin_config == cls.bin_config
        # TODO check to make sure the BinnedStats are equivalent in each input
        cls.daterange[0] = numpy.min([s.daterange[0] for s in stats])
        cls.daterange[1] = numpy.max([s.daterange[1] for s in stats])
        for s in stats[1:]:
            for b in s.binned_stats:
                cls.binned_stats[b] += s.binned_stats[b]
        return cls

    def __add__(self, other):
        return BinnedStatsCollection.merge((self, other))

    def __sub__(self, other):
        return BinnedStatsCollectionDiff(self, other)

    def __str__(self):
        return ('<BinnedStatsCollection exp="{}" variable="{}" bins="{}" dates="{} to {}">'.format(
            self.exp(), self.obsvar, len(self.binned_stats), self.daterange[0].strftime("%Y%m%dT%X"), self.daterange[1].strftime("%Y%m%dT%X")))


class BinnedStatsTimeseries:
    def __init__(self, series):
        self.name = series[0].name
        self.obsvar = series[0].obsvar
        self.bin_dims = series[0].bin_dims +('time',)
        self._series = series

        # TODO use the correct datetimes
        self.bin_edges = series[0].bin_edges
        self.bin_edges += ( list(range(len(series)+1)), )

    def __str__(self):
        return ('<BinnedStatsTimeseries name="{}" variable="{}" dims="{}">'.format(
            self.name, self.obsvar, self.bin_dims))

    def count(self, qc=False):
        ar=[]
        for s in self._series:
            ar.append(s.count(qc))
        return numpy.array(ar)
    
    def rmsd(self, mode):
        ar=[]
        for s in self._series:
            ar.append(s.rmsd(mode))
        return numpy.array(ar)

    def mean(self, mode):
        ar=[]
        for s in self._series:
            ar.append(s.mean(mode))
        return numpy.array(ar)

class BinnedStatsDiff:
    def __init__(self, stats1, stats2):
        self.name = stats1.name
        self.bin_dims = stats1.bin_dims
        self.bin_edges = stats1.bin_edges
        self.obsvar = stats1.obsvar
        self._stats1 = stats1
        self._stats2 = stats2

    def exps(self):
        return 2

    def count(self, qc=False):
        return numpy.minimum(self._stats1.count(qc), self._stats2.count(qc))

    def rmsd(self, mode):
        return self._stats1.rmsd(mode)-self._stats2.rmsd(mode)

    def mean(self, mode):
        return self._stats1.mean(mode)-self._stats2.mean(mode)

    def __str__(self):
        return ('<BinnedStatsDiff name="{}" variable="{}" dims="{}">'.format(
            self.name, self.obsvar, self.bin_dims))


class BinnedStats:
    ''' stats for a single timeslice within arbitrary dimensions'''

    def __init__(self, data, binning_spec):
        self.obsvar = None        
        self.bin_dims = None
        self.bin_edges = None
        self.name = binning_spec['name']        
        self._data = {}

        # Get the list of variables this binning should operate on
        # and abort early if no binning will be done with these vars
        self.obsvar = get_valid_obsvars(data)
        assert len(self.obsvar) == 1
        self.obsvar = self.obsvar[0]
        if 'vars' in binning_spec:
            if self.obsvar not in binning_spec['vars']:
                self.obsvar = None
        if self.obsvar is None:
            return

        # setup the bins for each dimension
        if 'dims' in binning_spec:
            self.bin_dims, self.bin_edges = gen_bins(binning_spec['dims'])

        assert self.bin_dims is not None

        # get the coordinate values for each dimension
        dim_val = []
        for d in self.bin_dims:
            assert (d+'@MetaData') in data.variables.keys()
            dv = data[d+'@MetaData']
            if d == 'longitude':
                dv[dv < 0] += 360.0
            dim_val.append(dv)

        # calculate qc masks
        mask_qc = data[self.obsvar+'@EffectiveQC0'] == 0
        dim_val_qc = []
        for d in dim_val:
            dim_val_qc.append(d[mask_qc])

        # counts
        H, _ = numpy.histogramdd(dim_val, self.bin_edges)
        self._data['count'] = H

        # counts (qc)
        H, _ = numpy.histogramdd(dim_val_qc, self.bin_edges)
        self._data['count_qc'] = H

        # oman, ombg
        for v in ('oman', 'ombg'):
            val = data[self.obsvar+'@'+v][mask_qc]
            H, _ = numpy.histogramdd(dim_val_qc, self.bin_edges, weights=val)
            self._data[v+'_sum'] = H

            val = val**2
            H, _ = numpy.histogramdd(dim_val_qc, self.bin_edges, weights=val)
            self._data[v+'_sum2'] = H

    def __add__(self, other):
        # TODO check to make sure they are equivalent first      
        cls = copy.deepcopy(self)
        for d in cls._data:
            cls._data[d] += other._data[d]
        return cls
    
    def __sub__(self, other):
        return BinnedStatsDiff(self, other)

    def __str__(self):
        return ('<BinnedStats name="{}" variable="{}" dims="{}">'.format(
            self.name, self.obsvar, self.bin_dims))

    def count(self, qc=False):
        if qc==False:
            return self._data['count']
        else:
            return self._data['count_qc']

    def rmsd(self, mode):
        assert mode in ('ombg','oman')
        d = self._data[mode+'_sum2']
        count_qc = self.count(qc=True)
        d[count_qc > 0] /= count_qc[count_qc > 0]
        d = numpy.sqrt(d)
        d = numpy.ma.masked_where(count_qc == 0, d)
        return d

    def mean(self, mode):
        assert mode in ('ombg','oman')
        d = self._data[mode +'_sum']
        count_qc = self.count(qc=True)
        d[count_qc > 0] /= count_qc[count_qc > 0]
        d = numpy.ma.masked_where(count_qc == 0, d)
        return d
    
    def exps(self):
        return 1



class Region:
    def __init__(self, name):
        self.name = name


def gen_bins(bin_spec):
    bin_dim = ()
    bin_edges = ()

    for v, k in bin_spec.items():
        edges = None

        # make sure either res/bins specified, but not both
        if len(set(k.keys()).intersection(set(('bins', 'res')))) != 1:
            raise Exception("error in bin specification: each dimension must "
                            + " contain either 'res' or 'bins', but not both")
        if 'bins' in k:
            edges = numpy.array(k['bins'])
        elif 'res' in k:
            res = k['res']

            # set default bounds
            bounds = None
            if v == 'latitude':
                bounds = (-90, 90)
            elif v == 'longitude':
                bounds = (0, 360)

            # overwrite with anything given in the config file
            if 'bounds' in k:
                bounds = k['bounds']

            # check
            if bounds is None:
                raise Exception("Bounds need to be given for non lat/lon "
                                + "dimensions if 'res' is specified")

            edges = numpy.arange(bounds[0], bounds[1] + res/2.0, res)

        assert v not in bin_dim, "A duplicate dimension was specified."
        bin_dim += (v,)

        assert(edges is not None)
        bin_edges += (edges, )

    return bin_dim, bin_edges


def get_valid_obsvars(data):
    # get the list of variables, and columns for each variable
    all_obsvars = collections.defaultdict(set)
    for v in data.variables.keys():
        token = v.split('@')
        all_obsvars[token[0]].add(token[1])

    # check which variables have the required columns
    valid_obsvars = set()
    required_cols = ('oman', 'ombg', 'EffectiveQC0', 'ObsValue')
    for v in all_obsvars:
        if all_obsvars[v].issuperset(required_cols):
            valid_obsvars.add(v)
    return list(valid_obsvars)


def _threaded_read_proc(args):
    ioda_file, other_args = args
    yaml_file = other_args.binning
    data = BinnedStatsCollection.create(ioda_file, yaml_file)
    return data


def main():
    import argparse
    import multiprocessing

    # get command line arguments
    parser = argparse.ArgumentParser(
        description="")
    parser.add_argument("iodafiles", nargs="+",
                        help="One or more IODA formatted files to process")
    parser.add_argument('-b', '--binning', type=str, default="binning.yaml",
                        help="path to the yaml file specifying the binning "
                             + " specification  (Default: %(default)s)")
    parser.add_argument("--output", required=True, type=str,
                        help="name of the output file")
    parser.add_argument("--threads", default=multiprocessing.cpu_count(),
                        type=int,
                        help="number of threads to use (Default: %(default)s)")
    args = parser.parse_args()

    # process each file and merge the stats
    pool = multiprocessing.Pool(args.threads)
    paramIn = []
    for f in args.iodafiles:
        paramIn.append((f, args))
    data = pool.map(_threaded_read_proc, paramIn)
    data = BinnedStatsCollection.merge(data)
    data.save(args.output)


if __name__ == "__main__":
    main()
