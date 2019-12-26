#!/usr/bin/env python3
import collections
import gzip
import numpy
import pickle
import xarray
import yaml
import copy
import dateutil

# TODO add a "merge" command useable from main
# TODO make histogram calls go faster
# TODO handle region subselection
# TODO handle binning extents 
class BinnedStatsCollection:
    def __init__(self):
        self.obsvar = None
        self.bin_config = {}
        self.binned_stats = {}
        self.daterange = []

    @staticmethod
    def load(filename):
        # TODO, load/save are having pickle problems
        cls = BinnedStatsCollection()
        f = gzip.GzipFile(filename, 'rb')
        cls.__dict__.update(pickle.loads(f.read()))
        f.close()
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


class BinnedStats:
    def __init__(self, data, binning_spec):
        self.name = None
        self.obsvar = None
        self.data = {}
        self.bin_dims = None
        self.bin_edges = None

        self.name = binning_spec['name']

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
        self.data['count'] = H

        # counts (qc)
        H, _ = numpy.histogramdd(dim_val_qc, self.bin_edges)
        self.data['count_qc'] = H

        # oman, ombg
        for v in ('oman', 'ombg'):
            val = data[self.obsvar+'@'+v][mask_qc]
            H, _ = numpy.histogramdd(dim_val_qc, self.bin_edges, weights=val)
            self.data[v+'_sum'] = H

            val = val**2
            H, _ = numpy.histogramdd(dim_val_qc, self.bin_edges, weights=val)
            self.data[v+'_sum2'] = H

    def __add__(self, other):
        # TODO check to make sure they are equivalent first
        cls = copy.deepcopy(self)
        for d in cls.data:
            cls.data[d] += other.data[d]
        return cls

    def __str__(self):
        return ('<BinnedStats name="{}" variable="{}" dims="{}">'.format(
            self.name, self.obsvar, self.bin_dims))


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
