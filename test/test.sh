#!/bin/bash
set -eu

CDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
WDIR=$CDIR/..

export PYTHONPATH=$CDIR/../src
export PATH=$PYTHONPATH/iodaplots/bin:$PATH
EXE=ioda_plots.py

for b in prod nightly; do
  for p in sst ; do # temp sst adt; do
    echo ""
    echo "soca - $p - $b"

    # binning of each day
    mkdir -p $WDIR/bins/achto/$b/$p
    for d in $WDIR/data/achto/$b/$p/*; do
      d2=${d##*/}
      $EXE bin $d/*.nc -o $WDIR/bins/achto/$b/$p/$d2.nc -c $CDIR/$p.yaml
    done

    # merging of all the days together
    $EXE merge $WDIR/bins/achto/$b/$p/2020*.nc -o $WDIR/bins/achto/$b/$p/merged.nc

    # concatenating all the days together, creating temporal dimension
    $EXE cat $WDIR/bins/achto/$b/$p/2020*.nc -o $WDIR/bins/achto/$b/$p/cat.nc
  done
done

# plots!!!
for p in sst; do #adt temp sst; do
  for t in merged cat; do # merged cat; do
    # normal plots of just one experiment
    mkdir -p $WDIR/plots/achto/prod/$p/$t
    $EXE plot -e prod $WDIR/bins/achto/prod/$p/$t.nc \
              -o $WDIR/plots/achto/prod/$p/$t/ \
              -c $CDIR/$p.yaml

    # plots comparing multiple experiments
    mkdir -p $WDIR/plots/achto/comp/$p/$t
    $EXE plot -e prod $WDIR/bins/achto/prod/$p/$t.nc \
              -e nightly $WDIR/bins/achto/nightly/$p/$t.nc \
              -o $WDIR/plots/achto/comp/$p/$t/ \
              -c $CDIR/$p.yaml

    # plots taking the difference between two experiments
    mkdir -p $WDIR/plots/achto/diff/$p/$t
    $EXE plot -e prod $WDIR/bins/achto/prod/$p/$t.nc \
              -e nightly $WDIR/bins/achto/nightly/$p/$t.nc \
              -o $WDIR/plots/achto/diff/$p/$t/ \
              --diff \
              -c $CDIR/$p.yaml
  done
done
