#!/usr/bin/env python3

# (C) Copyright 2020-2020 UCAR
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

import click
import iodaplots
import logging
import traceback

# setup logging
class _AnsiColorStreamHandler(logging.StreamHandler):
  DEFAULT = '\x1b[0m'
  RED     = '\x1b[31m'
  GREEN   = '\x1b[32m'
  YELLOW  = '\x1b[33m'
  CYAN    = '\x1b[36m'

  CRITICAL = RED
  ERROR    = RED
  WARNING  = YELLOW
  INFO     = DEFAULT
  DEBUG    = CYAN

  @classmethod
  def _get_color(cls, level):
    if level >= logging.CRITICAL:  return cls.CRITICAL
    elif level >= logging.ERROR:   return cls.ERROR
    elif level >= logging.WARNING: return cls.WARNING
    elif level >= logging.INFO:    return cls.INFO
    elif level >= logging.DEBUG:   return cls.DEBUG
    else:                          return cls.DEFAULT

  def __init__(self, stream=None):
    logging.StreamHandler.__init__(self, stream)

  def format(self, record):
    text = logging.StreamHandler.format(self, record)
    color = self._get_color(record.levelno)
    return color + text + self.DEFAULT

#logging.basicConfig(level=logging.DEBUG,
logging.basicConfig(level=logging.INFO,
  handlers=[_AnsiColorStreamHandler(),],
  format="%(asctime)s [%(levelname)s]: %(message)s")
_logger = logging.getLogger('iodaplots')


@click.group()
def cli():
  """
  Binning and plotting of JEDI IODA files for generating
  observation space statistics
  """
  pass


@cli.command()
@click.argument('ioda_files', nargs=-1)
@click.option('-c','--config', type=click.File('r'),
              help="configuration yaml file")
@click.option('-o','--output', type=str, required=True,
              help="output binned statistics")
def bin(ioda_files, config, output):
  """
  Perform binning on one or more ioda files
  """
  _logger.info(f"iodaplots: binning {len(ioda_files)} ioda file(s).")
  _logger.debug(f"configuration file: {config}")
  # Read in and merge
  # TODO do this in parallel
  # TODO move this to a function in the library?
  stats = None
  for f in ioda_files:
    if config is not None:
      config.seek(0)
    stats2 = iodaplots.BinnedStatsCollection.from_ioda(f, config)
    if stats is None:
      stats = stats2
    else:
      stats.merge(stats2)

  # save output bin file
  stats.save(output)
  _logger.info(f"binned output saved to {output}")


@cli.command()
@click.argument('bin_files', nargs=-1)
@click.option('-o', '--output', type=str, required=True,
  help="output binned statistics file.")
def cat(bin_files, output):
  """
  Combine several binned stats files along the time dimension
  """
  _logger.info(f'Concatenating {len(bin_files)} files(s).')
  stats=None
  for f in bin_files:
    s = iodaplots.BinnedStatsCollection.load(f)
    if stats is None:
      stats = s
    else:
      stats.cat(s)
  _logger.info(f'saving concatenated file to {output}')
  stats.save(output)


@cli.command()
@click.argument('bin_files', nargs=-1)
@click.option('-o','--output', type=str, required=True,
              help="output binned statistics")
def merge(bin_files, output):
  """  Combine multiple binned files into a single binned file """
  _logger.info(f'merging {len(bin_files)} file(s).')
  stats=None
  for f in bin_files:
    s = iodaplots.BinnedStatsCollection.load(f)
    if stats is None:
      stats = s
    else:
      stats.merge(s)
  stats.save(output)


@cli.command()
@click.option('-e', '--exp', multiple=True, metavar='NAME FILE', required=True, type=(str,str),
              help='for each experiment, an experiment NAME and FILE of the binned '+
              ' statistics. Multiple "--exp" arguments can be passed to show multiple'+
              ' lines on the 1D plots, or 2D plots if "--diff" is used.')
@click.option('-o','--output', type=str, required=True, metavar='FILE',
              help='output FILE template')
@click.option('--diff', is_flag=True,
              help='Plots are the differences from a control experiment.'+
              ' The first "--exp" is considered the control experiment and is '+
              ' subtracted from each subsequent "--exp".'+
              ' This option can only be used if more than 1 "--exp" is given.'+
              ' If more than 2 "--exp" are given, only 1D plots will be made.')
@click.option('--thumbnails', is_flag=True,
              help='Create simpler images suitable for thumbnails.'+
              ' (smaller and with fewer labels)')
def plot(exp, output, **kwargs) :
  """ Using binned files from one or more experiments, generate plots. """

  names, files = zip(*exp)
  iodaplots.plot(files, names, output=output, **kwargs)

if __name__=="__main__":
  cli()
