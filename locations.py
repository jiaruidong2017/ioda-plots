#!/usr/bin/env python3

# (C) Copyright 2019-2020 UCAR
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

import cartopy.crs as ccrs
import datetime
import numpy as np
import pandas as pd
from PIL import Image
import sys
import xarray

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


def plot(iodafiles, output, **kwargs):
  """Generate an animated gif of the locations given in the input IODA files.

  Args:

    iodafiles: a list of one or more IODA formatted files to read in

    output: Output animated gif filename

    **kwargs: optional arguments listed below


  Optional Args in kwargs:

    end_date: end date of plot in YYYY-MM-DDTHH:MM:SSZ format. If not present,
      the date is set from the last given observation

    obs_interval: period, in minutes, between frames of the animation. If not
      present, this is determined based on the target_frames paramter

    obs_window: the length of the window for determining how many frames to show
      for each frame. If not present, this is set to obs_interval * 2

    start_date: start date of plot in YYYY-MM-DDTHH:MM:SSZ format. If not present,
      the start date is set from the first given observation

    target_frames: If obs_interval is not present it will be calculated based on
      the value of target_frames. target_frames is the number of frames that
      should be shown for the animation. The resulting obs_interval will be rounded
      in orderl to be a nice round fraction of hours or days (e.g. 15, 30,
      60 minutes, 2 hours, 6 hours)
  """

  # remove "None" items in optional arg list, to make my life easier
  kwargs = {k: v for k,v in kwargs.items() if v is not None}

  # these will eventually hold the concatenated values
  lats=[]
  lons=[]
  dts=[]

  # read data from each file given
  # TODO, do this in parallel?
  print("Reading ioda files...")
  for f in iodafiles:
    print(f)
    data = xarray.open_dataset(f)
    lats.append( np.array(data['latitude@MetaData']) )
    lons.append( np.array(data['longitude@MetaData']) )
    dt = data['datetime@MetaData'].astype(str)
    dts.append( dt )

  # concatenate into a single dataset
  empty = len(dts)==0
  if not empty:
    lat = np.concatenate(lats)
    lon = np.concatenate(lons)
    dt = np.concatenate(dts)
    dt = pd.to_datetime(dt)
    dt = np.array([d.replace(tzinfo=None) for d in dt])

  # prepare the plot grid
  fig = plt.figure(figsize=(5.0,2.8))
  proj = ccrs.Robinson(central_longitude=-155)
  ax=plt.axes(projection=proj)
  ax.coastlines(color='k', alpha=0.5)
  ax.set_global()
  gl = ax.gridlines(alpha=0.5, color='k', linestyle='--')
  gl.ylocator = mticker.FixedLocator([-90,-60,-30,0,30,60,90])
  gl.xlocator = mticker.FixedLocator([-90, 0, 90, 180, 270])
  fig.tight_layout()

  # special case of no observations, print "no obs", save image, and give up
  if empty:
    print("NO observations to plot")
    frames = 0
    plt.annotate("NO OBSERVATIONS", (0.5,0.5), xycoords='figure fraction',
                 fontweight='bold', ha='center', va='center', size='xx-large')
    fig.canvas.draw()
    w,h=fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf.shape = (w,h,3)
    img = Image.frombytes("RGB", (w,h), buf.tostring())
    img.save(output)
    return

  # plot all locations as gray
  plt.scatter(lon, lat, s=2, transform=ccrs.PlateCarree(), color='lightgray')

  # initial start/end dates
  start_date = kwargs.get('start_date', np.min(dt)).replace(tzinfo=None)
  end_date = kwargs.get('end_date', np.max(dt)).replace(tzinfo=None)

  # calc/get obs window and interval
  target_frames = kwargs.get('target_frames', 40)
  target_obs_intervals=[1, 5, 10, 15, 30, 60, 120, 180, 360, 720, 1440] # in minutes  (1 min to 1 day)
  default_obs_interval=min(target_obs_intervals, key=lambda
                           x:abs(x-((end_date-start_date)/target_frames).total_seconds() // 60))
  default_obs_interval=datetime.timedelta(minutes=default_obs_interval)
  obs_interval = kwargs.get('obs_interval', default_obs_interval)
  obs_window = kwargs.get('obs_window', obs_interval*2)

  # rounding start/end times to intervals of "obs_interval"
  def round_datetime(dt):
    base_day=dt.replace(minute=0, tzinfo=None, second=0, hour=0, microsecond=0)
    seconds=(dt.replace(tzinfo=None)-base_day).total_seconds()
    roundTo = obs_interval.seconds
    rounding = 0 if roundTo == 0.0 else (seconds+roundTo/2) // roundTo * roundTo
    return base_day + datetime.timedelta(seconds=rounding)
  start_date = round_datetime(start_date)
  end_date = round_datetime(end_date)
  frames = int((end_date - start_date).total_seconds() // obs_interval.total_seconds())
  print("start_date  : ", start_date)
  print("end_date    : ", end_date)
  print("obs_interval: ", obs_interval)
  print("obs_window  : ", obs_window)
  print("frames      : ", frames)

  # create annotation postfix
  v, t=(obs_window/2).total_seconds() // 60, 'min'
  if v > 60:
     v, t = v // 60, 'hr'
     if v > 24:
       v, t = v // 24, 'days'
  annotation_pfx = " ±"+str(int(v))+t

  # generate each frame
  palette_idx=0
  images = []
  for frame in range(frames):
    sys.stdout.write(".")
    sys.stdout.flush()

    # what are the beginning and ending times of the window for this frame?
    ds = start_date.replace(tzinfo=None) + obs_interval*frame
    dw1 = ds - obs_window/2
    dw2 = ds + obs_window/2

    # only keep observations in that window
    mask = np.logical_and(dt <= dw2, dt >=dw1 )

    # if there ARE observations in this frame, let this frame serve
    # as the source of the overall GIF palette
    if (np.sum(mask) > 1):
      palette_idx = frame

    # scatter plot, and save
    scatter = plt.scatter(lon[mask], lat[mask], s=2.5, transform=ccrs.PlateCarree(), color='cornflowerblue')
    annotate = ax.annotate(str(ds)+annotation_pfx, xycoords='axes points', xy=(0,-12))

    # save the frame
    fig.canvas.draw()
    w,h=fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf.shape = (w,h,3)
    img = Image.frombytes("RGB", (w,h), buf.tostring())
    images.append(img)

    # cleanup
    scatter.remove()
    annotate.remove()

  print("")

  # optimize by creating limited palette
  # (still not as optimized as "gifsicle -O3")
  palette_size=16
  palette=images[palette_idx].quantize(palette_size, method=2)
  for i, img in enumerate(images):
    images[i] = img.quantize(palette_size, method=2, palette=palette)

  # write out the animated GIF, looping but pausing at first and last frame
  duration = [100 for x in range(len(images))]
  duration[0]=500
  duration[-1]=1000
  images[0].save(output, save_all=True, append_images=images[1:], loop=0, duration=duration)
  print("saved to ",output)
  print("")


if __name__ == "__main__":
  import argparse

  # get command line arguments
  parser = argparse.ArgumentParser(
  description="Generate an animation of observation locations in IODA file(s), saving as an animated GIF")
  parser.add_argument("iodafiles", nargs="*",
                      help="one or more IODA files")
  parser.add_argument("-o", "--output", type=str, default="obs.gif",
                      help="Output filename (default: %(default)s)")

  parser.add_argument("-s", "--start_date", default=None,
                      help="start date/time of animation. Default: first observation in input files. "+
                       "example format: 2020-02-01T01:00:00Z")
  parser.add_argument("-e", "--end_date", default=None,
                      help="end date/time of animation. Default: last observation in input files.")

  parser.add_argument("-t", "--target_frames", type=int, default=30,
                      help="target number of frames to render. The actual target is rounded to convenient fractions "+
                      " of days or hours. Used to calculated obs_interval if it is not given. (default: %(default)s)")
  parser.add_argument("-i", "--obs_interval", type=int, default=None,
                      help="obs time between animation frames in minutes Default: calculated from target_frames")
  parser.add_argument("-w", "--obs_window", type=int, default=None,
                      help="time window of obs to plot for each frame of animation, in minutes. Default: obs_interval*2")

  # parse command line args
  args = parser.parse_args()
  if args.obs_window:
    args.obs_window = datetime.timedelta(minutes=args.obs_window)
  if args.obs_interval:
    args.obs_interval = datetime.timedelta(minutes=args.obs_interval)
  if args.start_date:
    args.start_date = pd.to_datetime(args.start_date)

  # plot!
  plot(**vars(args))
