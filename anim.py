#!/usr/bin/env python3
import collections
import xarray
import numpy as np
import dateutil
import pandas as pd
import datetime
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
from matplotlib.animation import FuncAnimation


# get command line arguments
parser = argparse.ArgumentParser(description="Generate an animation of observation locations.")
parser.add_argument("iodafiles", nargs="*")
parser.add_argument("--output","-o", type=str, default="obs.gif",
  help="Output filename (default: %(default)s)")
parser.add_argument("--obs_window", type=float, default=30,
  help="time window of obs to plot for each frame of animation, in minutes (default: %(default)s)")
parser.add_argument("--obs_interval", type=float, default=15,
  help="obs time between animation frames in minutes (default: %(default)s)")


# parse command line args
args = parser.parse_args()
args.obs_window = datetime.timedelta(minutes=args.obs_window)
args.obs_interval = datetime.timedelta(minutes=args.obs_interval)


# read in the ioda data
print("Reading ioda files...")
lats=[]
lons=[]
dts=[]

for f in args.iodafiles:
  print(f)
  data = xarray.open_dataset(f)
  lats.append( np.array(data['latitude@MetaData']) )
  lons.append( np.array(data['longitude@MetaData']) )
  dt = data['datetime@MetaData'].astype(str)
  dts.append( dt )

empty=len(args.iodafiles)==0
if not empty:
  lat=np.concatenate(lats)
  lon=np.concatenate(lons)
  dt=np.concatenate(dts)
  dt = pd.to_datetime(dt)


# determine start/end times and number of frames
# rounding to intervals of "obs_interval"
if not empty:
  start_time = np.min(dt) - args.obs_window
  stop_time = np.max(dt) + args.obs_window
  def round_datetime(dt):
    base_day=dt.replace(minute=0, tzinfo=None, second=0, hour=0, microsecond=0)
    seconds=(dt.replace(tzinfo=None)-base_day).seconds
    roundTo = args.obs_interval.seconds
    rounding = (seconds+roundTo/2) // roundTo * roundTo
    return base_day + datetime.timedelta(seconds=rounding)
  start_time = round_datetime(start_time)
  stop_time = round_datetime(stop_time)
  frames = (stop_time - start_time).seconds // args.obs_interval.seconds
  print("Plotting start/stop datetimes: ",start_time, stop_time)
  print("frames: ", frames)


# plot gray circles for ALL the observations
print("generating frames...")
fig = plt.figure(figsize=(5.0,2.8))
proj = ccrs.Robinson(central_longitude=-155)
ax=plt.axes(projection=proj)
ax.coastlines(color='k', alpha=0.5)
ax.set_global()
gl = ax.gridlines(alpha=0.5, color='k', linestyle='--')
gl.ylocator = mticker.FixedLocator([-90,-60,-30,0,30,60,90])
gl.xlocator = mticker.FixedLocator([-90, 0, 90, 180, 270])
fig.tight_layout()


# special case of no observations, so that things dont break
if empty:
  frames = 1
  plt.annotate("NO OBSERVATIONS", (0.5,0.5), xycoords='figure fraction',
    fontweight='bold', ha='center', va='center', size='xx-large')
else:
  s=plt.scatter(lon,lat, s=2, transform=ccrs.PlateCarree())
  plt.scatter(lon, lat, s=2, transform=ccrs.PlateCarree(), color='lightgray')

# plot all the frames in the animation
prev=None
def update(i):
  if empty:
    return

  global prev

  # what are the beginning and ending times of the window for this frame?
  ds = start_time.replace(tzinfo=datetime.timezone.utc) + args.obs_interval*i
  dw1 = ds - args.obs_window
  dw2 = ds + args.obs_window

  # only keep observations in that window
  mask = np.logical_and(dt <= dw2, dt >=dw1 )

  # if this is not the first animation frame, remove the plotted elements from the previous frame
  if prev is not None:
    for p in prev:
      p.remove()
  prev=[]

  # scatter plot, saving the handles for use by the next frame
  x = plt.scatter(lon[mask], lat[mask], s=2.5, transform=ccrs.PlateCarree(), color='cornflowerblue')
  prev.append(x)
  x = ax.annotate(ds, xycoords='axes points', xy=(0,-12))
  prev.append(x)


# start generating frames and save the animated gif (hmm, repeat_delay doesnt seem to work)
anim = FuncAnimation(fig, update, frames=np.arange(0, frames), interval=100, repeat_delay=2000)
anim.save(args.output, writer='imagemagick')
plt.close()

