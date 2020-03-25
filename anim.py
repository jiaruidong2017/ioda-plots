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


interval = datetime.timedelta(minutes=15)
window = datetime.timedelta(minutes=30)
count=24*4

# get command line arguments
parser = argparse.ArgumentParser(description="")
parser.add_argument("iodafiles", nargs="*")
parser.add_argument("--output", type=str, default="obs.gif")
args = parser.parse_args()

# read in the ioda data
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


prev=None

def update(i):
  if empty:
    return
  global prev
  ds = np.min(dt) + ((np.max(dt) - np.min(dt))/2)
  ds = ds.date()
  ds = datetime.datetime.combine(ds, datetime.time(hour=0), tzinfo=datetime.timezone.utc)
  ds += interval*i

  dw1 = ds - window
  dw2 = ds + window
  mask = np.logical_and(dt <= dw2, dt >=dw1 )
  if prev is not None:
    for p in prev:
      p.remove()
  prev=[]
  x = plt.scatter(lon[mask], lat[mask], s=2.5, transform=ccrs.PlateCarree(), color='cornflowerblue')
  prev.append(x)
  x = ax.annotate(ds, xycoords='axes points', xy=(0,-12))
  prev.append(x)
  

fig = plt.figure(figsize=(5.0,2.8))
proj = ccrs.Robinson(central_longitude=-155)
ax=plt.axes(projection=proj)
ax.coastlines(color='k', alpha=0.5)
ax.set_global()
gl = ax.gridlines(alpha=0.5, color='k', linestyle='--')
gl.ylocator = mticker.FixedLocator([-90,-60,-30,0,30,60,90])
gl.xlocator = mticker.FixedLocator([-90, 0, 90, 180, 270])
fig.tight_layout()


if empty:
  count = 1
  plt.annotate("NO OBSERVATIONS", (0.5,0.5), xycoords='figure fraction', 
    fontweight='bold', ha='center', va='center', size='xx-large')
else:
  s=plt.scatter(lon,lat, s=2, transform=ccrs.PlateCarree())
  plt.scatter(lon, lat, s=2, transform=ccrs.PlateCarree(), color='lightgray')

anim = FuncAnimation(fig, update, frames=np.arange(0, count), interval=100, repeat_delay=2000)  
anim.save(args.output, writer='imagemagick')
plt.close()
 
  

def get_valid_obsvars(data):
    # get the list of variables, and columns for each variable
    all_obsvars = collections.defaultdict(set)
    for v in data.variables.keys():
        token = v.split('@')
        all_obsvars[token[0]].add(token[1])

    # check which variables have the required columns
    valid_obsvars = set()
    required_cols = ('ObsError', 'ObsValue')
    for v in all_obsvars:
        if all_obsvars[v].issuperset(required_cols):
            valid_obsvars.add(v)
    return list(valid_obsvars)

    
