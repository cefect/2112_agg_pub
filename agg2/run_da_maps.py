'''
Created on Oct. 14, 2022

@author: cefect
'''


import os, pathlib, math, pprint, logging, sys
from definitions import proj_lib

import pandas as pd
idx = pd.IndexSlice
from pyproj.crs import CRS
import geopandas as gpd
import shapely.geometry as sgeo
 
from agg2.da import CombinedDASession as Session
from agg2.haz.da import cat_mdex
from agg2.coms import log_dxcol
from agg2.haz.run import res_fp_lib

from hp.pd import view
from hp.basic import get_dict_str, today_str

#===============================================================================
# setup matplotlib----------
#===============================================================================
cm = 1/2.54

output_format='pdf'
usetex=True
if usetex:
    os.environ['PATH'] += R";C:\Users\cefect\AppData\Local\Programs\MiKTeX\miktex\bin\x64"
    
  
import matplotlib
#matplotlib.use('Qt5Agg') #sets the backend (case sensitive)
matplotlib.set_loglevel("info") #reduce logging level
import matplotlib.pyplot as plt
 
#set teh styles
plt.style.use('default')
 
#font
matplotlib.rc('font', **{
        'family' : 'serif',
        'weight' : 'normal',
        'size'   : 8})
 
 
for k,v in {
    'axes.titlesize':10,
    'axes.labelsize':10,
    'xtick.labelsize':8,
    'ytick.labelsize':8,
    'figure.titlesize':12,
    'figure.autolayout':False,
    'figure.figsize':(10,10),
    'legend.title_fontsize':'large',
    'text.usetex':usetex,
    }.items():
        matplotlib.rcParams[k] = v
  
print('loaded matplotlib %s'%matplotlib.__version__)


#===============================================================================
# setup logger
#===============================================================================
logging.basicConfig(
                #filename='xCurve.log', #basicConfig can only do file or stream
                force=True, #overwrite root handlers
                stream=sys.stdout, #send to stdout (supports colors)
                level=logging.INFO, #lowest level to display
                )


def SJ_run(
        run_name='r11',
        method='direct',
        dsName='diffXR',
        case_name='SJ',
        bbox=sgeo.box(2491392.0, 7437314.0, 
                      2491392.0+2**9, 7437314.0+2**9),
 
        **kwargs):    
        
        
    proj_d = proj_lib[case_name] 
 
    crs = CRS.from_epsg(proj_d['EPSG'])
    
    return run_grid_plots(res_fp_lib[run_name][method][dsName],
                           proj_d['finv_fp'],
                          case_name=case_name, run_name=f'{run_name}',
                          scen_name=f'{method}_{dsName}',
                          crs=crs,bbox=bbox,
                           **kwargs)


def run_grid_plots(xr_dir,finv_fp, **kwargs):
    #===========================================================================
    # get base dir
    #=========================================================================== 
    """single method"""
    out_dir = os.path.join(
        #pathlib.Path(os.path.dirname(xr_dir)).parents[0],  # C:/LS/10_IO/2112_Agg/outs/agg2/r5
        os.path.dirname(xr_dir),
        'da', today_str)
    
    
    #===========================================================================
    # execute
    #===========================================================================
    with Session(out_dir=out_dir, logger=logging.getLogger('r'),output_format=output_format,**kwargs) as ses:
        """for haz, working with aggregated zonal stats.
            these are computed on:
                aggregated (s2) data with UpsampleSession.run_stats()
                raw/fine (s1) data with UpsampleSession.run_stats_fine()
                local diff (s2-s1) with UpsampleSession.run_diff_stats()
            
 
        """
        idxn = ses.idxn
        log = ses.logger
        bbox = ses.bbox
 
        #log.info(ses.out_dir)
        #=======================================================================
        # DIFFERNCE GRIDS---------
        #=======================================================================
        #=======================================================================
        # load data
        #=======================================================================
        #grids
        ds = ses.get_ds_merge(xr_dir)        
        xar = ds['wd'].squeeze(drop=True).transpose(ses.idxn, ...)[1:] #drop the first
        
        #clip
        if not bbox is None:
            """this will fail on dev"""
            xar1 = xar.rio.clip_box(*bbox.bounds)
        else:
            xar1 = xar
        
 
        
        gdf_raw = gpd.read_file(finv_fp, bbox=sgeo.box(*xar1.rio.bounds())).rename_axis('fid')
        

        
        """
        gdf_raw.plot()
        """
        assert len(gdf_raw)>0
        gdf_c = gdf_raw.centroid
 
        #=======================================================================
        # plots
        #=======================================================================
        ses.plot_maps(xar1, gdf_c)
        
        #ds['wd'].squeeze(drop=True).plot.imshow(col='scale')
        
        
        
        
        
        
        
        
        
if __name__ == "__main__":
    
    SJ_run(run_name='r11')
    print('finished')