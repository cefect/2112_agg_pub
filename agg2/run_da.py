'''
Created on Sep. 26, 2022

@author: cefect
'''
import os, pathlib, itertools, logging, sys
import pandas as pd
import numpy as np
idx = pd.IndexSlice
from hp.basic import get_dict_str, today_str, lib_iter
from hp.pd import append_levels, view
 
from definitions import proj_lib
from pyproj.crs import CRS
from agg2.da import CombinedDASession as Session
from agg2.coms import log_dxcol
from agg2.haz.run import res_fp_lib as hrfp_lib
#===============================================================================
# setup matplotlib----------
#===============================================================================
output_format='pdf'
usetex=True
if usetex:
    os.environ['PATH'] += R";C:\Users\cefect\AppData\Local\Programs\MiKTeX\miktex\bin\x64"

cm = 1/2.54
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
# setup logger----
#===============================================================================
logging.basicConfig(
                #filename='xCurve.log', #basicConfig can only do file or stream
                force=True, #overwrite root handlers
                stream=sys.stdout, #send to stdout (supports colors)
                level=logging.INFO, #lowest level to display
                )

#===============================================================================
# globals
#===============================================================================
res_fp_lib = {
    'r11':{
        'haz':r'C:\LS\10_IO\2112_Agg\outs\agg2\r11\SJ\da\haz\20221013\SJ_r11_direct_1013_dprep.pkl',
        'exp':r'C:\LS\10_IO\2112_Agg\outs\agg2\r11\da\20221013\bstats\SJ_r11_expo_da_1013_bstats.pkl'
        
        },
    
    'r10':
              {
            'haz': r'C:\LS\10_IO\2112_Agg\outs\agg2\r10\SJ\da\haz\20220926\SJ_r10_direct_0926_dprep.pkl',
            #'exp':r'C:\LS\10_IO\2112_Agg\outs\agg2\r8\da\20220926\bstats\SJ_r8_expo_da_0926_bstats.pkl'
            'exp':r'C:\LS\10_IO\2112_Agg\outs\agg2\r8\da\20220927\bstats\SJ_r8_expo_da_0927_bstats.pkl'
            }
              }

 

def SJ_da_run(
        run_name='r10',
        case_name='SJ',
        **kwargs):    
    
    proj_d = proj_lib[case_name] 
    crs = CRS.from_epsg(proj_d['EPSG'])
    
    return run_plots_combine(res_fp_lib[run_name], 
                             xr_dir = hrfp_lib[run_name]['direct']['aggXR'],
                             #xr_dir='C:\\LS\\10_IO\\2112_Agg\\outs\\agg2\\t\\SJ\\direct\\20221013\\_xr', 
                             case_name=case_name, run_name=run_name,crs=crs, **kwargs)



 
    
def run_plots_combine(fp_lib,pick_fp=None,xr_dir=None, write=True,**kwargs):
    """da and figures which combine hazard and exposure    
    
    """
 
    #===========================================================================
    # get base dir
    #=========================================================================== 
    out_dir = os.path.join(pathlib.Path(os.path.dirname(fp_lib['exp'])).parents[1],'da', 'expo',today_str)
    print('out_dir:   %s'%out_dir)
    #===========================================================================
    # execute
    #===========================================================================
    with Session(out_dir=out_dir, write=write,logger=logging.getLogger('r'), **kwargs) as ses:
        log = ses.logger 
        
        #combined stats
        if pick_fp is None:
            dx = ses.build_combined(fp_lib)
        else:
            dx = pd.read_pickle(pick_fp)
            
        #raster data
        if not xr_dir is None:
            xds = ses.get_ds_merge(xr_dir)

        #=======================================================================
        # plots-------
        #=======================================================================
        #Figure 6. Bias from aggregation of four metrics
        #ses.plot_4x4_subfigs(dx1)
 
        
        #=======================================================================
        # #Figure 5. Resample case classification progression
        #=======================================================================
        """asset exposed count"""
        #data prep 
        dx1 = dx.loc[:, idx[:, 's2', 'direct', 'catMosaic', :, 'count']].droplevel((1, 2, 3, 5), axis=1
                               ).drop(1).fillna(0.0).astype(int).loc[:, idx[('haz', 'exp'), :]] 
                               
        
        xar = xds['catMosaic'].squeeze(drop=True).transpose(ses.idxn, ...)[1:] #drop the first
        #plot
        ses.plot_3xRscProg(dx1, xar)
        """
        view(dx1.loc[:, idx[:, 's2',:,:,:,:]].T)
        """

        

if __name__ == "__main__":
    SJ_da_run(run_name='r11', output_format=output_format)
 
    
    print('finished')
        