'''
Created on Sep. 30, 2022

@author: cefect
'''
import os, pathlib, pprint, webbrowser, logging, sys, pickle
from hp.basic import get_dict_str, now, today_str
from rasterio.crs import CRS
import pandas as pd
import xarray as xr
#from dask.distributed import Client
import dask
import dask.config
from definitions import proj_lib
from hp.pd import view
idx = pd.IndexSlice

from agg2.haz.scripts import UpsampleSessionXR
from agg2.haz.run_stats import xr_lib
from agg2.coms import Agg2DAComs
#===============================================================================
# setup matplotlib----------
#===============================================================================
  
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
    'legend.title_fontsize':'large'
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

class Session(Agg2DAComs, UpsampleSessionXR):
    def __init__(self, scen_name='daXR',**kwargs):
 
 
        super().__init__(scen_name=scen_name, **kwargs)
        
    def plot_hist_xar(self, xar_raw,
                      method='direct',
                      map_d={'row':'scale', 
                             #'col':'method','color':'dsc', 'x':'pixelLength'
                             },
                       matrix_kwargs=dict(
                           #figsize_scaler=3,
                           figsize=(6,12), 
                           set_ax_title=True, add_subfigLabel=True, fig_id=0, constrained_layout=True),
                       bin_kwargs = dict(density=True, bins=100, rwidth=0.99),
                       output_fig_kwargs=dict(),
                         **kwargs):
        """histograms of each raster per scale"""
        #=======================================================================
        # defautlts
        #=======================================================================
        log, tmp_dir, out_dir, ofp, resname, write = self._func_setup('hist_xar_%s'%method,  ext='.svg', **kwargs)
 
        idxn = self.idxn
        xar = xar_raw
        
        keys_all_d = {k:xar[v].values.tolist() for k,v in map_d.items()} #order matters
        
        
        #=======================================================================
        # setup figure
        #=======================================================================
        #plt.close('all')
 
 
        fig, ax_d = self.get_matrix_fig(keys_all_d['row'], ['method'], 
                                    sharey='all',sharex='all',  
                                    logger=log, **matrix_kwargs)
        
        fig.suptitle(method)
        #=======================================================================
        # loop and plot each
        #=======================================================================
        bvals_d = dict()
        stats_lib = dict()
        log.info(f'building plots %i on {xar.shape}'%len(keys_all_d['row']))
        
        
        hrange = (float(xar.min().values), float(xar.max().values))
        for i, (scale, xari) in enumerate(xar.groupby(idxn, squeeze=False)):
            ax = ax_d[scale]['method']
            log.info(f'    on scale={scale}')
            
            #===================================================================
            # data prep
            #===================================================================
            if not scale==1:
                xari_s2 = xari.coarsen(dim={'x':scale, 'y':scale}, boundary='exact').min()
                """
                fig, (ax1, ax2) = plt.subplots(2)
                plt.close('all')
                xari.plot(ax=ax1)
                xari_s2.plot(ax=ax2)
                plt.show()
                """
            else:
                xari_s2 = xari
                
 
            ar = xari_s2.stack(rav=list(xari.coords)).dropna('rav').data.compute()
            
            #===================================================================
            # plot
            #===================================================================
            bvals_d[scale], bins, patches = ax.hist(ar,range=hrange, **bin_kwargs)
            
            #===================================================================
            # mean line
            #===================================================================
            ax.axvline(ar.mean(), color='black', linestyle='dashed')
            
            #===================================================================
            # stats
            #===================================================================
 
            stats_d = {'min':ar.min(), 'max':ar.max(), 'var':ar.var(), 'size':ar.size, 'mean':ar.mean()}
            
            
            ax.text(0.1, 0.8, get_dict_str(stats_d),transform=ax.transAxes, va='top',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", lw=0.0,alpha=0.5 ),
                    )
            
            stats_lib[scale] = pd.Series(stats_d)
            
            del ar, xari
            
        #=======================================================================
        # post format
        #=======================================================================
        for row_key, d in ax_d.items():
            for col_key, ax in d.items():
                ax.grid()
                #first col
                ax.set_ylabel('frequency')
                
                #last row
                if row_key==keys_all_d['row'][-1]:
                    ax.set_xlabel('WSH (m)')
                    
                    
                
            
        #=======================================================================
        # output
        #=======================================================================
        if write:
            #write plot data
            ofpi = os.path.join(out_dir, f'{resname}_bvals.pkl')
            with open(ofpi,  'wb') as f:
                pickle.dump(bvals_d, f, pickle.HIGHEST_PROTOCOL)
            log.info(f'pickle.dump on {len(bvals_d)} bin value arrays to \n    {ofpi}')
            
            #write stats
            df = pd.concat(stats_lib, axis=1).T.rename_axis(idxn).rename_axis('metric', axis=1)
            ofpi = os.path.join(out_dir, f'{resname}_stats_df.pkl')
            df.to_pickle(ofpi)
            log.info(f'wrote stats_df {df.shape} to \n    {ofpi}')
            return self.output_fig(fig, ofp=ofp, logger=log, **output_fig_kwargs)
        else:
            return fig, ax_d
        
        """
        plt.show()
        """
        

def plot_hist(xr_dir_lib, 
                  **kwargs):
    """histograms"""

    
    #===========================================================================
    # get base dir
    #=========================================================================== 
    """combines filter and direct"""
    out_dir = os.path.join(
        pathlib.Path(os.path.dirname(xr_dir_lib['direct'])).parents[1],  # C:/LS/10_OUT/2112_Agg/outs/agg2/r5
        'da', 'haz', today_str)
    
    #===========================================================================
    # execute
    #===========================================================================
    with Session(out_dir=out_dir,logger=logging.getLogger('r'), **kwargs) as ses:
        """
        raw domain clipped is showing a negative bias!
        
        """
 
        idxn = ses.idxn
        log = ses.logger
        
        #=======================================================================
        # loop and build gaussian values on each
        #=======================================================================
 
        #build a datasource from the netcdf files
        d=dict()
        for method, xr_dir in xr_dir_lib.items():
            log.info(f'\n\non {method} from\n    {xr_dir}\n\n')
            #===================================================================
            # load data
            #===================================================================        
            ds = ses.get_ds_merge(xr_dir) 
            dar = ds['wd'].squeeze(drop=True).reset_coords( #drop band
                names=['spatial_ref'], drop=True
                ).transpose(idxn, ...)  #put scale first for iterating
            
            """
            dar.plot(col=idxn)
            dar1.plot(col=idxn)
            """
            
            #data prep
            
            dar1 = dar.where(dar[0]!=0)
            #dar1 = dar
            
            plt.close('all')
            ofp = ses.plot_hist_xar(dar1,method=method)
            
        #=======================================================================
        # wrap
        #=======================================================================
    print('finsihed')

def SJ_plot_hist_run(
        run_name='r10', 
        case_name='SJ',
        **kwargs): 
 
    crs = CRS.from_epsg(proj_lib[case_name]['EPSG'])    
    return plot_hist(xr_lib[run_name], case_name=case_name, run_name=run_name,crs=crs,**kwargs)
    
if __name__ == "__main__": 
    
    start = now()
    
    scheduler='single-threaded'
    #scheduler='threads'
    with dask.config.set(scheduler=scheduler):
        print(scheduler)
        #print(pprint.pformat(dask.config.config, width=30, indent=3, compact=True, sort_dicts =False))
    
 
        SJ_plot_hist_run(run_name='r11')
    
    print('finished in %.2f'%((now()-start).total_seconds())) 