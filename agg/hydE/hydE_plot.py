'''
Created on Apr. 26, 2022

@author: cefect

visualizetion on raster calcs
'''

#===============================================================================
# imports-----------
#===============================================================================
import os, datetime, math, pickle, copy
start = datetime.datetime.now()
print('start at %s' % start)

import qgis.core
import pandas as pd
import numpy as np
from pandas.testing import assert_index_equal, assert_frame_equal, assert_series_equal
idx = pd.IndexSlice
#===============================================================================
# setup matplotlib
#===============================================================================
 
import matplotlib
matplotlib.use('Qt5Agg') #sets the backend (case sensitive)
matplotlib.set_loglevel("info") #reduce logging level
import matplotlib.pyplot as plt

#set teh styles
plt.style.use('default')

#font
matplotlib_font = {
        'family' : 'Arial',
        'weight' : 'normal',
        'size'   : 8}

matplotlib.rc('font', **matplotlib_font)
matplotlib.rcParams['axes.titlesize'] = 10 
matplotlib.rcParams['axes.labelsize'] = 10

matplotlib.rcParams['figure.titlesize'] = 4
#matplotlib.rcParams['figure.titleweight']='bold'

#spacing parameters
matplotlib.rcParams['figure.autolayout'] = False #use tight layout

#legends
matplotlib.rcParams['legend.title_fontsize'] = 'large'
matplotlib.rcParams['lines.markersize'] = 3.0

print('loaded matplotlib %s'%matplotlib.__version__)

from agg.hydR.hydR_plot import RasterPlotr
from agg.hydE.hydE_scripts import ExpoRun
from hp.plot import Plotr
from hp.gdal import rlay_to_array, getRasterMetadata
from hp.basic import set_info, get_dict_str
from hp.pd import get_bx_multiVal, view
#from hp.animation import capture_images

class ExpoPlotr(RasterPlotr, ExpoRun): #analysis of model results

    
    def __init__(self,
 
                 name='hydE_plot',
                 colorMap_d={
                     
                     },
 
                 **kwargs):
        
        
        colorMap_d.update({
            'aggLevel':'viridis'
                        })
 
        
        super().__init__(name=name,colorMap_d=colorMap_d, **kwargs)
        
 
 
    
    #===========================================================================
    # PLOTRS------
    #===========================================================================
 
def run( #run a basic model configuration
        #=======================================================================
        # #generic
        #=======================================================================
        tag='r0',
        overwrite=True,
        
        #=======================================================================
        # data files
        #=======================================================================
        catalog_fp='',
        
        #=======================================================================
        # parameters
        #=======================================================================
 
        #=======================================================================
        # plot control
        #=======================================================================
        transparent=False,
        
        #=======================================================================
        # debugging
        #=======================================================================
 
        
        **kwargs):
    
    with ExpoPlotr(tag=tag, overwrite=overwrite,  transparent=transparent, plt=plt, 
                        
                       bk_lib = {
                           'catalog':dict(catalog_fp=catalog_fp),
                           
                           },
                 **kwargs) as ses:
        
        #=======================================================================
        # compiling-----
        #=======================================================================
 
        ses.compileAnalysis()
 
        #=======================================================================
        # data prep------
        #=======================================================================
        #change order
        ax_title_d = ses.ax_title_d
        
        #drop some
        #for sa in ['LMFRA', 'noise']: del ax_title_d[sa]
 
        dx_raw = ses.retrieve('catalog').loc[idx[list(ax_title_d.keys()), :], :]
        
        dx1=dx_raw.copy()
        
        #add dummy missing values
        assert not 'rmseD' in dx1.columns.get_level_values('dkey')
        dx1.loc[:, idx['rmseD', 'crs']]=np.nan #so the plot loop works below
        
 
        #dx1 = dx_raw.loc[:, idx[('rsampStats'), :]].droplevel(0, axis=1)
        """
        view(dx1)
        """
        
        #=======================================================================
        # #sub-set filters
        #=======================================================================
        #hi resolution
        hi_bx = np.logical_and(
            dx1.index.get_level_values('aggLevel')<=100,
            dx1.index.get_level_values('resolution')<=1000)
        
        #filter 1
        bx1 = np.logical_and(dx1.index.get_level_values('downSampling')=='Average',
                np.logical_and(dx1.index.get_level_values('dsampStage')=='post',
                np.logical_and(dx1.index.get_level_values('aggLevel')<=1024,
                             dx1.index.get_level_values('studyArea')!='noise')))
        
        #filter 2
        bx2 = np.logical_and(dx1.index.get_level_values('downSampling')=='Average',
                np.logical_and(dx1.index.get_level_values('dsampStage')=='post',
                             dx1.index.get_level_values('studyArea')!='noise'))
        
        #filter 3
        bx3 = np.logical_and(bx2,dx1.index.get_level_values('resolution')<=1000)
                             
                            
 
        
 
        #=======================================================================
        # parameters
        #=======================================================================
        #figsize=(16,16)
        figsize=(7,7)
        #=======================================================================
        # plot loop-------
        #=======================================================================
        #looping different types of plots
        for plotName, dxi, xlims,  ylims,xscale, yscale, drop_zeros in [
            #('full',dx1, None,None, 'log', 'linear', True),
            #('hi',dx1.loc[hi_bx, :], (10, 1001),None, 'log', 'linear', True),
            #('filter1',dx1.loc[bx1, :], (10, 10**3),None, 'log', 'linear', True),
            #('filter2',dx1.loc[bx2, :], (1, 10**4),None, 'log', 'linear', True),
            ('filter3',dx1.loc[bx3, :], (1, 10**4),None, 'log', 'linear', True),
 
 
            ]:
            
            
            #===================================================================
            # multi- Resolution and AggLevel
            #===================================================================
            #paramteer combinations to plot over
            """these are more like 'scenarios'... variables of secondary interest"""
            gcols = [
                'downSampling', 
                #'dsampStage', 
                'aggType']
        
            for gkeys, gdx in dxi.groupby(level=gcols):
                #===============================================================
                # prep
                #===============================================================
                keys_d = dict(zip(gcols, gkeys))
                title = '_'.join([plotName]+list(gkeys))
                print('\n\n%s %s vs. AggLevel\n\n'%(plotName, keys_d))
                
                #prep slices
                dx_expo = gdx.loc[:, idx[('rsampStats', 'rsampErr', 'rstats'), :]].droplevel(0, axis=1)
                
                #filter titles
                at_d = {k:v for k,v in ax_title_d.items() if k in gdx.index.get_level_values('studyArea')}
                #===============================================================
                # vs. aggLevel
                #===============================================================
                col_d={#exposure values
                        #'mean': 'sampled mean (m)',
                        #'max': 'sampled max (m)',
                        #'min': 'sampled min (m)',
                        'wet_mean': 'sampled wet mean (m)',
                        #'wet_max': 'sampled wet max (m)',
                        #'wet_min': 'sampled wet min (m)',
                        'wet_pct': 'wet assets (pct)',
                        'RMSE':'RMSE (m)',
                        
                        }
                
 
                
                ax_d = ses.plot_statVsIter(
                    xvar='aggLevel',xlabel='Aggregation Level',
                    plot_bgrp='resolution',
                    set_ax_title=False,figsize=figsize,
                    dx_raw=dx_expo, 
                    coln_l=list(col_d.keys()), xlims=xlims,ylab_l = list(col_d.values()),
                    title=title + ' vs. Aggregation', ax_title_d=at_d,
                    write=True, legend_kwargs=dict(title='resolution'))
                
 
                #===============================================================
                # vs. Resolution
                #===============================================================
                print('\n\n%s %s vs. Resolution\n\n'%(plotName, keys_d))
                                         
                ax_d = ses.plot_statVsIter(
                    xvar='resolution', plot_bgrp='aggLevel',
                    set_ax_title=False,figsize=figsize,
                    dx_raw=dx_expo, grid=False,
                    coln_l=list(col_d.keys()), xlims=xlims,ylab_l = list(col_d.values()),
                    title=title+'vs. Resolution', ax_title_d=at_d,
                    write=False, )
                
                #===============================================================
                # add raster
                #===============================================================
                """order matters here
                we plot teh new data on whatever axis order is specified above"""
                col_d1={#raster values
                         # 'MAX':'max depth (m)',
                        # 'MIN':'min depth (m)',
                         #'MEAN':'global mean depth (m)',
 
                        'wetMean':'wet mean depth (m)',
                        #'wetVolume':'wet volume (m^3)', 
                        #'wetArea': 'wet area (m^2)', 
                         
                         #'gwArea':'gwArea',
                         #'STD_DEV':'stdev (m)',
                         #'noData_cnt':'noData_cnt',
                         #'noData_pct':'no data (%)',
                         #'rmse':'RMSE (m)', #didnt run diffs
                         'wetPct':'wet cells (pct)',
                         #'rmseD':'blank',
                         'blank':'blank'
                    
                          }
                assert len(col_d1)==len(col_d)
                
                #prep slice
                bx =  gdx.index.get_level_values('aggLevel')==1 #only 1 agg level
                dx_haz=gdx[bx].loc[:, idx[('rstats', 'wetStats', 'noData_pct'), :]].droplevel(0, axis=1)
                
                dx_haz['blank'] = np.nan
                
                dx_haz['wetPct'] = dx_haz['wetCnt']/dx_haz['cell_cnt']
                
                assert dx_haz.columns.is_unique
                """
                gdx.columns
                
                dx_haz.columns
                view(dx_haz)
                """
 
                                       
                plot_colr='dsampStage'
                ses.plot_statVsIter(ax_d=ax_d,
                                    #xvar='resolution', plot_bgrp='aggLevel', 
                                    plot_colr=plot_colr,figsize=figsize,
                                   color_d={k:'black' for k in dx_haz.index.unique(plot_colr)},
                                   marker_d={k:'x' for k in dx_haz.index.unique(plot_colr)},
                                   plot_kwargs = dict(alpha=0.8, linestyle='dashed'),
                                                           
                                    set_ax_title=False,
                                    dx_raw=dx_haz, 
                                    coln_l=list(col_d1.keys()), 
                                    #xlims=xlims,
                                    ylab_l = list(col_d.values()),ax_title_d=at_d,
                                    title=title + ' vs. Resolution',
                                    legend_kwargs=dict(title='aggLevel'))
 
            

        
        out_dir = ses.out_dir
    return out_dir

 
 

def r01():
    return run(tag='r01',catalog_fp=r'C:\LS\10_OUT\2112_Agg\lib\hydE01\hydE01_run_index.csv',)

def r02():
    return run(tag='r02',catalog_fp=r'C:\LS\10_OUT\2112_Agg\lib\hydE02\hydE02_run_index.csv',)

if __name__ == "__main__": 
    #wet mean

    r02()
 

    tdelta = datetime.datetime.now() - start
    print('finished in %s' % (tdelta))
    