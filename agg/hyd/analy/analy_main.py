'''
Created on Feb. 21, 2022

@author: cefect

analysis on hyd.model outputs
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
        'family' : 'serif',
        'weight' : 'normal',
        'size'   : 8}

matplotlib.rc('font', **matplotlib_font)
matplotlib.rcParams['axes.titlesize'] = 10 #set the figure title size
matplotlib.rcParams['figure.titlesize'] = 16
matplotlib.rcParams['figure.titleweight']='bold'

#spacing parameters
matplotlib.rcParams['figure.autolayout'] = False #use tight layout

#legends
matplotlib.rcParams['legend.title_fontsize'] = 'large'

print('loaded matplotlib %s'%matplotlib.__version__)


from agg.hyd.analy.analy_scripts import ModelAnalysis
from hp.err_calc import ErrorCalcs

#===============================================================================
# meta funcs------
#===============================================================================

def meta_all( #get all the stats
        meta_d={}, #inherited meta
        pred_ser=None,
        true_ser=None,
        logger=None,
        
        ):
    #error calcs
    """would be nice to move this up for the other plot_types?
        need to sort out iters...
        doesnt handle groups"""
    
    #init the worker
    eW = ErrorCalcs(logger=logger,pred_ser=pred_ser,true_ser=true_ser)

 
    #calc everything
    meta_d.update(eW.get_all(dkeys_l = ['bias', 'meanError', 'meanErrorAbs', 'RMSE', 'pearson']))
    
    #confusion
    _, cm_dx = eW.get_confusion()
    meta_d.update(cm_dx.droplevel(['pred', 'true']).iloc[:,0].to_dict())
    
    #prediction basic stats 
    meta_d.update(eW.get_stats(stats_l = ['min', 'mean', 'max']))
    
    return meta_d

def meta_slim( #get all the stats
        meta_d={}, #inherited meta
        **kwargs):
 
    
    #init the worker
    eW = ErrorCalcs(**kwargs)
 
    #calc everything
    meta_d.update(eW.get_all(dkeys_l = ['bias', 'meanError', 'meanErrorAbs', 'RMSE', 'pearson']))
    
 
    
    return meta_d

def meta_basic(meta_d={}, pred_ser=None, logger=None):
    assert isinstance(pred_ser, pd.Series)
    stats_l = ['min', 'mean', 'max', 'skew', 'count']
    return {**meta_d, **{stat:getattr(pred_ser, stat)() for stat in stats_l}}
    
#===============================================================================
# runners--------
#===============================================================================
def run( #run a basic model configuration
        #=======================================================================
        # #generic
        #=======================================================================
        tag='r2',
        overwrite=True,
 
        #=======================================================================
        # #data
        #=======================================================================
        catalog_fp = None,
        modelID_l=None,
        baseID_l=[0, 40, 50, 80], #models representing the base run (for building Trues)
 
        #=======================================================================
        # plot control
        #=======================================================================
        transparent=False,
        
        #=======================================================================
        # debugging
        #=======================================================================
 
        
        **kwargs):
    
    with ModelAnalysis(tag=tag, overwrite=overwrite,  transparent=transparent, plt=plt, 
                       catalog_fp=catalog_fp,
 
                       modelID_l=modelID_l,
                       bk_lib = {
                           'outs':dict(),
                           'finv_agg_fps':dict(),
                           'trues':dict(baseID_l=baseID_l),
                           
                           },
                 **kwargs) as ses:
        
        #=======================================================================
        # compiling-----
        #=======================================================================
        #ses.runCompileSuite()
        
        mids = list(range(3))+list(range(60,66))
        #ses.write_suite_smry(baseID=0,modelID_l=mids),# + list(range(60,66)),)
        
        #=======================================================================
        # individual model summaries---------
        #=======================================================================
        mids = list(range(63,66))
        #ses.write_resvlay(dkey='rsamps', modelID_l=mids)
        #=======================================================================
        # for mid in mids:
        #     ses.plot_model_smry(mid)
        #=======================================================================
 

            
        #=======================================================================
        # hazard data----------
        #=======================================================================
        """moved all this to intersection"""
 
        
 
        #=======================================================================
        # Intersection-----------
        #=======================================================================
        dkey='rsamps'
        
        #=======================================================================
        # #hazard vs asset resolution (depths)
        #=======================================================================
        #simple
 
        #ses.plot_dkey_mat(dkey='rsamps', modelID_l=list(range(9)), plot_rown='aggLevel', plot_coln='resolution',  fmt='svg',sharex='all',sharey='all', plot_colr='aggLevel')
        
        #main matrix plots (of depth)
        for plotName, mids, baseID in [
            #('pre',         list(range(9)),     0), #base
            #('depth convexHull',       [0, 21,22,3,34,35,6,36,37,], 0),
                             #
                             
            #('centroid',    list(range(50,59)), 50), #sgType='centroids'
            ('wse convexHull',  list(range(3))+list(range(60,66)), 0),
            ]: 
            print('\n%s + %s\n'%(dkey, plotName))
            pass
            #===================================================================
            # errors
            #===================================================================
            #hist2d (aggLevel=0)
             
            #===================================================================
            # ses.plot_err_mat(dkey=dkey, modelID_l=mids, baseID=baseID,
            #                  plot_rown='studyArea', plot_coln='resolution', sharex='all',sharey='all',
            #                      title='%s \'%s\' errors'%(plotName, dkey),
            #                      plot_type='hist2d',
            #                      bins=50, 
            #                      vmin=0.0, vmax=0.5, xlims=(0,6),
            #                      slice_d={'aggLevel':0}, meta_txt=True,
            #                      meta_func = lambda **kwargs:meta_slim(**kwargs))
            #===================================================================
              
            #hist2d (resolution=0) 

            #===================================================================
            # ses.plot_err_mat(dkey=dkey, modelID_l=mids, baseID=baseID,
            #                  plot_rown='studyArea', plot_coln='aggLevel', sharex='all',sharey='all',
            #                      title='%s \'%s\' errors'%(plotName, dkey),
            #                      plot_type='hist2d',
            #                      bins=50, 
            #                      vmin=0.0, vmax=5.0, xlims=(0,4),
            #                      slice_d={'resolution':10}, meta_txt=True,
            #                      meta_func = lambda **kwargs:meta_slim(**kwargs))
            #===================================================================
  
            #hist2d matrix (studyArea x resolution) 
 
            #===================================================================
            # ses.plot_err_mat(dkey=dkey, modelID_l=mids, baseID=baseID,
            #                 plot_rown='resolution', plot_coln='aggLevel', sharex='all',sharey='all',
            #                      title='%s \'%s\' errors'%(plotName, dkey),
            #                      plot_type='hist2d',
            #                      bins=50, 
            #                      vmin=0.0, vmax=0.5, xlims=(0,6),
            #                      meta_txt=True,
            #                      meta_func = lambda **kwargs:meta_slim(**kwargs))
            #===================================================================
             
  
            #===================================================================
            # total errors  (bar matrix) studyArea x resolution (aggLevel) 
            #===================================================================
            #===================================================================
            # err_type='meanError'
            # ses.plot_err_mat(dkey=dkey, modelID_l=mids,baseID=baseID,
            #                      plot_rown='studyArea', plot_coln='resolution',  plot_colr='aggLevel', 
            #                      fmt='svg',sharex='all',sharey='all', plot_type='bars',
            #                      title='%s \'%s\' %s'%(plotName, dkey, err_type), err_type=err_type, 
            #                      #meta_func=lambda **kwargs:meta_slim(**kwargs),
            #                      )
            # #   
            #===================================================================
            # #(violin matrix) studyArea x resolution (aggLevel) total errors 
            # err_type='errors'
            # ses.plot_err_mat(dkey=dkey, modelID_l=mids,baseID=baseID,
            #                      plot_rown='studyArea', plot_coln='resolution',  plot_colr='aggLevel', 
            #                      fmt='svg',sharex='all',sharey='row', 
            #                      plot_type='violin',zero_line=True,
            #                      title='%s \'%s\' %s'%(plotName, dkey, err_type), err_type=err_type, 
            #                       #meta_func=lambda **kwargs:meta_slim(**kwargs),
            #                       )
            #===================================================================
 #==============================================================================
 #            err_type='confusion'
 #            ses.plot_err_mat(dkey=dkey, modelID_l=mids,baseID=baseID,
 #                                 plot_rown='studyArea', plot_coln='resolution',  plot_bgrp='aggLevel',
 #                                 plot_colr='confusion',  bar_labels=False,normed=True,
 #                                 base_index='true',aggMethod='join',
 #                                 fmt='svg',sharex='all',sharey='all', plot_type='bars',
 #                                 title='%s \'%s\' %s'%(plotName, dkey, err_type), err_type=err_type,
 # 
 #                                 #meta_func=lambda **kwargs:meta_slim(**kwargs),
 #                                 )
 #==============================================================================
  
            #===================================================================
            # rsamp values (no error)
            #===================================================================
            #sampled only (gaussian)
 #==============================================================================
 #            ses.plot_dkey_mat2(dkey=dkey, modelID_l=mids,
 #                                 plot_rown='studyArea', plot_coln='resolution',  plot_colr='aggLevel', plot_bgrp='aggLevel',
 #                                   plot_type='gaussian_kde', density=True,drop_zeros=False,
 #                                 mean_line=False, sharey='none', sharex='all',
 #                                 title='%s \'%s\''%(plotName, dkey),
 #                                 xlims = (0, 2), 
 #                                 #ylims=(0.1, 0.9),  
 #                                 #slice_d={'studyArea':'Calgary'},
 #                                 baseID=baseID,
 # 
 #                                  #meta_func=lambda **kwargs:meta_basic(**kwargs),
 #                                    val_lab='sampled depths (m)')
 #==============================================================================
            
            #===================================================================
            # ses.plot_dkey_mat2(dkey=dkey, modelID_l=mids,
            #                      plot_rown='aggLevel', plot_coln='resolution',  
            #                      plot_colr='studyArea', plot_bgrp='studyArea',
            #                        plot_type='hist', 
            #                        xlims=(0,2),
            #                      drop_zeros=True,mean_line=False, sharey='row', sharex='all',
            #                      title='%s \'%s\' relative errors'%(plotName, dkey),
            #                      #xlims = (0, 2),  
            #                      slice_d={'studyArea':'Calgary'},
            #                       #meta_func=lambda **kwargs:meta_basic(**kwargs),
            #                         val_lab='sampled depths (m)')
            #===================================================================
             
            #===================================================================
            # #samples + raster values
            #===================================================================
            plot_type='gaussian_kde'
            density=True
            xlims = (0, 2)
            drop_zeros=False
            slice_d={}
             
            ax_d = ses.plot_rast(modelID_l = mids, plot_bgrp='resolution',                                  
                      plot_type=plot_type, mean_line=False,density=density, 
                      meta_txt=False, 
                      drop_zeros=drop_zeros, #depth rasters are mostly zeros
                      debug_max_len=1e6, write=False, linestyle='dashed', xlims=xlims, slice_d=slice_d)
                              
            ses.plot_dkey_mat2(dkey=dkey, modelID_l=mids,
                                 plot_rown='resolution', plot_coln='studyArea',  plot_colr='aggLevel', plot_bgrp='aggLevel',
                                 fmt='svg',sharex='col',sharey='col', plot_type=plot_type,
                                 drop_zeros=drop_zeros,mean_line=False,grid=True,density=density,
                                 #title='%s \'%s\' relative errors'%(plotName, dkey), 
                                  #meta_func=lambda **kwargs:meta_basic(**kwargs),
                                  ax_d=ax_d, val_lab='depths (m)',
                                  xlims=xlims, ylims=(0, 0.5),
                                  slice_d=slice_d)
            


        
        #=======================================================================
        # intersection method (sgType, samp_method)
        #=======================================================================
        #comparing all of the methods at one resolution and aggLevel
        mids = [3,53,10]
        #=======================================================================
        # ses.plot_compare_mat(dkey=dkey, modelID_l=mids,plot_rown='samp_method', plot_coln='studyArea', fmt=fmt,sharex='all',sharey='all',
        #                          title='\'%s\' errors by intersection method'%(dkey),
        #                          )
        #=======================================================================
        
        #studyArea vs. resolution (aggLevel=100)
        for plotName, mids, baseID in [
            ('zonal',        [6,7,8],     0), #sgType='zonal'
 
            ('centroid',    [56,57,58], 50), #sgType='centroids'
            ]: 
            pass
         
      
            #===================================================================
            # ses.plot_compare_mat(dkey=dkey, modelID_l=mids,plot_rown='studyArea', plot_coln='resolution', fmt=fmt,sharex='all',sharey='all',
            #                      title='%s \'%s\' errors'%(plotName, dkey), baseID=baseID)
            #===================================================================
        
        
        
        #=======================================================================
        # loss calc: Relative loss---------
        #=======================================================================
        dkey='rloss'
        
        #=======================================================================
        # rsamps vs rloss (aggLevel=0)
        #=======================================================================

        #=======================================================================
        # hazard vs asset resolution 
        #=======================================================================
 
        agg0_d = dict() #collecting agg0
        for plotName, mids, baseID in [
            #('798',list(range(9)),      0), #gridded
            ('798',  list(range(3))+list(range(60,66)), 0), #convex hulls
            #('049',list(range(40,49)),  40), #worst case FLEMO
            #('lin_g', list(range(80,89)), 80), #gridded
            ('linear', list(range(90,99)), 90), #convexHull
            ('AB',   list(range(70,79)), 70),
            ]:
 
            print('\n%s + %s\n'%(dkey, plotName))
            #===================================================================
            # values
            #===================================================================
            #===================================================================
            # ses.plot_dkey_mat2(dkey=dkey, modelID_l=mids,
            #                      plot_rown='resolution', plot_coln='studyArea',  plot_colr='aggLevel', plot_bgrp='aggLevel',
            #                      fmt='svg',sharex='col',sharey='col', 
            #                      plot_type='gaussian_kde',
            #                      drop_zeros=True,mean_line=False,
            #                      density=True,
            #                      title='%s \'%s\' values'%(plotName, dkey), 
            #                       #meta_func=lambda **kwargs:meta_basic(**kwargs),
            #                         #val_lab='sampled depths (m)',
            #                         )
            #===================================================================

            #aggLevel vs. resolution (points on vfunc)
            #===================================================================
            # ses.plot_vs_mat(modelID_l=mids, fmt='png', plot_rown='aggLevel', plot_coln='resolution',
            #                 dkey_y='rloss', dkey_x='rsamps', sharex='all', 
            #                 slice_d = {'resolution':300},
            #                 xlims=(0,3), ylims=(0,100), 
            #                 title='%s \'%s\' vs \'%s\''%(plotName, 'rloss', 'rsamps'))
            #===================================================================
             
 
            #===================================================================
            # errors
            #===================================================================
            #studyArea x resolution (aggLevel) total errors (bar matrix)
            #===================================================================
            # err_type='meanError'
            # ses.plot_err_mat(dkey=dkey, modelID_l=mids,baseID=baseID,
            #                      plot_rown='studyArea', plot_coln='resolution',  plot_colr='aggLevel', 
            #                      fmt='svg',sharex='all',sharey='all', plot_type='bars',
            #                      title='%s \'%s\' %s'%(plotName, dkey, err_type), 
            #                      err_type=err_type, 
            #                      #meta_func=lambda **kwargs:meta_slim(**kwargs),
            #                      )
            #===================================================================
            #    
            # #studyArea x resolution (aggLevel) total errors (violin matrix)
            # ses.plot_err_mat(dkey=dkey, modelID_l=mids,baseID=baseID,
            #                      plot_rown='studyArea', plot_coln='resolution',  plot_colr='aggLevel', 
            #                      fmt='svg',sharex='all',sharey='row', 
            #                      plot_type='violin',err_type='error',zero_line=True,
            #                      title='%s \'%s\' error'%(plotName, dkey),  
            #                       #meta_func=lambda **kwargs:meta_slim(**kwargs),
            #                       )
            # 
            # #scatter
 
            #===================================================================
            # 
            # ses.plot_err_mat(dkey=dkey, modelID_l=mids, baseID=baseID,
            #                  plot_rown='aggLevel', plot_coln='resolution', sharex='all',sharey='all',
            #                      title='%s \'%s\' errors'%(plotName, dkey),
            #                      plot_type='hist2d',
            #                      bins=50, 
            #                      vmin=0.0, vmax=0.01, xlims=(0,70),
            #                      meta_txt=True,
            #                      meta_func = lambda **kwargs:meta_slim(**kwargs))
            #===================================================================
            
            #=======================================================================
            # comparing with rsamps
            #=======================================================================
            #===================================================================
            # ses.plot_dkeyS_mat(dkey_l=[dkey, 'rsamps'], modelID_l=mids,
            #                      plot_rown='resolution', plot_coln='dkey', plot_bgrp='aggLevel',
            #                      fmt='svg',sharey='col', plot_type='gaussian_kde',
            #                      drop_zeros=False,mean_line=False,
            #                      density=True,slice_d={'studyArea':'LMFRA'},
            #                      title='%s \'%s\' values'%(plotName, dkey), 
            #                      xlims=(0,8),sharex='all',
            #                       #meta_func=lambda **kwargs:meta_basic(**kwargs),
            #                         #val_lab='sampled depths (m)',
            #                         )
            #===================================================================
            
            agg0_d[plotName] = mids
        
        #collect all of these mids
        mid_df = pd.DataFrame.from_dict(agg0_d)        
        print('%s post'%dkey)
        #=======================================================================
        # distirubtion per vid
        #=======================================================================
 
                    
        #histogram
        #=======================================================================
        # ses.plot_dkey_mat2(dkey='rloss', modelID_l= mid_df.stack().values.tolist(), #collapse into a list,
        #                      plot_rown='aggLevel', plot_coln='resolution',  plot_colr='vid', plot_bgrp='vid',
        #                      fmt='svg',sharex='all',sharey='row', 
        #                      plot_type='hist',
        #                      drop_zeros=True,mean_line=False,density=False,
        #                      slice_d = {'studyArea':'Calgary'}, xlims=(0,70),
        #                      #title='%s \'%s\' values'%(plotName, dkey), 
        #                       #meta_func=lambda **kwargs:meta_basic(**kwargs),
        #                         #val_lab='sampled depths (m)',
        #                         )
        #=======================================================================
 
        #gaussian
        #=======================================================================
        # ses.plot_dkey_mat2(dkey='rloss', modelID_l= mid_df.stack().values.tolist(), #collapse into a list,
        #                      plot_rown='resolution', plot_coln='vid',  #plot_colr='vid', plot_bgrp='vid',
        #                      plot_bgrp='aggLevel',
        #                      fmt='svg',sharex='all',sharey='row', 
        #                      plot_type='gaussian_kde',
        #                      drop_zeros=False,mean_line=False,density=True,
        #                      slice_d = {'studyArea':'Calgary'}, xlims=(0,70),
        #                      baseID=0,
        #                      #title='%s \'%s\' values'%(plotName, dkey), 
        #                       #meta_func=lambda **kwargs:meta_basic(**kwargs),
        #                         #val_lab='sampled depths (m)',
        #                         )
        #=======================================================================
        #=======================================================================
        # loss calc: total loss-------
        #=======================================================================
        dkey='tloss'
        #=======================================================================
        # for plotName, col in mid_df.items(): #use the frame built from above
        #     print('%s, %s'%(plotName, col.tolist()))
        #=======================================================================
        d=dict()
        for plotName, mids in (
            #('798 uni', [0, 1, 2, 60, 61, 62, 63, 64, 65]),
            ('798 area', list(range(110,119))),
            #('linear', [90, 91, 92, 93, 94, 95, 96, 97, 98]),
            #('AB uni', [70, 71, 72, 73, 74, 75, 76, 77, 78]),      
            #('AB area', list(range(100,109))),      
            ):
            d[plotName] = mids
            print('\n%s + %s\n'%(dkey, plotName))
 
            
            #===================================================================
            # values
            #===================================================================
            """not super useful as aggregated inventories have higher individual tloss"""
            
            #tvals gaussian 
            """not useful either... need to of course the values increase"""
 
            
 
            #===================================================================
            # errors
            #===================================================================
            #studyArea x resolution (aggLevel) total errors (bar matrix)
            #===================================================================
            # err_type='bias'
            # ses.plot_err_mat(dkey=dkey, modelID_l=mids,baseID=mids[0],
            #                      plot_rown='studyArea', plot_coln='resolution',  plot_colr='aggLevel', 
            #                      fmt='svg',sharex='all',sharey='all', plot_type='bars',
            #                      title='%s \'%s\' %s'%(plotName, dkey, err_type), 
            #                      err_type=err_type, #mean is not very meaningful 
            #                      #meta_func=lambda **kwargs:meta_slim(**kwargs),
            #                      )
            #===================================================================
            #    
            # #studyArea x resolution (aggLevel) total errors (violin matrix)
            # ses.plot_err_mat(dkey=dkey, modelID_l=mids,baseID=baseID,
            #                      plot_rown='studyArea', plot_coln='resolution',  plot_colr='aggLevel', 
            #                      fmt='svg',sharex='all',sharey='row', 
            #                      plot_type='violin',err_type='error',zero_line=True,
            #                      title='%s \'%s\' error'%(plotName, dkey),  
            #                       #meta_func=lambda **kwargs:meta_slim(**kwargs),
            #                       )
            # 
            #scatter
            #===================================================================
            # ses.plot_err_mat(dkey=dkey, modelID_l=mids,baseID=baseID,
            #                      plot_rown='aggLevel', plot_coln='resolution',  plot_bgrp='studyArea', 
            #                      sharex='all',sharey='all', plot_type='scatter',
            #                      title='%s \'%s\' deviation'%(plotName, dkey), 
            #                      #err_type='meanError', 
            #                      meta_func=lambda **kwargs:meta_all(**kwargs),
            #                      )
            #===================================================================
            
            #===================================================================
            # tval correlations
            #===================================================================
            #===================================================================
            # ses.plot_vs_mat(dkey_y='rloss', dkey_x='tvals',
            #                 modelID_l=mids, 
            #                 plot_rown='aggLevel', plot_coln='resolution',
            #                 slice_d={'studyArea':'Calgary'},
            #                 drop_zeros=True, 
            #                 )
            #===================================================================
            
        
        #collect all of these mids
        mid_df = pd.DataFrame.from_dict(d)     
        
        
        #=======================================================================
        # distirubtion per tval_type
        #=======================================================================
 
        #=======================================================================
        # ses.plot_dkey_mat2(dkey=dkey, modelID_l= mid_df.stack().values.tolist(), #collapse into a list,
        #                      plot_rown='aggLevel', plot_coln='studyArea',  plot_colr='vid', plot_bgrp='vid',
        #                      fmt='svg',
        #                      sharex='all',sharey='none', 
        #                      plot_type='hist',
        #                      drop_zeros=True,mean_line=False,density=False,
        #                      slice_d = {'resolution':100}, 
        #                      xlims=(0,15),
        #                      #title='%s \'%s\' values'%(plotName, dkey), 
        #                       #meta_func=lambda **kwargs:meta_basic(**kwargs),
        #                         #val_lab='sampled depths (m)',
        #                         )
        #=======================================================================

        #=======================================================================
        # Total Values----------
        #=======================================================================
        #=======================================================================
        # tdistirubtion per tval_type
        #=======================================================================
        #=======================================================================
        # dkey='tvals'
        # ses.plot_dkey_mat2(dkey=dkey, modelID_l= mid_df.stack().values.tolist(), #collapse into a list,
        #                      plot_rown='studyArea', plot_coln='tval_type',  plot_colr='aggLevel', plot_bgrp='aggLevel',
        #                      fmt='svg',
        #                      sharex='all',sharey='row', 
        #                      plot_type='violin',
        #                      drop_zeros=False,mean_line=False,density=True,
        #                      slice_d = {'resolution':10}, 
        #                      #xlims=(0,1.0),
        #                      #title='%s \'%s\' values'%(plotName, dkey), 
        #                       #meta_func=lambda **kwargs:meta_basic(**kwargs),
        #                         #val_lab='sampled depths (m)',
        #                         )
        #=======================================================================
        
        #=======================================================================
        # multi varirable-------
        #=======================================================================
        for plotName, mids, baseID in [
            #('798',list(range(9)),      0), #gridded
            ('798',  list(range(3))+list(range(60,66)), 0), #convex hulls
            #('049',list(range(40,49)),  40),
            #('lin_g', list(range(80,89)), 80), #gridded
            #('linear', list(range(90,99)), 90), #convexHull
            #('rfda',   list(range(70,79)), 70),
            ]:
 
            print('\n%s + %s\n'%(dkey, plotName))
            
            #===================================================================
            # ses.plot_perf_mat(modelID_l=mids, 
            #                   title='\'%s\' error superposition'%plotName)
            #===================================================================
            

        
        out_dir = ses.out_dir
        
    print('\nfinished %s'%tag)
    
    return out_dir

def dev():
    """problem with trues?"""
    return run(
        tag='dev',
        catalog_fp=r'C:\LS\10_OUT\2112_Agg\lib\hyd2_dev\model_run_index.csv',
        modelID_l = None,
        
        compiled_fp_d = {
        'outs':r'C:\LS\10_OUT\2112_Agg\outs\analy\dev\20220331\working\outs_analy_dev_0331.pickle',
        'agg_mindex':r'C:\LS\10_OUT\2112_Agg\outs\analy\dev\20220331\working\agg_mindex_analy_dev_0331.pickle',
        #'trues':r'C:\LS\10_OUT\2112_Agg\outs\analy\dev\20220331\working\trues_analy_dev_0331.pickle',
        
 
            }
 
        
        )
    

def r2():
    return run(
        #modelID_l = [0, 11],
        tag='r2',
        catalog_fp = r'C:\LS\10_OUT\2112_Agg\lib\hyd2\model_run_index.csv',
        compiled_fp_d = {
              'outs':r'C:\LS\10_OUT\2112_Agg\outs\analy\r2\20220316\working\outs_analy_r2_0316.pickle',
            'agg_mindex':r'C:\LS\10_OUT\2112_Agg\outs\analy\r2\20220331\working\agg_mindex_analy_r2_0331.pickle',
            'trues':r'C:\LS\10_OUT\2112_Agg\outs\analy\r2\20220331\working\trues_analy_r2_0331.pickle',
            },
        )

def r3():
    return run(
        #modelID_l = [0, 11],
        tag='r3',
        catalog_fp = r'C:\LS\10_OUT\2112_Agg\lib\hyd3\model_run_index.csv',
        compiled_fp_d = {
 
            },
        )

def r4():
    return run(
        #modelID_l = [0, 11],
        tag='r4',
        catalog_fp = r'C:\LS\10_OUT\2112_Agg\lib\hyd4\model_run_index.csv',
        compiled_fp_d = {
         'outs':r'C:\LS\10_OUT\2112_Agg\outs\analy\r4\20220410\working\outs_analy_r4_0410.pickle',
        'agg_mindex':r'C:\LS\10_OUT\2112_Agg\outs\analy\r4\20220410\working\agg_mindex_analy_r4_0410.pickle',
        #'trues':r'C:\LS\10_OUT\2112_Agg\outs\analy\r4\20220410\working\trues_analy_r4_0410.pickle',
            },
        )

def r5():
    return run(
        #modelID_l = [0, 11],
        tag='r5',
        catalog_fp = r'C:\LS\10_OUT\2112_Agg\lib\hyd5\model_run_index.csv',
        compiled_fp_d = {
        'outs':r'C:\LS\10_OUT\2112_Agg\outs\analy\r5\20220413\working\outs_analy_r5_0413.pickle',
        'agg_mindex':r'C:\LS\10_OUT\2112_Agg\outs\analy\r5\20220413\working\agg_mindex_analy_r5_0413.pickle',
        'trues':r'C:\LS\10_OUT\2112_Agg\outs\analy\r5\20220413\working\trues_analy_r5_0413.pickle',
            },
        baseID_l=[0, 40, 50], #model
        )

def r6():
    return run(
        #modelID_l = [0, 11],
        tag='r6',
        catalog_fp = r'C:\LS\10_OUT\2112_Agg\lib\hyd6\model_run_index.csv',
        baseID_l=[0, 70, 80, 90], #model
        compiled_fp_d = {
'outs':r'C:\LS\10_OUT\2112_Agg\outs\analy\r6\20220418\working\outs_analy_r6_0418.pickle',
            },
        )
    
def r7():
    return run(
        #modelID_l = [0, 11],
        tag='r7',
        catalog_fp = r'C:\LS\10_OUT\2112_Agg\lib\hyd7\model_run_index.csv',
        baseID_l=[0,
                  70, 
                  90, 100, 110
                  ], #model
        compiled_fp_d = {
        'outs':r'C:\LS\10_OUT\2112_Agg\outs\analy\r7\20220424\working\outs_analy_r7_0424.pickle',
        'agg_mindex':r'C:\LS\10_OUT\2112_Agg\outs\analy\r7\20220424\working\agg_mindex_analy_r7_0424.pickle',
        'trues':r'C:\LS\10_OUT\2112_Agg\outs\analy\r7\20220424\working\trues_analy_r7_0424.pickle',
         'drlay_fps':r'C:\LS\10_OUT\2112_Agg\outs\analy\r7\20220424\working\drlay_fps_analy_r7_0424.pickle',
         'finv_agg_fps':r'C:\LS\10_OUT\2112_Agg\outs\analy\r7\20220425\working\finv_agg_fps_analy_r7_0425.pickle',
            },
        )
if __name__ == "__main__": 
    
 
    #output=dev()
    #output=r2()
    #output=r6()
    
    output=r7()
        
        
 
    tdelta = datetime.datetime.now() - start
    print('finished in %s \n    %s' % (tdelta, output))