'''
Created on Sep. 9, 2022

@author: cefect
'''
import faulthandler
faulthandler.enable()

import os, pathlib, math, pprint, logging, sys
from definitions import proj_lib

import pandas as pd
idx = pd.IndexSlice
 
 
from agg2.haz.da import UpsampleDASession as Session
from agg2.haz.da import cat_mdex
from agg2.coms import log_dxcol

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

#===============================================================================
# globals
#===============================================================================

res_fp_lib = { #see haz.run_stats.run_haz_stats()
    #===========================================================================
    # 'r9':{
    #         'filter':{  
    #             's2': 'C:\\LS\\10_IO\\2112_Agg\\outs\\agg2\\r9\\SJ\\filter\\20220921\\stats\\SJ_r9_filter_0921_stats.pkl',
    #             's1': 'C:\\LS\\10_IO\\2112_Agg\\outs\\agg2\\r9\\SJ\\filter\\20220921\\statsF\\SJ_r9_filter_0921_statsF.pkl',
    #             'diffs': 'C:\\LS\\10_IO\\2112_Agg\\outs\\agg2\\r9\\SJ\\filter\\20220922\\diffStats\\SJ_r9_filter_0922_diffStats.pkl',
    #             },
    #         'direct':{  
    #             's2': 'C:\\LS\\10_IO\\2112_Agg\\outs\\agg2\\r9\\SJ\\direct\\20220921\\stats\\SJ_r9_direct_0921_stats.pkl',
    #             's1': 'C:\\LS\\10_IO\\2112_Agg\\outs\\agg2\\r9\\SJ\\direct\\20220921\\statsF\\SJ_r9_direct_0921_statsF.pkl',
    #             'diffs': 'C:\\LS\\10_IO\\2112_Agg\\outs\\agg2\\r9\\SJ\\direct\\20220922\\diffStats\\SJ_r9_direct_0922_diffStats.pkl'
    #             }
    #         },
    # 'r10t':{
    #     'direct':r'C:\LS\10_IO\2112_Agg\outs\agg2\t\SJ\direct\20220925\hstats\20220925\SJ_r1_hs_0925_stats.pkl',
    #     'filter':r'C:\LS\10_IO\2112_Agg\outs\agg2\t\SJ\filter\20220925\hstats\20220925\SJ_r1_hs_0925_stats.pkl',
    #     },
    # 'r10':{
    #     'direct':r'C:\LS\10_IO\2112_Agg\outs\agg2\r10\SJ\direct\20220925\hstats\20220926\SJ_r10_hs_0926_stats.pkl',
    #     'filter':r'C:\LS\10_IO\2112_Agg\outs\agg2\r10\SJ\filter\20220925\hstats\20220926\SJ_r10_hs_0926_stats.pkl'
    #     },
    #===========================================================================
    'r11':{
        'direct':r'C:\LS\10_IO\2112_Agg\outs\agg2\r11\SJ\direct\20220930\hstats\20221013\SJ_r11_hs_1013_stats.pkl',
        'filter':r'C:\LS\10_IO\2112_Agg\outs\agg2\r11\SJ\filter\20220930\hstats\20221013\SJ_r11_hs_1013_stats.pkl'
        },
    'dev':{
        'direct':r'C:\LS\10_IO\2112_Agg\outs\agg2\t\SJ\direct\20220930\hstats\20220930\SJ_dev_hs_0930_stats.pkl',
        'filter':r'C:\LS\10_IO\2112_Agg\outs\agg2\t\SJ\filter\20220930\hstats\20220930\SJ_dev_hs_0930_stats.pkl'
        }
    }

 

def SJ_da_run(
        run_name='r9',
        **kwargs):    
    
    return run_haz_plots(res_fp_lib[run_name], proj_name='SJ', run_name=run_name, **kwargs)

def SJ_dev_run(
        fp_d = {
            'filter':r'C:\LS\10_IO\2112_Agg\outs\agg2\t\SJ\filter\hstats\20220925\SJ_r1_hs_0925_stats.pkl',
            'direct':r'C:\LS\10_IO\2112_Agg\outs\agg2\t\SJ\direct\hstats\20220925\SJ_r1_hs_0925_stats.pkl'
            
            }
        
        ):
    return run_haz_plots(fp_d, proj_name='SJ', run_name='t')

 


def run_haz_plots(fp_lib,
                  pick_fp=None,
                  write=True,
                  **kwargs):
    """construct figure from SJ downscale cat results"""

    
    #===========================================================================
    # get base dir
    #=========================================================================== 
    """combines filter and direct"""
    out_dir = os.path.join(
        pathlib.Path(os.path.dirname(fp_lib['filter'])).parents[3],  # C:/LS/10_IO/2112_Agg/outs/agg2/r5
        'da', 'haz', today_str)
    
    #===========================================================================
    # execute
    #===========================================================================
    with Session(out_dir=out_dir, logger=logging.getLogger('r'),**kwargs) as ses:
        """for haz, working with aggregated zonal stats.
            these are computed on:
                aggregated (s2) data with UpsampleSession.run_stats()
                raw/fine (s1) data with UpsampleSession.run_stats_fine()
                local diff (s2-s1) with UpsampleSession.run_diff_stats()
            
 
        """
        idxn = ses.idxn
        log = ses.logger
        #log.info(ses.out_dir)
        #=======================================================================
        # DATA PREP---------
        #=======================================================================
        if pick_fp is None:
            # join the simulation results (and clean up indicides
            d = {k:pd.read_pickle(fp) for k,fp in fp_lib.items()}
            dxcol_raw = pd.concat(d, axis=1, names=['method']).swaplevel('base', 'method', axis=1).sort_index(axis=1)
     
            log_dxcol(log, dxcol_raw)
     
            
            """Windows fatal exception: access violation
            but only when I assign this to a variable used below?
            """  
            dx2 = ses.data_prep(dxcol_raw, write=True)
        else:
            dx2 = pd.read_pickle(pick_fp)
             
 
        """
        view(dx2['s2']['direct'].T)
        """
        #=======================================================================
        # HELPERS------
        #=======================================================================
        #=======================================================================
        # SUPPLEMENT PLOT-------
        #=======================================================================
        #=======================================================================
        # data prep
        #=======================================================================
        dx = dx2.reorder_levels(ses.names_l, axis=1).droplevel(['pixelLength', 'pixelArea'], axis=0)
        ses.log_dxcol(dx)
        dxi = pd.concat([
                dx.loc[:, idx['s12', :, 'wd', :, 'mean']], 
                dx.loc[:, idx['s12', :, 'wse', :, 'mean']], 
 
                dx.loc[:, idx['s12A', :, 'wd', :, 'mean']], 
                dx.loc[:, idx['s12A', :, 'wse', :, 'mean']], 
     
            ], axis=1).sort_index(axis=1) 
            
        dxi.columns, mcoln = cat_mdex(dxi.columns, levels=['base', 'layer', 'metric']) #cat layer and metric
        print(dxi.columns.unique(mcoln).tolist())
        
        #stack into a series
        serx = dxi.stack(dxi.columns.names)
        assert set(serx.index.unique(mcoln)).symmetric_difference(dxi.columns.unique(mcoln)) == set()
            
        row_l = ['s12_wd_mean','s12A_wd_mean', 's12_wse_mean',  's12A_wse_mean']
        #=======================================================================
        # plot
        #=======================================================================
        ses.plot_matrix_metric_method_var(serx, 
                                      #title=baseName,
                                      row_l=row_l, 
                                      map_d = {'row':mcoln,'col':'method', 'color':'dsc', 'x':'scale'},
                                      ylab_d={
                                        's12_wd_mean':'$\overline{WSH_{s2}-WSH_{s1}}$',
                                        's12A_wd_mean':'$\overline{WSH_{s2}} - \overline{WSH_{s1}}$',
                                         's12_wse_mean':'$\overline{WSE_{s2}-WSE_{s1}}$',
                                         's12A_wse_mean':'$\overline{WSE_{s2}} - \overline{WSE_{s1}}$'
                                          },
                                     ofp=os.path.join(ses.out_dir, f'haz_4x4_sup.{output_format}'),
                                      matrix_kwargs=dict(set_ax_title=False, add_subfigLabel=True, 
                                                            constrained_layout=True, figsize=(12 * cm, 18 * cm),fig_id=None), 
                                      ax_lims_d = {'y':{
                                          #=====================================
                                          # 's12_wd_mean':ax_ylims_d[0], 
                                          # 's12_wse_mean':ax_ylims_d[1],
                                          # 's12AN_expo_sum':(-5, 20), #'expo':(-10, 4000)... separate from full domain
                                          #=====================================
                                          }},
                                      yfmt_func= lambda x,p:'%.1f'%x,
                                      legend_kwargs={},
                                      write=True,
                                      output_fig_kwargs=dict(fmt=output_format)
                                      )
        

        return
        #=======================================================================
        # GLOBAL  ZONAL---------
        #=======================================================================
        #=======================================================================
        # lines on residuals (s12)
        #=======================================================================
 
        #=======================================================================
        # dxcol3 = dx1.loc[:, idx['s12', :, :, coln_l]].droplevel(0, axis=1)
        # serx = dxcol3.unstack().reindex(index=coln_l, level=2) 
        #=======================================================================
        
        #=======================================================================
        # ses.plot_matrix_metric_method_var(serx,
        #                                   map_d = {'row':'metric','col':'method', 'color':'dsc', 'x':'downscale'},
        #                                   ylab_d={
        #                                       'vol':'$\sum V_{s2}-\sum V_{s1}$ (m3)', 
        #                                       'wd_mean':'$\overline{WD_{s2}}-\overline{WD_{s1}}$ (m)', 
        #                                       'wse_area':'$\sum A_{s2}-\sum A_{s1}$ (m2)'},
        #                                   ofp=os.path.join(ses.out_dir, 'metric_method_rsc_resid.svg'))
        #=======================================================================
        
        #=======================================================================
        # lines on residuals NORMALIZED (s12N)
        #=======================================================================
 #==============================================================================
 #        #just water depth and the metrics
 #        dxcol3 = dx1.loc[:, idx['s12N', :, 'wd',:, metrics_l]].droplevel(['base', 'layer'], axis=1)
 # 
 #        #stack into a series
 #        serx = dxcol3.stack(level=dxcol3.columns.names).sort_index(sort_remaining=True
 #                                       ).reindex(index=metrics_l, level='metric'
 #                                        ).droplevel(['scale', 'pixelArea'])
 #==============================================================================
 
        #=======================================================================
        # ses.plot_matrix_metric_method_var(serx,
        #                                   map_d = {'row':'metric','col':'method', 'color':'dsc', 'x':'pixelLength'},
        #                                   ylab_d={
        #                                       'vol':r'$\frac{\sum V_{s2}-\sum V_{s1}}{\sum V_{s1}}$', 
        #                                       'mean':r'$\frac{\overline{WD_{s2}}-\overline{WD_{s1}}}{\overline{WD_{s1}}}$', 
        #                                       'posi_area':r'$\frac{\sum A_{s2}-\sum A_{s1}}{\sum A_{s1}}$'},
        #                                   ofp=os.path.join(ses.out_dir, 'metric_method_rsc_resid_normd.svg'))
        #=======================================================================
        
 
        
        #=======================================================================
        # four metrics
        #=======================================================================

        
        mcoln = 'layer_metric'
        m1_l = ['wd_mean', 'wse_mean', 'wse_real_area', 'wd_vol']

#===============================================================================
#         def get_stack(baseName, 
#                       metrics_l=['mean', 'posi_area', 'vol'],
#                       metrics_cat_l=None,
#                       ):
#             """common collection of data stack
#              
#              Parameters
#       
#             metrics_cat_l: list
#                 post concat order
#             """
#             
#             dxi = dx2[baseName]
#             #check all the metrics are there
#             assert set(metrics_l).difference(dxi.columns.unique('metric'))==set(), f'requested metrics no present on {baseName}'
#             
#             dxi1 = dxi.loc[:, idx[:, ('wd', 'wse'),:, metrics_l]]
#             
#             assert not dxi1.isna().all().all()
#             
#             
#             dxi1.columns, mcoln = cat_mdex(dxi1.columns) #cat layer and metric
#             
#             
#             if not metrics_cat_l is None:
#                 assert set(dxi1.columns.unique('layer_metric')).symmetric_difference(m1_l)==set(), 'metric list mismatch'
#             else:
#                 metrics_cat_l = dxi1.columns.unique('layer_metric').tolist()
#                
#             # stack into a series
#             serx = dxi1.stack(level=dxi1.columns.names).sort_index(sort_remaining=True
#                                            ).reindex(index=metrics_cat_l, level=mcoln
#                                             ).droplevel(['scale', 'pixelArea'])
#             assert len(serx)>0
#  
#                                             
# 
#             return serx, mcoln
#===============================================================================
 
        
        #=======================================================================
        # single-base raster stats
        #=======================================================================
        """
        s1 methods:
            these should be the same (masks are the same, baseline is the same)
             
        """
        #=======================================================================
        # for baseName in [
        #     's2', 's1', 's12', 
        #     #'diffs','diffsN', #these are granular
        #     ]:
        #     log.info(f'plotting {baseName} \n\n')
        #     serx = get_stack(baseName)
        #         
        #     # plot
        #     ses.plot_matrix_metric_method_var(serx,
        #                                       map_d={'row':mcoln, 'col':'method', 'color':'dsc', 'x':'pixelLength'},
        #                                       ylab_d={},
        #                                       ofp=os.path.join(ses.out_dir, 'metric_method_rsc_%s.svg' % baseName),
        #                                       matrix_kwargs=dict(figsize=(6.5, 7.25), set_ax_title=False, add_subfigLabel=True),
        #                                       ax_lims_d={
        #                                           # 'y':{'wd_mean':(-1.5, 0.2), 'wse_mean':(-0.1, 1.5), 'wse_real_area':(-0.2, 1.0), 'wd_vol':(-0.3, 0.1)},
        #                                           }
        #                                       )
        #=======================================================================
        
        #=======================================================================
        # resid normed.  Figure 5: Bias from upscaling 
        #=======================================================================

        """
        direct:
            why is direct flat when s2 is changing so much?
                because s1 and s2 are identical
                remember... direct just takes the zonal average anyway
                so the compute metric is the same as the stat
                
        wse:
            dont want to normalize this one
            
 
        direct: delta WSE has changed slightly
 
        
        """
 
        """join wd with wse from different base"""
        dxi = pd.concat([
            dx2.loc[:, idx['s12AN',:, 'wd',:, ('mean', 'vol')]],
            dx2.loc[:, idx['s12AN',:, 'wse',:, 'real_area']],
            dx2.loc[:, idx['s12A',:, 'wse',:, 'mean']],
            #dx2.loc[:, idx['s12AN',:, 'wse',:, 'real_area']]
            ], axis=1).droplevel('base', axis=1).sort_index(axis=1)
                                                
        dxi.columns, mcoln = cat_mdex(dxi.columns) #cat layer and metric
              
        serx = dxi.droplevel((0,2), axis=0).stack(dxi.columns.names
                          ).reindex(index=['full', 'DD', 'DP', 'WP', 'WW'], level='dsc' #need a bucket with all the metrics to be first
                        ).reindex(index=m1_l, level=mcoln) 
        """
        view(serx)
        """
                 
        # plot
        plt.close('all')
        ses.plot_matrix_metric_method_var(serx,
                                          map_d={'row':mcoln, 'col':'method', 'color':'dsc', 'x':'pixelLength'},
                                          ylab_d={
                                              'wd_vol':r'$\frac{\sum V_{s2}-\sum V_{s1}}{\sum V_{s1}}$',
                                              'wd_mean':r'$\frac{\overline{WSH_{s2}}-\overline{WSH_{s1}}}{\overline{WSH_{s1}}}$',
                                              'wse_real_area':r'$\frac{\sum A_{s2}-\sum A_{s1}}{\sum A_{s1}}$',
                                              'wse_mean':r'$\overline{WSE_{s2}}-\overline{WSE_{s1}}$',
                                              },
                                          ofp=os.path.join(ses.out_dir, 'metric_method_rsc_%s.svg' % ('global')),
                                          matrix_kwargs=dict(figsize=(6.5, 7.25), set_ax_title=False, add_subfigLabel=True),
                                          ax_lims_d={
                                              'y':{'wd_mean':(-1.5, 0.2), 'wse_real_area':(-0.2, 1.0), 'wd_vol':(-0.3, 0.1),
                                                   'wse_mean':(-1.0, 15.0),
                                                   },
                                              }
                                          )
 
        
        #=======================================================================
        # for presentation (WD and A)
        #=======================================================================
        #=======================================================================
        # m_l = ['mean', 'posi_area']
        # dx1 = dx1.loc[:, idx['s12N', :, 'wd',:, m_l]
        #                  ].droplevel(['base', 'layer'], axis=1).droplevel(('scale', 'pixelArea')
        #                       ).drop('direct', level='method',axis=1)
        #   
        # #stack into a series
        # serx = dx1.stack(level=dx1.columns.names).sort_index(sort_remaining=True
        #                                ).reindex(index=m_l, level='metric') 
        #   
        #   
        # #plot
        # ses.plot_matrix_metric_method_var(serx,
        #                                   map_d = {'row':'metric','col':'method', 'color':'dsc', 'x':'pixelLength'},
        #                                   ylab_d={
        #                                     #'vol':r'$\frac{\sum V_{s2}-\sum V_{s1}}{\sum V_{s1}}$', 
        #                                     'mean':r'$\frac{\overline{WD_{s2}}-\overline{WD_{s1}}}{\overline{WD_{s1}}}$', 
        #                                     'posi_area':r'$\frac{\sum A_{s2}-\sum A_{s1}}{\sum A_{s1}}$',
        #                                       },
        #                                   ofp=os.path.join(ses.out_dir, 'metric_method_rsc_resid_normd_present.svg'),
        #                                   matrix_kwargs = dict(figsize=(3,4), set_ax_title=False, add_subfigLabel=False),
        #                                   ax_lims_d = {
        #                                       'y':{'mean':(-1.5, 0.2),  'posi_area':(-0.2, 1.0), 'wd_vol':(-0.3, 0.1)},
        #                                       },
        #                                   output_fig_kwargs=dict(transparent=False, add_stamp=False),
        #                                   legend_kwargs=dict(loc=3),
        #                                   ax_title_d=dict(), #no axis titles
        #                                   )
        #=======================================================================
        #=======================================================================
        # LOCAL ZONAL---------
        #=======================================================================
        #=======================================================================
        # single-base raster stats
        #=======================================================================
        """
        No sum on s12? no granular vol -- added
 
        
        wse.real_area
            this is constant because filter to the s1 inundation area
            thats why we use s12_TP
            
        filter.s12_TP.full
            where did this go?
        
        """
        dx2.columns.names
        #dx2['s12']['direct']['wd']['full'] #.loc[:, idx[:, 'real_area']]
        
        dxi = pd.concat([
            dx2.loc[:, idx['s12N',:, 'wd',:, ('mean')]],

            dx2.loc[:, idx['s12',:, 'wse',:, 'mean']],
            dx2.loc[:, idx['s12_TP',:, 'wse',:, 'mean']],
            dx2.loc[:, idx['s12N',:, 'wd',:, 'vol']],
            ], axis=1).sort_index(axis=1)
                                             
        dxi.columns, mcoln = cat_mdex(dxi.columns, levels=['base', 'layer', 'metric']) #cat layer and metric
        
        print(dxi.columns.unique(mcoln).tolist())
        m1_l = ['s12N_wd_mean', 's12_wse_mean',  's12_TP_wse_mean','s12N_wd_vol',]
           
        serx = dxi.droplevel((0,2), axis=0).stack(dxi.columns.names) 
                        
        assert set(serx.index.unique(mcoln)).symmetric_difference(dxi.columns.unique(mcoln))==set() 
        """
        view(serx.loc[:, 'filter', :, 's12_TP_wse_mean'])
        
        dxi.droplevel((0,2), axis=0).sort_index(axis=1).columns.unique(mcoln)
        view(dxi.droplevel((0,2), axis=0).sort_index(axis=1))
        view(serx)
        """
              
        # plot
        plt.close('all')
        ses.plot_matrix_metric_method_var(serx,
                                          map_d={'row':mcoln, 'col':'method', 'color':'dsc', 'x':'pixelLength'},
                                          row_l=m1_l,
                                          ylab_d={
                                              #=================================
                                              # 'wd_vol':r'$\frac{\sum V_{s2}-\sum V_{s1}}{\sum V_{s1}}$',
                                              # 'wd_mean':r'$\frac{\overline{WSH_{s2}}-\overline{WSH_{s1}}}{\overline{WSH_{s1}}}$',
                                              # 'wse_real_area':r'$\frac{\sum A_{s2}-\sum A_{s1}}{\sum A_{s1}}$',
                                              # 'wse_mean':r'$\overline{WSE_{s2}}-\overline{WSE_{s1}}$',
                                              #=================================
                                              },
                                          ofp=os.path.join(ses.out_dir, 'metric_method_rsc_%s.svg' % ('local')),
                                          matrix_kwargs=dict(figsize=(6.5, 7.25), set_ax_title=False, add_subfigLabel=True),
                                          ax_lims_d={
                                              'y':{
                                                    #'s12N_wd_mean':(-1.5, 0.2), 
                                                   #'wse_real_area':(-0.2, 1.0), 
                                                   's12N_wd_vol':(-0.3, 0.1),
                                                   's12_wse_mean':(-1.0, 15.0),
                                                   },
                                              }
                                          )
        
        #=======================================================================
        # stackced areas ratios
        #=======================================================================
        
        #=======================================================================
        # #compute fraction
        #=======================================================================
        
        #=======================================================================
        # #reduce
        # dx1 = dxcol_raw.loc[:, idx['s2',:,'wd',:,'post_count']].droplevel(('base', 'metric', 'layer'), axis=1).droplevel((1,2), axis=0)
        # df1 = dx1.drop('full',level='dsc', axis=1).dropna().drop('filter',level='method', axis=1).droplevel('method', axis=1)
        #  
        # #compute fraction
        # fdf = df1.divide(df1.sum(axis=1), axis=0)
        #   
        # #plot
        # ses.plot_dsc_ratios(fdf)
        #=======================================================================
        

def run_kde_plots(pick_fp, 
                  **kwargs):
    """plot lines from kDE calcs
    
    Parameters
    -----------
    pick_fp: str
        dxcol output from run_stats.compute_kde()
        
        
    """

    
    #===========================================================================
    # get base dir
    #=========================================================================== 
    """combines filter and direct"""
    out_dir = os.path.dirname(pick_fp)
    
    #===========================================================================
    # execute
    #===========================================================================
    with Session(out_dir=out_dir,logger=logging.getLogger('r'), **kwargs) as ses:
        """
        too much difference between 1 and 8
            try histograms?
        """
        idxn = ses.idxn
        log = ses.logger
        
        #=======================================================================
        # load data
        #=======================================================================
        dxcol = pd.read_pickle(pick_fp)
 
        
        dx1 = pd.concat({'haz':dxcol}, axis=1, names=['base']) #add dummy level
        
        serx_raw = dx1.stack(dx1.columns.names).droplevel('coord')
        
        #drop zeros
        serx = serx_raw[serx_raw!=0]
        
        """
        df1 = dx1['haz']['direct'][1].droplevel('coord').reset_index()
        view(df1)
        df1.plot(x='x', y=1)
        """
              
        #=======================================================================
        # plot      
        #=======================================================================
        ses.plot_matrix_metric_method_var(serx,
                                          map_d={'row':'base', 'col':'method', 'color':'scale', 'x':'x'},
 
                                          plot_kwargs_lib={},
                                          plot_kwargs={'linestyle':'solid', 'marker':None, 'markersize':7, 'alpha':0.6},
                                          colorMap='copper',
                                           matrix_kwargs=dict(figsize=(10,4), set_ax_title=False, add_subfigLabel=True, fig_id=0, constrained_layout=True),
                                           xlab='WSH (m)',
                                           ylab_d={'haz':'frequency'}
                                          )
        #ses.plot_kde_set(dxcol)
   
if __name__ == "__main__":

  
    SJ_da_run(run_name='r11',
              #pick_fp = r'C:\LS\10_IO\2112_Agg\outs\agg2\r11\SJ\da\haz\20221006\SJ_r11_direct_1006_dprep.pkl',
              )
    
    #===========================================================================
    # run_kde_plots(
    #     r'C:\LS\10_IO\2112_Agg\outs\agg2\r10\SJ\da\rast\20220930\SJ_r10_direct_0930_kde_dxcol.pkl'
    #     )
    #===========================================================================
    
    print('finished')
        
