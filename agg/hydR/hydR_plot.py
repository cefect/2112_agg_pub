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
        'family' : 'serif',
        'weight' : 'normal',
        'size'   : 6}

matplotlib.rc('font', **matplotlib_font)
matplotlib.rcParams['axes.titlesize'] = 8 
matplotlib.rcParams['axes.labelsize'] = 8

matplotlib.rcParams['figure.titlesize'] = 4
#matplotlib.rcParams['figure.titleweight']='bold'

#spacing parameters
matplotlib.rcParams['figure.autolayout'] = False #use tight layout

#legends
matplotlib.rcParams['legend.title_fontsize'] = 'large'

matplotlib.rcParams['lines.markersize'] = 3.0

print('loaded matplotlib %s'%matplotlib.__version__)

from agg.hydR.hydR_scripts import RastRun,  Catalog, Error
from hp.pd import view
from hp.plot import Plotr
from hp.gdal import rlay_to_array, getRasterMetadata
from hp.basic import set_info, get_dict_str
from hp.pd import get_bx_multiVal
#from hp.animation import capture_images

class RasterPlotr(RastRun, Plotr): #analysis of model results
    
    ax_title_d = {
            'obwb':'Lake Coastal',
              'LMFRA':'Lower R. (protected)',
               'Calgary':'Middle R. (confined)', 
               'noise':'Uniform Noise'
            }
    
    color_lib = {'dsampStage': {'post': '#8e0152', 'postFN': '#f5c4e1', 'pre': '#c7e89f', 'preGW': '#276419'}}
    
    
    def __init__(self,
 
                 name='hydR_plot',
                 plt=None,
                 exit_summary=False,
                 colorMap_d={},
                 **kwargs):
        
        data_retrieve_hndls = {
            'catalog':{
                #probably best not to have a compliled version of this
                'build':lambda **kwargs:self.build_catalog(**kwargs), #
                },
            
            }
        
        colorMap_d.update({
            'studyArea':'Dark2',
            'resolution':'copper',
            'dkey':'Pastel2',
            'dsampStage':'PiYG',
            'downSampling':'Set1',
            'sequenceType':'Set2',
                        })
        
        self.colorMap_d=colorMap_d
        self.plt=plt
        
        super().__init__(data_retrieve_hndls=data_retrieve_hndls,name=name,init_plt_d=None,
                         exit_summary=exit_summary,**kwargs)
        
    def compileAnalysis(self,
                        ):
        """not much compiling... mostly working with raw rasters"""
        self.retrieve('catalog')
        
        
    def build_catalog(self,
                      dkey='catalog',
                      catalog_fp=None,
                      logger=None,
                      **kwargs):
        if logger is None: logger=self.logger
        assert dkey=='catalog'
        if catalog_fp is None: catalog_fp=self.catalog_fp
        
        return Catalog(catalog_fp=catalog_fp, logger=logger, overwrite=False,
                       index_col=self.index_col,
                        **kwargs).get()
    

 
        
    
    #===========================================================================
    # PLOTRS------
    #===========================================================================
 
    def plot_rvalues(self, #flexible plotting of raster values
                  
                    #data control
                    drop_zeros=True, 
 
                    #data 
                    fp_serx=None, #catalog
                    debug_max_len=1e4,
                                      
                    
                    #plot config
                    plot_type='gaussian_kde', 
                    plot_rown='studyArea',
                    plot_coln='dsampStage',
                    plot_colr='resolution',                    
                    plot_bgrp=None, #grouping (for plotType==bars)

                     
                    #histwargs
                    bins=20, rwidth=0.9, 
                    hrange=None, #slicing the data (similar to xlims.. but this affects the calcs)
                    mean_line=False, #plot a vertical line on the mean
                    density=True,
 
 
                    #meta labelling
                    meta_txt=True, #add meta info to plot as text
                    meta_func = lambda meta_d={}, **kwargs:meta_d, #lambda for calculating additional meta information (add_meta=True)        
                    write_meta=False, #write all the meta info to a csv            
                    
                    write_stats=True, #calc and write stats from raster arrays
                    
                    #plot style                    
                    colorMap=None, title=None, val_lab='depth (m)', 
                    grid=True,
                    sharey='row',sharex='row',
                    xlims=None, #also see hrange
                    ylims=None,
                    figsize=None,
                    yscale='linear',
                    
                    #output
                    fmt='svg',write=None,
 
                    **kwargs):
        """"
        similar to plot_dkey_mat (more flexible)
        similar to plot_err_mat (1 dkey only)
        
        
        """
        
        #=======================================================================
        # defaults
        #=======================================================================
        log = self.logger.getChild('plot_rvalues')
 
        resCn, saCn = self.resCn, self.saCn
        if write is None: write=self.write
        #retrieve data
        """just the catalog... we load each raster in the loop"""
        if fp_serx is None:
            fp_serx = self.retrieve('catalog')['rlay_fp']
 
            
        #plot keys
        if plot_colr is None: 
            plot_colr=plot_bgrp
        
        if plot_colr is None: 
            plot_colr=plot_rown
            
        if plot_bgrp is None:
            plot_bgrp = plot_colr
            
        #=======================================================================
        # key checks
        #=======================================================================
 
        assert not plot_rown==plot_coln
            
        #plot style
                 
        if title is None:
            title = 'Depth Raster Values'
                
        if colorMap is None: colorMap = self.colorMap_d[plot_colr]
        
 
        
        if plot_type in ['violin', 'bar']:
            assert xlims is None
            
        #=======================================================================
        # data prep
        #=======================================================================
        if plot_coln is None:
            """add a dummy level for c onsistent indexing"""
            plot_coln = ''
            fp_serx = pd.concat({plot_coln:fp_serx}, axis=0, names=[plot_coln])
        
 
        #=======================================================================
        # setup the figure
        #=======================================================================
        mdex = fp_serx.index
        plt.close('all')
 
        col_keys =mdex.unique(plot_coln).tolist()
        row_keys = mdex.unique(plot_rown).tolist()
 
        fig, ax_d = self.get_matrix_fig(row_keys, col_keys,
                                    figsize_scaler=4,
                                    constrained_layout=True,
                                    sharey=sharey, 
                                    sharex=sharex,  
                                    fig_id=0,
                                    set_ax_title=True,figsize=figsize,
                                    )
     
 

        assert isinstance(fig, matplotlib.figure.Figure)
        fig.suptitle(title)
        #=======================================================================
        # #get colors
        #=======================================================================
 
        ckeys = mdex.unique(plot_colr) 
        color_d = self.get_color_d(ckeys, colorMap=colorMap)
        
        #=======================================================================
        # loop and plot
        #=======================================================================
        stats_dx = None
        meta_dx=None
        for gkeys, gdx0 in fp_serx.groupby(level=[plot_coln, plot_rown]): #loop by axis data
            
            #===================================================================
            # setup
            #===================================================================
            keys_d = dict(zip([plot_coln, plot_rown], gkeys))
            ax = ax_d[gkeys[1]][gkeys[0]]
            log.info('on %s'%keys_d)
            
            assert len(gdx0.index.unique('resolution'))==len(gdx0.index.get_level_values('resolution'))
            
            meta_d = {'layers':len(gdx0), 'drop_zeros':drop_zeros}
            #===================================================================
            # build data-------
            #===================================================================
            #===================================================================
            # get filepaths
            #===================================================================
            gdf0 = gdx0.droplevel(list(range(1,5)))
            
            #set order
            if plot_type=='gaussian_kde':
                fp_d = gdf0.sort_index(ascending=False).to_dict()
            else:
                fp_d = gdf0.to_dict()
                
            
            
            #===================================================================
            # pull the raster data
            #===================================================================
            data_d, sdxi, d = self.get_raster_data(fp_d,  
                                drop_zeros=drop_zeros,debug_max_len=debug_max_len,
                                                logger=log.getChild(str(gkeys)))
            
            meta_d.update(d)
            
            
            if stats_dx is None:
                stats_dx = sdxi
            else:
                stats_dx = stats_dx.append(sdxi)
            
            #===============================================================
            # plot------
            #===============================================================
            cdi = {k:color_d[k] for k in data_d.keys()} #subset and reorder
 
            """setup to plot a set... but just using the loop here for memory efficiency"""
            md1 = self.ax_data(ax, data_d,
                           plot_type=plot_type, 
                           bins=bins, rwidth=rwidth, 
                           mean_line=None, hrange=hrange, density=density,
                           color_d=cdi, 
                           logger=log,
                           violin_line_kwargs=dict(color='red', alpha=0.75, linewidth=0.75),
                           label_key=plot_bgrp, **kwargs) 
            """
                fig.show()
                """
 
             
            #===================================================================
            # add plots--------
            #===================================================================
            if mean_line:
                raise Error('not implemented')
                mval =gdx0.mean().mean()
            else: mval=None 
            

 
            meta_d.update(md1)
            
            """this is usually too long
            labels = ['%s=%s'%(plot_bgrp, k) for k in data_d.keys()]"""
            labels = [str(v) for v in data_d.keys()]
 
            #===================================================================
            # post format 
            #===================================================================
            
            ax.set_title(' & '.join(['%s:%s' % (k, v) for (k, v) in keys_d.items()]))
            #===================================================================
            # meta  text
            #===================================================================
            """for bars, this ignores the bgrp key"""
            meta_d = meta_func(logger=log, meta_d=meta_d, pred_ser=gdx0)
            if meta_txt:
                ax.text(0.1, 0.9, get_dict_str(meta_d), 
                        bbox=dict(boxstyle="round,pad=0.05", fc="white", lw=0.0,alpha=0.9 ),
                        transform=ax.transAxes, va='top', fontsize=8, color='black')
            
            
            #===================================================================
            # collect meta 
            #===================================================================
            meta_serx = pd.Series(meta_d, name=gkeys)
            
            if meta_dx is None:
                meta_dx = meta_serx.to_frame().T
                meta_dx.index.set_names(keys_d.keys(), inplace=True)
            else:
                meta_dx = meta_dx.append(meta_serx)
                
 
                
        #===============================================================
        # post format subplot ----------
        #===============================================================
        """best to loop on the axis container in case a plot was missed"""
        for row_key, d in ax_d.items():
            for col_key, ax in d.items():
                if grid: ax.grid()
                

                
                if plot_type in ['box', 'violin']:
                    ax.set_xticklabels([])
                    
                
                
                if not xlims is None:
                    ax.set_xlim(xlims)
                    
                if not ylims is None:
                    ax.set_ylim(ylims)
                    
                #chang eto log scale
                ax.set_yscale(yscale)
                
                
                # first row
                if row_key == row_keys[0]:
                    #last col
                    if col_key == col_keys[-1]:
                        if plot_type in ['hist', 'gaussian_kde', 
                                         #'violin' #no handles 
                                         ]:
                            ax.legend()
                
                        
                # first col
                if col_key == col_keys[0]:
                    if plot_type in ['hist', 'gaussian_kde']:
                        if density:
                            ax.set_ylabel('density')
                        else:
                            ax.set_ylabel('count')
                    elif plot_type in ['box', 'violin']:
                        if not val_lab is None: ax.set_ylabel(val_lab)
                
                #last row
                if row_key == row_keys[-1]:
                    if plot_type in ['hist', 'gaussian_kde']:
                        if not val_lab is None: ax.set_xlabel(val_lab)
                    elif plot_type in ['violin', 'box']:
                        ax.set_xticks(np.arange(1, len(labels) + 1))
                        ax.set_xticklabels(labels)
                    #last col
                    if col_key == col_keys[-1]:
                        pass
                        
                    
 
 
        #=======================================================================
        # wrap---------
        #=======================================================================
        log.info('finsihed')
        """
        fig.show()
        plt.show()
        """
 
        fname = 'rvalues_%s_%s_%sX%s_%s_%s' % (
            title,plot_type, plot_rown, plot_coln, val_lab, self.longname)
                
        fname = fname.replace('=', '-').replace(' ','').replace('\'','')
        
        try:
            if write_meta:
                
                ofp =  os.path.join(self.out_dir, fname+'_meta.csv')
                meta_dx.to_csv(ofp)
                log.info('wrote meta_dx %s to \n    %s'%(str(meta_dx.shape), ofp))
                
            if write_stats:
                ofp =os.path.join(self.out_dir, fname+'_stats.csv')
                stats_dx.to_csv(ofp)
                log.info('wrote stats_dx %s to \n    %s'%(str(stats_dx.shape), ofp))
        except Exception as e:
            log.error('failed to write meta or stats data w/ \n    %s'%e)
               
        if write:
            self.output_fig(fig, fname=fname, fmt=fmt)
        
        return ax_d
 
 


    def plot_rValsVs(self, #plot raster values against some indexer
 
                    #data 
                    fp_serx=None, #catalog

                    
                    ystat='mean', #for statistical plot types
                    xcoln='resolution',
                    
                    #raster data retrival [get_raster_data]
                    rkwargs = dict(
                        debug_max_len=None,
                        min_cell_cnt=1,
                        drop_zeros=False, 
                        ),
                                      
                    
                    #plot config
                    ax_d=None,
                    plot_types=['line', 'box'], 
                    plot_rown='studyArea',
                    plot_coln=None,
                    plot_colr=None,                    
                    plot_bgrp='dsampStage',
                    
                    #box config
                    shift_d=None,
                    width=2,
 
 
                    #meta labelling
                    meta_txt=False, #add meta info to plot as text
                    meta_func = lambda meta_d={}, **kwargs:meta_d, #lambda for calculating additional meta information (add_meta=True)        
                    write_meta=False, #write all the meta info to a csv            
                    
                    write_stats=False, #calc and write stats from raster arrays
                    
                    #plot style                    
                    colorMap=None, title=None, 
                    grid=True,
                    sharey='row',sharex='row',
                    xlims=None, #also see hrange
                    ylims=None,
                    ylab='depth (m)',
                    xlab=None, 
                    figsize=None,
                    yscale='linear',xscale='linear',
                    plot_kwargs = dict(),
                    
                    #output
                    fmt='png',write_fig=True,
 
                    ):
        """"
        similar to plot_dkey_mat (more flexible)
        similar to plot_err_mat (1 dkey only)
        
        
        """
        
        #=======================================================================
        # defaults
        #=======================================================================
        log = self.logger.getChild('plot_rvalues')
 
        resCn, saCn = self.resCn, self.saCn
 
        #retrieve data
        """just the catalog... we load each raster in the loop"""
        if fp_serx is None:
            fp_serx = self.retrieve('catalog')['rlay_fp']
 
        assert isinstance(fp_serx, pd.Series)

        #=======================================================================
        # #plot keys
        #=======================================================================
        if plot_colr is None: 
            plot_colr=plot_bgrp
        
        if plot_colr is None: 
            plot_colr=plot_rown
            
        if plot_bgrp is None:
            plot_bgrp = plot_colr
            
        
            
        #=======================================================================
        # key checks
        #=======================================================================
        """set up to plot against some indexer... for most plot types this has to be resolution?"""
        assert xcoln in fp_serx.index.names
        assert not plot_rown==plot_coln
            
        #=======================================================================
        # #plot style
        #=======================================================================                 
        if title is None:
            title = 'Depth Raster Values'
            
        if xlab is None: xlab=xcoln
                
        if colorMap is None: colorMap = self.colorMap_d[plot_colr]
        
            
            
        #=======================================================================
        # data prep
        #=======================================================================
        if plot_coln is None:
            """add a dummy level for c onsistent indexing"""
            plot_coln = ''
            fp_serx = pd.concat({plot_coln:fp_serx}, axis=0, names=[plot_coln])
        
 
        #=======================================================================
        # setup the figure
        #=======================================================================
        mdex = fp_serx.index
        
 
        col_keys =mdex.unique(plot_coln).tolist()
        row_keys = mdex.unique(plot_rown).tolist()
 
        if ax_d is None:
            plt.close('all')
            fig, ax_d = self.get_matrix_fig(row_keys, col_keys,
                                        figsize_scaler=4,
                                        constrained_layout=True,
                                        sharey=sharey, 
                                        sharex=sharex,  
                                        fig_id=0,
                                        set_ax_title=True,figsize=figsize,
                                        )
         
     
    
            
            
        else:
            for k,v in ax_d.items():
                fig = v[list(v.keys())[0]].figure
                break
            
        assert isinstance(fig, matplotlib.figure.Figure)
        fig.suptitle(title)
        #=======================================================================
        # #get colors
        #=======================================================================
 
        ckeys = mdex.unique(plot_colr) 
        
        
        """nasty workaround to get colors to match w/ hyd""" 
        if plot_colr =='dsampStage':
            ckeys = ['none'] + ckeys.values.tolist()
        
        color_d = self.get_color_d(ckeys, colorMap=colorMap)
        #=======================================================================
        # get shift
        #=======================================================================
        if shift_d is None and 'box' in plot_types:
            """distance percentage per group"""
            shift_d = dict(zip(mdex.unique(plot_bgrp), 
                               np.linspace(0, width*1.1, len(mdex.unique(plot_bgrp)))-width*0.5
                               ))
 
 
 
        
        #=======================================================================
        # loop and plot
        #=======================================================================
        stats_dx = None
        meta_dx=None
        for gkeys0, gdx0 in fp_serx.groupby(level=[plot_coln, plot_rown]): #loop by axis data
            
            #===================================================================
            # setup
            #===================================================================
            
            keys_d = dict(zip([plot_coln, plot_rown], gkeys0))
            ax = ax_d[gkeys0[1]][gkeys0[0]]
            log.info('on %s'%keys_d)
            
            
            
            meta_d = {'layers':len(gdx0)}
            
            #===================================================================
            # loop on group 
            #===================================================================
            for gk1, gdx1 in gdx0.groupby(level=plot_bgrp):
                keys_d[plot_bgrp] = gk1
                kstr = '.'.join(keys_d.values())
                color = color_d[keys_d[plot_colr]]
                assert plot_colr==plot_bgrp
                
                assert len(gdx1.index.unique('resolution'))==len(gdx1.index.get_level_values('resolution'))
                #===================================================================
                # build data-------
                #===================================================================
                #===================================================================
                # get filepaths
                #===================================================================
                #drop all the levels except the plotting indexer
                gdf1 = gdx1.droplevel(list(set(gdx0.index.names).difference([xcoln])))
                
                #get filepahts (order shouldnt matter)
                fp_d = gdf1.to_dict()
     
                
                #===================================================================
                # pull the raster data
                #===================================================================
                """not the most efficient... but more flexible than relying on raster functions"""
                data_d, sdxi, d = self.get_raster_data(fp_d, logger=log.getChild(kstr), **rkwargs)
                
                meta_d.update({'%s_%s'%(gk1, k):v for k,v in d.items()})
                
                
                if stats_dx is None:
                    stats_dx = sdxi
                else:
                    stats_dx = stats_dx.append(sdxi)
                    
                #===================================================================
                # get the plotting stat
                #===================================================================
                xar = np.array(list(data_d.keys()))
                if not ystat is None:
                    yar = np.array([getattr(ar, ystat)() for k,ar in data_d.items()])
                    
                
                #===============================================================
                # plot each
                #===============================================================
                for plot_type in plot_types:
                    #===============================================================
                    # plot.line------
                    #===============================================================
                    if plot_type=='line':
                        ax.plot(xar, yar, color=color,
                                label ='%s (%s; zeros=%s)'%(keys_d[plot_bgrp], ystat, not rkwargs['drop_zeros']),
                                **plot_kwargs)
                    
                    #===============================================================
                    # plot.box-------    
                    #===============================================================
                        """
                        fig.show()
                        boxres_d.keys()
                        """
                    elif plot_type =='box':
                        #apply shift 
                        #shift_ar=np.append(np.diff(xar), np.diff(xar)[-1])*shift_d[gk1]
                        pos_ar = xar +  shift_d[gk1] 
                        #wid_ar = (np.append(np.diff(xar), np.diff(xar)[-1])/len(shift_d))*0.5
                        
                        boxres_d = ax.boxplot(data_d.values(),   meanline=False, 
                                              positions=pos_ar,widths=width,notch=True,
                            boxprops={'color':color, 'facecolor':color, 'linewidth':0.0, 'alpha':0.5}, patch_artist=True,
                            medianprops={'color':'black'},
                            whiskerprops={'color':color},
                            showcaps=False,
                            flierprops={'markeredgecolor':color, 'markersize':2,'alpha':0.2},
                            )
                        
                    elif plot_type=='zero_line':
                        ax.axhline(0.0, color='black', linestyle='solid', linewidth=0.5)
 
                    else:
                        raise Error(plot_type)
                    
 
     
                #===================================================================
                # post format 
                #===================================================================
                labels = [str(v) for v in data_d.keys()]
     
                del keys_d[plot_bgrp]
                ax.set_title(' & '.join(['%s:%s' % (k, v) for (k, v) in keys_d.items() if not k=='']))
                #===================================================================
                # meta  text
                #===================================================================
     
                meta_d = meta_func(logger=log, meta_d=meta_d, pred_ser=gdx0)
                if meta_txt:
                    ax.text(0.1, 0.9, get_dict_str(meta_d), 
                            bbox=dict(boxstyle="round,pad=0.05", fc="white", lw=0.0,alpha=0.5 ),
                            transform=ax.transAxes, va='top', fontsize=8, color='black')
                
                
                #===================================================================
                # collect meta 
                #===================================================================
                meta_serx = pd.Series(meta_d, name=tuple(keys_d.keys()))
                
                if meta_dx is None:
                    meta_dx = meta_serx.to_frame().T
                    meta_dx.index.set_names(keys_d.keys(), inplace=True)
                else:
                    meta_dx = meta_dx.append(meta_serx)
                
 
                
        #===============================================================
        # post format subplot ----------
        #===============================================================
        """best to loop on the axis container in case a plot was missed"""
        for row_key, d in ax_d.items():
            for col_key, ax in d.items():
                if grid: ax.grid()
                
 
                
                if not xlims is None:
                    ax.set_xlim(xlims)
                    
                if not ylims is None:
                    ax.set_ylim(ylims)
                    
                #chang eto log scale
                if not yscale is None:
                    ax.set_yscale(yscale)
                ax.set_xscale(xscale)
                
                
                # first row
                if row_key == row_keys[0]:
                    #last col
                    if col_key == col_keys[-1]:
                        ax.legend()
                            
                
                        
                # first col
                if col_key == col_keys[0]:
                    if not ylab is None: ax.set_ylabel(ylab)
                        
                
                #last row
                if row_key == row_keys[-1]:
 
                    if not xlab is None:ax.set_xlabel(xlab)
                        
 
                    #last col
                    if col_key == col_keys[-1]:
                        pass
                        
                    
 
 
        #=======================================================================
        # wrap---------
        #=======================================================================
        log.info('finsihed')
        """
        fig.show()
        plt.show()
        """
 
        fname = 'rValsVs_%s_%s_%sX%s_%s_%s' % (
            title,plot_type, plot_rown, plot_coln, ylab, self.longname)
                
        fname = fname.replace('=', '-').replace(' ','').replace('\'','')
        
        try:
            if write_meta:
                
                ofp =  os.path.join(self.out_dir, fname+'_meta.csv')
                meta_dx.to_csv(ofp)
                log.info('wrote meta_dx %s to \n    %s'%(str(meta_dx.shape), ofp))
                
            if write_stats:
                ofp =os.path.join(self.out_dir, fname+'_stats.csv')
                stats_dx.to_csv(ofp)
                log.info('wrote stats_dx %s to \n    %s'%(str(stats_dx.shape), ofp))
        except Exception as e:
            log.error('failed to write meta or stats data w/ \n    %s'%e)
               
        if write_fig:
            log.info('outputting fig')
            return self.output_fig(fig, fname=fname, fmt=fmt)
        else:
            return ax_d
        
 
 
 
    
    def plot_StatVsResolution(self, #single stat against resolution plots
                         #data
                         dx_raw=None, #combined model results
                         coln = 'MEAN', #variable to plot against resolution
                         
                         #plot control
                        plot_type='line', 
                        plot_rown='studyArea',
                        plot_coln=None,
                        plot_colr=None,
                        plot_bgrp='dsampStage', #sub-group onto an axis
                        
                        
                           colorMap=None,
                         plot_kwargs = dict(marker='x'),
                         title=None,xlabel=None,ylabel=None,
                         xscale='log',
                         xlims=None,
                         
                         
                         #plot control [matrix]
                         figsize=None,
                         sharey='none',sharex='col',
                         
                         
                         logger=None, write=None):
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is  None: logger=self.logger
        log = logger.getChild('plot_StatVsResolution')
        resCn, saCn = self.resCn, self.saCn
        if write is None: write=self.write
        
        if plot_colr is None: plot_colr=plot_bgrp
        if colorMap is None: colorMap=self.colorMap_d[plot_colr]
        
        if xlabel is None: xlabel = resCn
        if ylabel is None: ylabel=coln
        #=======================================================================
        # retrival
        #=======================================================================
        if dx_raw is None: dx_raw = self.retrieve('catalog')
        """
        view(dx_raw)
        dx_raw.index.names
        """
        
        #=======================================================================
        # precheck
        #=======================================================================
        assert coln in dx_raw.columns, coln
 
        
        #=======================================================================
        # data prep----
        #=======================================================================
        log.info('on %i'%len(dx_raw))
        serx = dx_raw[coln]
        
        if plot_coln is None:
            """add a dummy level for c onsistent indexing"""
            plot_coln = ''
            serx = pd.concat({plot_coln:serx}, axis=0, names=[plot_coln])
 
        mdex = serx.index
        gcols = set()
        for c in [plot_bgrp, plot_coln, plot_rown]:
            if not c is None: 
                gcols.add(c)
                assert c in mdex.names, c
        gcols = list(gcols)
        #=======================================================================
        # plot defaults
        #=======================================================================
        #title
        if title is None:
            title='%s vs. %s'%(coln, resCn)
        #get colors
        ckeys = mdex.unique(plot_colr)
        
        """nasty workaround to get colors to match w/ hyd""" 
        if plot_colr =='dsampStage':
            ckeys = ['none'] + ckeys.values.tolist()
        
        color_d = self.get_color_d(ckeys, colorMap=colorMap)
        
        
        #=======================================================================
        # setup the figure
        #=======================================================================
        plt.close('all')
 
        if plot_coln is None:
            col_keys = None
        else:
            col_keys =mdex.unique(plot_coln).tolist()
        row_keys = mdex.unique(plot_rown).tolist()
 
        fig, ax_d = self.get_matrix_fig(row_keys, col_keys,
                                    figsize_scaler=4,
                                    constrained_layout=True,
                                    sharey=sharey,sharex=sharex,  
                                    fig_id=0,
                                    set_ax_title=True,figsize=figsize
                                    )
 
        fig.suptitle(title)
 
        #=======================================================================
        # loop and plot------
        #=======================================================================
        
        for gkeys, gsx0 in serx.groupby(level=gcols):
            keys_d = dict(zip(gcols, gkeys))
            log.info('on %s w/ %i'%(keys_d, len(gsx0)))
            ax = ax_d[keys_d[plot_rown]][keys_d[plot_coln]]
            #===================================================================
            # data prep
            #===================================================================
            xar = gsx0.index.get_level_values(resCn).values #resolutions
            yar = gsx0.values
            color=color_d[keys_d[plot_colr]]
            #===================================================================
            # plot
            #===================================================================
            if plot_type=='line':
            
                ax.plot(xar, yar, color=color,label =keys_d[plot_colr], **plot_kwargs)
            else:
                raise IOError(plot_type)
            
            #===================================================================
            # format
            #===================================================================
            #chang eto log scale
            ax.set_xscale(xscale)
            
            if not xlims is None:
                ax.set_xlim(xlims)
            
            
        #===============================================================
        # #wrap format subplot
        #===============================================================
        """best to loop on the axis container in case a plot was missed"""
        for row_key, d in ax_d.items():
            for col_key, ax in d.items():
                # first row
                if row_key == row_keys[0]:
                    #last col
                    if col_key == col_keys[-1]:
                        ax.legend()
                    
                # first col
                if col_key == col_keys[0]:
                    ax.set_ylabel(ylabel)
                
                #last row
                if row_key == row_keys[-1]:
                    ax.set_xlabel(xlabel)
 
                ax.grid()
        
 
        #=======================================================================
        # wrap
        #=======================================================================
        
        fname = 'StatVsReso_%s_%s' % (title,  self.longname)
        
        if write: 
            self.output_fig(fig, fname=fname)
 
        return ax_d
    
    def plot_statVsIter(self, #multi-stat against resolution (or agglevel) plots
                         #data
                         dx_raw=None, #combined model results
                         coln_l = [], #list of dx_raw colums to plot
                         ax_d=None,
                         xvar=None, #iterating variable for x-axis
                         
                         #plot control
                        plot_type='line', 
                        
                        plot_coln='studyArea',
                        
                        plot_colr=None,
                        plot_bgrp='dsampStage', #sub-group onto an axis
                        
                        
                           colorMap=None,color_d=None,
                           marker_d=None,
                         plot_kwargs = dict(alpha=0.8),
                         title=None,xlabel=None,
                         ylab_l=None,
                         xscale='log',
                         xlims=None,
                         ax_title_d=None, #optional axis titles (columns)
                         
                         #legend customization
                         legLab_func=None, #optional legeend label customizer
                         legend_kwargs={},
                         #ascending=False,
                         
                         #plot control [matrix]
                         figsize=None,grid=True,
                         sharey='none',sharex='col',
                         set_ax_title=True,
                         
                         
                         logger=None, write=None):
        #=======================================================================
        # defaults
        #=======================================================================
        plot_rown='columns'
        if logger is  None: logger=self.logger
        log = logger.getChild('plot_statVsIter')
        resCn, saCn = self.resCn, self.saCn
        if write is None: write=self.write
        if xvar is None: xvar=resCn
        
        if plot_colr is None: plot_colr=plot_bgrp
        if colorMap is None: colorMap=self.colorMap_d[plot_colr]
        
        if xlabel is None: xlabel = xvar
 
        if ylab_l is None:
            ylab_l = coln_l
        #=======================================================================
        # retrival
        #=======================================================================
        if dx_raw is None: 
            dx_raw = self.retrieve('catalog').loc[:, idx['depth', :]].droplevel(0, axis=1)
        """
        view(serx)
        dx_raw.index.names
        """
 
        log.info("%s w/ %s"%(title, str(dx_raw.shape)))
        #=======================================================================
        # precheck
        #=======================================================================
        for c in coln_l:
            assert c in dx_raw.columns, c
 
        
        #=======================================================================
        # data prep----
        #=======================================================================
        log.info('on %i'%len(dx_raw))
        
        #reorder indexes for nice sequencing
        #=======================================================================
        # head_l = [xvar, plot_bgrp]
        # l = head_l + list(set(dx_raw.index.names).difference(head_l))
        # dx = dx_raw.reorder_levels(l).sort_index(ascending=ascending).loc[:, coln_l]
        #=======================================================================
        dx = dx_raw.loc[:, coln_l]
        
 
        
        if dx.isna().any().any():
            log.warning("got %i/%i null values"%(dx.isna().sum().sum(), dx.size))
        
        #promote for consistent indexing
        serx = dx.stack(dropna=False) #usually no nulls... but sometimes for dummy data
        serx.index.set_names(list(dx.index.names)+['columns'], inplace=True)
        

        
        mdex = serx.index
        
        #construct grouping columns
        gcols = set()
        for c in [plot_bgrp, plot_coln, plot_rown]:
            if (not c is None): 
                gcols.add(c)
                assert c in mdex.names, c
        gcols = list(gcols)
 
        #=======================================================================
        # plot defaults
        #=======================================================================
        #title
        if title is None:
            title='%i vs. %s'%(len(coln_l), xvar)
            
        #get colors
        ckeys = mdex.unique(plot_colr)
        if color_d is None:
            
            
            #===================================================================
            # """nasty workaround to get colors to match w/ hyd""" 
            # if plot_colr =='dsampStage':
            #     ckeys = ['none'] + ckeys.values.tolist() #['none', 'post', 'postFN', 'pre', 'preGW']
            # elif plot_colr=='downSampling':
            #     ckeys = ['nn', '0', 'Average', '1', '2']
            #===================================================================
            
            color_d = self.get_color_d(ckeys, colorMap=colorMap, plot_colr=plot_colr)
            
        #check the colors
        miss_l = set(ckeys).symmetric_difference(color_d.keys())
        assert len(miss_l)==0
            
        #markers
        if marker_d is None:
            marker_d = {k:plt.Line2D.filled_markers[i] for i, k in enumerate(ckeys)}
            
        miss_l = set(ckeys).symmetric_difference(marker_d.keys())
        assert len(miss_l)==0
        
        #=======================================================================
        # setup the figure
        #=======================================================================
        
        col_keys =mdex.unique(plot_coln).tolist()
        row_keys = mdex.unique(plot_rown).tolist()
        
        #axis titles
        if ax_title_d is None: ax_title_d={k:k for k in col_keys}
        for k in ax_title_d.keys(): assert k in col_keys, 'bad ax_title_d key:%s'%k

        
        if ax_d is None:
            plt.close('all')

        
            fig, ax_d = self.get_matrix_fig(row_keys, col_keys,
                                        figsize_scaler=4,
                                        constrained_layout=True,
                                        sharey=sharey,sharex=sharex,  
                                        fig_id=0,
                                        set_ax_title=set_ax_title,figsize=figsize
                                        )
        
        else:
            
            assert len(row_keys)==len(ax_d), row_keys
            #retrieve the figure and the old column keys
            old_col_keys=list()
            for i, (k,d) in enumerate(ax_d.items()):
                assert len(d)==len(col_keys)
 
                for j, (k1, ax) in enumerate(d.items()):
                    fig = ax.figure
                    #dnew[row_keys[i]][col_keys[j]]=ax
                    old_col_keys.append(k1)
                break
                    
            #rekey the container
            colMap_d=dict(zip(old_col_keys, col_keys))
            rowMap_d=dict(zip(ax_d.keys(), row_keys))
            
            ax_d = {rowMap_d[k0]:{colMap_d[k1]:ax for k1,ax in d.items()} for k0,d in ax_d.items()}
            
 
        fig.suptitle(title)
        """
        fig.show()
        """
        #=======================================================================
        # loop and plot------
        #=======================================================================
        
        for gkeys, gsx0 in serx.sort_index(level=plot_bgrp, sort_remaining=False, ascending=False
                                           ).groupby(level=gcols):
            keys_d = dict(zip(gcols, gkeys))
            log.debug('on %s w/ %i'%(keys_d, len(gsx0)))
            ax = ax_d[keys_d[plot_rown]][keys_d[plot_coln]]
            #===================================================================
            # data prep
            #===================================================================
            xar = gsx0.index.get_level_values(xvar).values #resolutions
            yar = gsx0.values
            color=color_d[keys_d[plot_colr]]
            #===================================================================
            # plot
            #===================================================================
            if plot_type=='line':
                 
                ax.plot(xar, yar, color=color,label =keys_d[plot_colr],
                        marker=marker_d[keys_d[plot_colr]], **plot_kwargs)
            else:
                raise IOError(plot_type)
            
            #===================================================================
            # format
            #===================================================================
            #chang eto log scale
            ax.set_xscale(xscale)
            
            if not xlims is None:
                ax.set_xlim(xlims)
            
            
        #===============================================================
        # #wrap format subplot------
        #===============================================================
        """best to loop on the axis container in case a plot was missed"""
        for row_key, d in ax_d.items():
            for col_key, ax in d.items():
                # first row
                if row_key == row_keys[0]:
                    if not set_ax_title:
                        ax.set_title(ax_title_d[col_key])
                     
                    #last col
                    if col_key == col_keys[-1]:
                        ax.legend(**legend_kwargs)
                        
                        #customize legend
                        if not legLab_func is None:
                            handles, labels = ax.get_legend_handles_labels()
 
                            l2 = [legLab_func(e) for e in labels]
                        
                            ax.legend(handles, l2)
                    
                # first col
                if col_key == col_keys[0]:
                    ax.set_ylabel(ylab_l.pop(0))
                
                #last row
                if row_key == row_keys[-1]:
                    ax.set_xlabel(xlabel)
 
                if grid: ax.grid()
        
 
        #=======================================================================
        # wrap
        #=======================================================================
        
        fname = 'statVsIter_%s_%s' % (title,  self.longname)
        
        if write: 
            self.output_fig(fig, fname=fname)
 
        return ax_d
    #===========================================================================
    # helpers-------
    #===========================================================================
    def get_raster_data(self,  #retrieve and process data on a set of rasters
                        fp_d,
                        min_cell_cnt=1,
                        drop_zeros=True, 
                        debug_max_len=None,   
 
                        logger=None):
        
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger=self.logger
        log=logger.getChild('get_raster_data')
        assert isinstance(min_cell_cnt, int), 'must specify an int'
        stats_d = dict()
        data_d = dict()
        zcnt = 0
        drop_cnt = 0
        meta_d={'drop_zeros':drop_zeros}
        
        #===================================================================
        # loop and ubild
        #===================================================================
        log.info('looping on %i rasters' % (len(fp_d)))

        for gkey, fp in fp_d.items(): #typically gkey=resolution
            
            #===============================================================
            # get the values
            #===============================================================
            ar_raw = rlay_to_array(fp)
            ser1 = pd.Series(ar_raw.reshape((1, -1))[0]).dropna()
 
            stats_d[gkey] = {'%s_raw' % k:getattr(ser1, k)() for k in ['mean', 'max', 'count', 'min']}
                
            #===================================================================
            # #remove zeros
            #===================================================================
            bx = ser1 == 0.0
            if drop_zeros:
                ser1 = ser1.loc[~bx]
                stats_d[gkey].update({'zero_count':bx.sum()})
 
                stats_d[gkey].update({'%s_noZeros' % k:getattr(ser1, k)() for k in ['mean', 'max', 'count', 'min']})
                
            #===================================================================
            # #count check
            #===================================================================
            zcnt += bx.sum()
 
            if not len(ser1) > min_cell_cnt:
                log.warning('got %i entries (<%i)... skipping' % (len(ser1), min_cell_cnt))
                drop_cnt += 1
                continue
        
 
            #===================================================================
            # #reduce
            #===================================================================
            if not debug_max_len is None:
                if len(ser1) > debug_max_len:
                    log.warning('reducing from %i to %i' % (len(ser1), debug_max_len))
 
                    ser1 = ser1.sample(int(debug_max_len)) #get a random sample of these
                    
            #===================================================================
            # wrap
            #===================================================================
            assert len(ser1)>0
                
            data_d[gkey] = ser1.values
        
        #===================================================================
        # post
        #===================================================================
        assert len(data_d)>0
        meta_d.update({'zero_count':zcnt, 'lay_drop_cnt':drop_cnt})
        
        log.info('finished on %i w/ \n    %s'%(len(data_d), meta_d))
        
 
        return  data_d, pd.DataFrame.from_dict(stats_d).T, meta_d


        
    
    

#===============================================================================
# runners--------
#===============================================================================
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
    
    with RasterPlotr(tag=tag, overwrite=overwrite,  transparent=transparent, plt=plt, 
                        
                       bk_lib = {
                           'catalog':dict(catalog_fp=catalog_fp),
                           
                           },
                 **kwargs) as ses:
        
        #=======================================================================
        # compiling-----
        #=======================================================================
 
        ses.compileAnalysis()
 
        #=======================================================================
        # PLOTS------
        #=======================================================================
        #change order
        ax_title_d = ses.ax_title_d
        #del ax_title_d['LMFRA']
        #del ax_title_d['noise']
        dx_raw = ses.retrieve('catalog').loc[idx[list(ax_title_d.keys()), :], :]
        """
        view(dx_raw)
        dx_raw.columns
        """
        #drop the difference stats (so we get unique selections)
        dx_raw = dx_raw.drop('rstatsD', axis=1, level=0)
 
        #resolution filtering

        
        #=======================================================================
        # #manuscript filter 1
        #=======================================================================
        
        hi_res=10**3 #max we're using for hyd is 300
        #dx1 = dx_raw.loc[dx_raw.index.get_level_values('resolution')<=hi_res, :]
        dx1=dx_raw
        
        #dx2=dx1
        dx2 = dx1.loc[dx1.index.get_level_values('dsampStage')!='pre', :]
        
        dx3 = dx2.loc[dx2.index.get_level_values('studyArea')!='noise', :]
        
        #=======================================================================
        # manu filter 2
        #=======================================================================
        
        
        
        figsize=(7,7)
        #figsize=(12,12)
        for plotName, dxi, xlims,  ylims,xscale, yscale, drop_zeros in [
            #('',dx_raw, None,None, 'log', 'linear', True),
            ('filter1',dx3, None,None, 'log', 'linear', True),
 
            #('hi_res',hr_dx, (10, hi_res),None, 'linear', 'linear', True),
            #('hi_res_2',hr_dx, (10, hi_res),(-0.1,1), 'linear', 'linear', False),
            ]:
            #paramteer combinations to plot over
            """these are more like 'scenarios'... variables of secondary interest"""
            
            #===================================================================
            # schemes and methods
            #===================================================================
            plot_coln=ses.saCn
            xvar = ses.resCn
 
            for plot_bgrp in [ #variable to compare on one plot
                'downSampling', 
                #'dsampStage',
                #'sequenceType',
                ]:
                
                #variables to slice for each plot
                gcols_l=list(set(dxi.index.names).difference([plot_bgrp, plot_coln, xvar]))
                
                for gkeys, gdx in dxi.droplevel(0, axis=1).groupby(level=gcols_l):
                    
                    #check if theres multiple dimensions
                    if len(gdx.index.unique(plot_bgrp))==1:
                        continue #not worth plotting

                    #===============================================================
                    # prep
                    #===============================================================
                    keys_d = dict(zip(gcols_l,gkeys))
                    title = '_'.join([plotName]+list(gkeys))
                    
 
                    print('\n\n %s \n\n'%(title))
 
 
                    #nice plot showing the major raw statistics 
                    col_d={
                        #===============================================================
                        # 'MAX':'max depth (m)',
                        # 'MIN':'min depth (m)',
                         'MEAN':'global mean depth (m)',
                        #===============================================================
                        'wetMean':'wet mean depth (m)',
                        #'wetVolume':'wet volume (m^3)', 
                         'wetArea': 'wet area (m^2)', 
                         'rmse':'RMSE (m)', #total.. no wet
                         #'gwArea':'gwArea',
                         #'STD_DEV':'stdev (m)',
                         #'noData_cnt':'noData_cnt',
                         #'noData_pct':'no data (%)'
                    
                          }
                    
                    #filter titles
                    at_d = {k:v for k,v in ax_title_d.items() if k in gdx.index.get_level_values('studyArea')}
     
                    #plot caller
                    ses.plot_statVsIter(
                        plot_bgrp=plot_bgrp,figsize=figsize,
                        set_ax_title=False,
                        dx_raw=gdx, 
                        coln_l=list(col_d.keys()), xlims=xlims,
                        ylab_l = list(col_d.values()), ax_title_d=at_d,
                        title=title)
                    
 
            
 
 
        #===================================================================
        # value distributions----
        #===================================================================
        #couldnt get this to be useful w/o droppping zeros (which corrupts the results)
        #=======================================================================
        # ses.plot_rvalues(figsize=(12,12), plot_type='gaussian_kde', drop_zeros=False,  write_stats=False,
        #                 #hrange=(0,10),  xlims=(0,10), 
        #                 ylims=(0,0.1),
        #                  #linewidth=0.75,
        #                  )
        #=======================================================================
 
        
        #=======================================================================
        # differences-----------
        #=======================================================================
        ylab='depth difference (agg - true; m)'
        ctup = ('diff','difrlay_fp')
        #===================================================================
        # #population and mean combo w/ box plots
        #===================================================================
        #(not interpretable... to omany zeros)
        #=======================================================================
        # ses.plot_rValsVs(fp_serx =  hr_dx.loc[:, ctup],figsize=(12,12),
        #                  xlims=(0,hi_res), yscale='linear',ylab=ylab,
        #                  plot_types=['zero_line','line', 'box',],
        #                  #ax_d=ax_d,                     
        #                  rkwargs = dict(debug_max_len=None,min_cell_cnt=1,drop_zeros=False),
        #                  plot_kwargs = dict(linestyle='dashed'), 
        #                  title='Depth Differences'
        #                  )
        #=======================================================================
        
        #line only (similar to StatsVsResolution... but slower and consistent w/ box ploters)
        #=======================================================================
        # ses.plot_rValsVs(fp_serx =  hr_dx['rlay_fp'],figsize=(12,12),
        #                  ylims=None,yscale='linear',
        #                  plot_types=['line'],
        #                  #ax_d=ax_d,                     
        #                  rkwargs = dict(debug_max_len=None,min_cell_cnt=1,drop_zeros=False),
        #                  plot_kwargs = dict(linestyle='solid', marker='x'), 
        #                  title='Raw Depths (hi-res)'
        #                  )
        #=======================================================================
        
        #line only (full)
        #=======================================================================
        # ses.plot_rValsVs(fp_serx =  dx_raw['rlay_fp'],figsize=(12,12),
        #                  ylims=None,yscale='linear',xscale='log',
        #                  plot_types=['line'],
        #                  #ax_d=ax_d,                     
        #                  rkwargs = dict(debug_max_len=None,min_cell_cnt=1,drop_zeros=False),
        #                  plot_kwargs = dict(linestyle='solid', marker='x'), 
        #                  title='Raw Depths'
        #                  )
        #=======================================================================
        

        
        


        
        out_dir = ses.out_dir
    return out_dir

 
 
    
 

def r7():
    return run(tag='r7',catalog_fp=r'C:\LS\10_OUT\2112_Agg\lib\hr7\hr7_run_index.csv',)

def r8():
    return run(tag='r8',catalog_fp=r'C:\LS\10_OUT\2112_Agg\lib\hr8\hr8_run_index.csv',)

def r01():
    return run(tag='r01',catalog_fp=r'C:\LS\10_OUT\2112_Agg\lib\hydR01\hydR01_run_index.csv',)

def r02():
    return run(tag='r02',catalog_fp=r'C:\LS\10_OUT\2112_Agg\lib\hydR02\hydR02_run_index.csv',)

def r03():
    return run(tag='r03',catalog_fp=r'C:\LS\10_OUT\2112_Agg\lib\hydR03\hydR03_run_index.csv',)

if __name__ == "__main__": 
    #wet mean

    r03()
 

    tdelta = datetime.datetime.now() - start
    print('finished in %s' % (tdelta))
    