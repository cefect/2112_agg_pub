'''
Created on Sep. 9, 2022

@author: cefect

commons for all of Agg
'''
import os, datetime
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr

from hp.oop import Session, today_str
from hp.plot import Plotr

#integer maps for buildilng the mosaic
cm_int_d = {'DD':11, 'WW':21, 'WP':31, 'DP':41}

class AggBase(object):
    cm_int_d=cm_int_d
    """placeholder"""
    pass

class Agg2Session(Session):
    
    idxn = 'scale' #main index name
 
    def __init__(self,
                 case_name='SJ',
                 out_dir=None, fancy_name=None,
                 proj_name='agg2',
                 scen_name='direct', 
                 run_name='rName',
                 subdir=False,
                 wrk_dir=None,
                 
                 **kwargs):
        """handle project specific naming
        
        Parameters
        ----------
        case_name: str
            case study name
        
        scen_name: str
            scenario name (i.e., method)
            

 
        """
        if wrk_dir is None:
            from definitions import wrk_dir
            
        if out_dir is None:
            out_dir = os.path.join(wrk_dir, 'outs', proj_name, run_name, case_name,   scen_name, today_str)
            
        if fancy_name is None:
            fancy_name = '_'.join([case_name,run_name, scen_name, datetime.datetime.now().strftime('%m%d')])
        
        super().__init__( wrk_dir=wrk_dir, proj_name=proj_name,
                         out_dir=out_dir, fancy_name=fancy_name,subdir=subdir,run_name=run_name,
                         **kwargs)
        
        self.scen_name=scen_name
        self.case_name=case_name
        
    
    def get_ds_merge(self, xr_dir, **kwargs):
        """collect structured outputs into one datasource
        
        we have 4 data_vars
    
        all coords and dims should be the same
        
        files are split along the 'scale' coord
        
        """
        #=======================================================================
        # defaults
        #=======================================================================
        idxn=self.idxn
        log, tmp_dir, out_dir, ofp, resname, write = self._func_setup('s12_TP',  subdir=False,ext='.pkl', **kwargs)
        assert os.path.exists(  xr_dir), xr_dir

        #=======================================================================
        # load each subdir----
        #=======================================================================
        ds_d = dict()
        assert os.path.exists(xr_dir),xr_dir
        assert len(os.listdir(xr_dir))>0, 'no files in passed xr_dir:\n    %s'%xr_dir
        for dirpath, _, fns in os.walk(xr_dir):
            varName = os.path.basename(dirpath)
            fp_l = [os.path.join(dirpath, e) for e in fns if e.endswith('.nc')]
            if len(fp_l) == 0:
                continue
            ds_l = list()
            #load eaech
            for i, fp in enumerate(fp_l):
                ds_l.append(xr.open_dataset(fp, engine='netcdf4', chunks='auto', 
                        decode_coords="all"))
            
            ds_d[varName] = xr.concat(ds_l, dim=idxn)
        
        #merge all the layers
        ds = xr.merge(ds_d.values())
        log.info(
            f'loaded {ds.dims}' + f'\n    coors: {list(ds.coords)}' + f'\n    data_vars: {list(ds.data_vars)}' + f'\n    crs:{ds.rio.crs}')
        assert ds.rio.crs == self.crs, ds.rio.crs
        return ds

        

class Agg2DAComs(Plotr):
    """data analysis common to all"""
    
    
    colorMap_d = {
        'dsc':'PiYG'
        }
    
    color_lib = {
        'dsc':{  
                'WW':'#0000ff',
                'WP':'#00ffff',
                'DP':'#ff6400',
                'DD':'#800000',
                'full': '#000000'}
        }
    
    #order of column index names
    names_l = ['base', 'method', 'layer', 'dsc', 'metric']
    
    def __init__(self,
                 output_format = 'svg', 
                   **kwargs): 
        
        self.output_format=output_format
        super().__init__(logfile_duplicate=False,  **kwargs)
        
    def log_dxcol(self, *args):
        return log_dxcol(self.logger, *args)
 
    
    def plot_matrix_metric_method_var(self,
                                      serx,
                                      map_d={'row':'metric', 'col':'method', 'color':'dsc', 'x':'pixelLength'},
                                      title=None, colorMap=None, color_d=None,
                                      row_l = None,
                                      ylab_d={'vol':'$V_{s2}$ (m3)', 'wd_mean':r'$WD_{s2}$ (m)', 'wse_area':'$A_{s2}$ (m2)'},
                                      ax_title_d={'direct':'$WSH$ Averaging', 'filter':'$WSE$ Averaging'},
                                      ax_lims_d=dict(),
                                      xscale='linear',
                                      matrix_kwargs=dict(figsize=(6.5, 6), set_ax_title=False, add_subfigLabel=True, fig_id=0, constrained_layout=True),
                                      plot_kwargs_lib={
                                          'full':{'marker':'x'},
                                          'DD':{'marker':'s', 'fillstyle':'none'},
                                          'WW':{'marker':'v', 'fillstyle':'full'},
                                          'WP':{'marker':'o', 'fillstyle':'top'},
                                          'DP':{'marker':'o', 'fillstyle':'bottom'},
                                          },
                                      plot_kwargs={'linestyle':'solid', 'marker':'x', 'markersize':7, 'alpha':0.8},
                                      output_fig_kwargs=dict(),
                                      legend_kwargs=dict(),
                                      yfmt_func = lambda x,p:'%.2f'%x,
                                      xlab='resolution (m)',
                                      **kwargs):
        
        """build matrix plot of variance
            x:pixelLength
            y:(series values)
            rows: key metrics (wd_mean, wse_area, vol)
            cols: all methods
            colors: downsample class (dsc)
            
        Parameters
        -----------
        serx: pd.Series w/ multindex
            see join_stats
        map_d: dict
            plot matrix name dict mapping dxcol data labels to the matrix plot
            
        plot_kwargs_lib: {series name: **plot_kwargs}
            series specific kwargs
        
        plot_kwargs: dict
            kwargs for all series (gives up precedent to series specific)
            
        Note
        --------
        cleaner to do all slicing and data maniupation before the plotter
        
        """
        #=======================================================================
        # defaults
        #=======================================================================
        log, tmp_dir, out_dir, ofp, _, write = self._func_setup('metric_method_var',  subdir=False,ext='.svg', **kwargs)
 
            
        #=======================================================================
        # extract data
        #=======================================================================
        assert len(serx)>0
        assert serx.notna().any().any()
        
        map_d = {k:map_d[k] for k in ['row', 'col', 'color', 'x']} #ensure order on map
        
        """leaving data order passed by theuser"""
        serx = serx.reorder_levels(list(map_d.values()))#.sort_index(level=map_d['x']) #ensure order on data
        
        mdex = serx.index
        keys_all_d = {k:mdex.unique(v).tolist() for k,v in map_d.items()} #order matters
        
        #change row order
        """because pulling from teh index can give inconsistent order (if there are labels missing)"""
        if not row_l is None:
            assert set(keys_all_d['row']).symmetric_difference(row_l)==set(), 'bad row labels'
            keys_all_d['row'] = row_l
            
        
        #check the keys
        for k,v in keys_all_d.items():
            assert len(v)>0, k
        
        if color_d is None:
            color_d = self._get_color_d(map_d['color'], keys_all_d['color'], colorMap=colorMap, color_d=color_d)
        
        #plot kwargs
        """here we populate with blank kwargs to ensure every series has some kwargs"""
        if plot_kwargs_lib is None: plot_kwargs_lib=dict()
        for k in keys_all_d['color']:
            if not k in plot_kwargs_lib:
                plot_kwargs_lib[k] = plot_kwargs
            else:
                plot_kwargs_lib[k] = {**plot_kwargs, **plot_kwargs_lib[k]} #respects precedent
 
        #=======================================================================
        # setup figure
        #=======================================================================
        #plt.close('all')
 
 
        fig, ax_d = self.get_matrix_fig(keys_all_d['row'], keys_all_d['col'], 
                                    sharey='row',sharex='all',  
                                    logger=log, **matrix_kwargs)
 
        if not title is None:
            fig.suptitle(title)
        
        #=======================================================================
        # loop and plot
        #=======================================================================
        levels = [map_d[k] for k in ['row', 'col']]
        for gk0, gsx0 in serx.groupby(level=levels):
            #===================================================================
            # setup
            #===================================================================
            ax = ax_d[gk0[0]][gk0[1]]
            keys_d = dict(zip(levels, gk0))
            
            ax.set_xscale(xscale)
            #===================================================================
            # loop each series (color group)
            #===================================================================
            for gk1, gsx1 in gsx0.groupby(level=map_d['color']):
                keys_d[map_d['color']] = gk1
                xar, yar = gsx1.index.get_level_values(map_d['x']).values, gsx1.values
                #===============================================================
                # plot
                #===============================================================
                ax.plot(xar, yar, color=color_d[gk1],label=gk1,**plot_kwargs_lib[gk1])
                
        #=======================================================================
        # post format
        #=======================================================================
        for row_key, d in ax_d.items():
            for col_key, ax in d.items():
                ax.grid()
                
                #first row
                if row_key==keys_all_d['row'][0]:
                    if col_key in ax_title_d:
                        ax.set_title(ax_title_d[col_key])
                    
                    
                #first col
                if col_key == keys_all_d['col'][0]:
                    if row_key in ylab_d:
                        ax.set_ylabel(ylab_d[row_key])
                    else:
                        ax.set_ylabel(row_key)
                    
                    #set lims
                    if 'y' in ax_lims_d:
                        if row_key in ax_lims_d['y']:
                            ax.set_ylim(ax_lims_d['y'][row_key])
                            
                            """
                            plt.show()
                            """
                    
                    #force 2decimal precision
                    ax.get_yaxis().set_major_formatter(yfmt_func)
                    
                    #first row
                    if row_key==keys_all_d['row'][0]:
                        if not legend_kwargs is None:
                            ax.legend(**legend_kwargs)
                    
                
                #last row
                if row_key==keys_all_d['row'][-1]:
                    ax.set_xlabel(xlab)
                
                #last col
                if col_key == keys_all_d['col'][-1]:
                    pass
 
                        
        #=======================================================================
        # output
        #=======================================================================
        if write:
            return self.output_fig(fig, ofp=ofp, logger=log, **output_fig_kwargs)
        else:
            return fig, ax_d
    
    def plot_dsc_ratios(self, df,
                        colorMap=None,color_d=None,
                        ylim = (0.6,1.0),
                        xlim = (0, 500),
                        figsize=(6.5,2),
                        ylabel= 'domain fraction',
                        **kwargs):
        """plot ratio of dsc class vs. resolution
        
        Parameters
        ---------
        df: pd.DataFrame
            data to plot {index:scale, columns:dsc}
        """
        log, tmp_dir, out_dir, ofp, _, write = self._func_setup('dsc_rats',  subdir=False,ext='.svg', **kwargs)
        
        #=======================================================================
        # setup
        #=======================================================================
 
        coln = df.columns.name
        keys_all_d={'color':df.columns.tolist()}
        
        if color_d is None:
            color_d = self._get_color_d(coln, keys_all_d['color'], colorMap=colorMap, color_d=color_d)
        
        color_l = [color_d[k] for k in df.columns]
            
        #=======================================================================
        # setup plot
        #=======================================================================
        plt.close('all')
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
            
        #=======================================================================
        # loop and plot
        #=======================================================================
        ax.stackplot(df.index, df.T.values, labels=df.columns, colors=color_l,
                     alpha=0.8)
 
        
        #=======================================================================
        # format
        #=======================================================================
        ax.legend(loc=3)
        ax.set_xlabel('pixel size (m)')
        ax.set_ylabel(ylabel)
        ax.set_ylim(ylim)
        ax.set_xlim(xlim)
        
        #=======================================================================
        # output
        #=======================================================================
        return self.output_fig(fig, ofp=ofp, logger=log)
    
    
def log_dxcol(log, dxcol_raw):
    mdex = dxcol_raw.columns
    log.info(
        f'for {str(dxcol_raw.shape)}' + 
        '\n    base:     %s' % mdex.unique('base').tolist() + 
        '\n    layers:   %s' % mdex.unique('layer').tolist() + 
        '\n    metrics:  %s' % mdex.unique('metric').tolist())
    return 

def cat_mdex(mdex, levels=['layer', 'metric']):
    """concatnate two levels into one of an mdex"""
    mcoln = '_'.join(levels)
    df = mdex.to_frame().reset_index(drop=True) 

    df[mcoln] = df[levels[0]].str.cat(others=df.loc[:, levels[1:]], sep='_')
    
    return pd.MultiIndex.from_frame(df.drop(levels, axis=1)), mcoln
