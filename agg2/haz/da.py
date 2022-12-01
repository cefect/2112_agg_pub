'''
Created on Aug. 30, 2022

@author: cefect
'''
import numpy as np
 
import pandas as pd
import os, copy, datetime, pprint
idx= pd.IndexSlice



from agg2.haz.scripts import UpsampleSession, assert_dx_names
from agg2.coms import Agg2DAComs, cat_mdex
from hp.plot import view


def now():
    return datetime.datetime.now()






class UpsampleDASession(Agg2DAComs, UpsampleSession):
    """dataanalysis of downsampling"""

    def __init__(self,  obj_name='ups', logfile_duplicate=False,**kwargs):
 
 
 
        super().__init__(obj_name=obj_name, logfile_duplicate=logfile_duplicate, **kwargs)
 
        
    def join_stats(self,fp_lib, **kwargs):
        """merge results from run_stats for different methodss and clean up the data"""
        log, tmp_dir, out_dir, ofp, resname, write = self._func_setup('jstats',  subdir=False,ext='.xls', **kwargs)
        
        #=======================================================================
        # preckec
        #=======================================================================
        for k1, d in fp_lib.items():
            for k2, fp in d.items():
                assert os.path.exists(fp), '%s.%s' % (k1, k2)
        
        #=======================================================================
        # loop and join
        #=======================================================================
        res_lib = dict()
        for k1, fp_d in fp_lib.items():
            res_d = dict() 
            for k2, fp in fp_d.items():
                
                dxcol_raw = pd.read_pickle(fp)            
                log.info('for %s.%s loading %s' % (k1, k2, str(dxcol_raw.shape)))
                
                # check
                assert_dx_names(dxcol_raw, msg='%s.%s' % (k1, k2))
                
                res_d[k2] = dxcol_raw
        
            #===================================================================
            # wrap reference
            #===================================================================
            res_lib[k1] = pd.concat(res_d, axis=1, names=['base'])            
        
        #=======================================================================
        # #concat
        #=======================================================================
        rdxcol = pd.concat(res_lib, axis=1, names=['method']
                   ).swaplevel('base', 'method', axis=1).sort_index(axis=1).sort_index(axis=0)
        
        #=======================================================================
        # #relabel all
        #=======================================================================
        idf = rdxcol.columns.to_frame().reset_index(drop=True)
        idf.loc[:, 'dsc'] = idf['dsc'].replace({'all':'full'})
        rdxcol.columns = pd.MultiIndex.from_frame(idf)
        
        #=======================================================================
        # write
        #=======================================================================
        if write:
            with pd.ExcelWriter(ofp, engine='xlsxwriter') as writer: 
                rdxcol.to_excel(writer, sheet_name='stats', index=True, header=True)
            log.info('wrote %s to \n    %s' % (str(rdxcol.shape), ofp))
        #=======================================================================
        # wrap
        #=======================================================================
        metric_l = rdxcol.columns.get_level_values('metric').unique().to_list()
        log.info('finished on %s w/ %i metrics \n    %s' % (str(rdxcol.shape), len(metric_l), metric_l))
        
        return rdxcol
    
    def get_normd(self,dx_raw,
                 to_be_normd='s12',
                 ):
        
        """probably some way to do this natively w/ panda (transform?)
        but couldnt figure out how to divide across 2 levels
        
        made a functio nto workaround the access violation
        """
        
        base_serx = dx_raw.loc[1, idx['s1', 'direct',:, 'full',:]].droplevel((0, 1, 3), axis=0)  # baseline values
        
        #add catMosaic
        #dx_raw.loc[1, idx['s2', 'direct',:, 'full',:]].droplevel((0, 1, 3), axis=0)
        
        d = dict()
        for layName, gdx in dx_raw[to_be_normd].groupby('layer', axis=1):
 
            d[layName] = gdx.droplevel('layer', axis=1).divide(base_serx[layName], axis=1, level='metric')
            
            """
            gdx.droplevel('layer', axis=1).loc[:, idx[:, :, 'real_count']]
            d[layName].loc[:, idx[:, :, 'real_count']]
            view()
            view(base_serx)
            view(gdx.droplevel('layer', axis=1))
            """
            
        # concat and promote
        div_dxcol = pd.concat(d, axis=1, names=['layer'])
        
        """
        view(base_serx)
        view(dx_raw[to_be_normd].loc[:, idx[:, 'wse', :,  ('real_count', 'real_area')]].sort_index(axis=1))
        view(div_dxcol.loc[:, idx['wse', :, :, ('real_count', 'real_area')]].sort_index(axis=1))
        """
        
        return pd.concat([div_dxcol], names=['base'], keys=[f'{to_be_normd}N'], axis=1).reorder_levels(dx_raw.columns.names, axis=1)
    
    def data_prep(self, dxcol_raw, **kwargs):
        """prepare data for plotting: compute secondary and normalized stats
        
        Notes
        -------
        had to make this a separate function to get around the memory exception
        """
        #===========================================================================
        # defaults
        #===========================================================================
        log, tmp_dir, out_dir, ofp, resname, write = self._func_setup('dprep',  subdir=False,ext='.pkl', **kwargs)
        idxn=self.idxn
        # add aggregated residuals
        """
        both these bases are stats computed on teh same (dynamic) zones:
            resid = stat[i=dsc@j_samp, j=j_samp] - stat[i=dsc@j_samp, j=j_base]
            
        
        
        """
        dxcol_raw['s2'].columns.names
        
        dx1a = dxcol_raw.join(pd.concat([dxcol_raw['s2'].drop('catMosaic', axis=1, level='layer') - dxcol_raw['s1']], names=['base'], keys=['s12A'], axis=1))
     
        #print({lvlName:i for i, lvlName in enumerate(dx1.columns.names)})
        l = [dx1a]
        #=======================================================================
        # area and volume
        #=======================================================================
        base_pixelLength = 1.0
        scale_df = pd.concat([
            pd.Series(dx1a.index), 
            pd.Series(dx1a.index * base_pixelLength, name='pixelLength', dtype=float), 
            pd.Series(dx1a.index * (base_pixelLength ** 2), name='pixelArea', dtype=float)], 
            axis=1).set_index(idxn)
            
        #volume
        """these are all at base resolution"""
        l.append(
            dx1a.loc[:, idx[:, :, 'wd', :, 'sum']].drop('s12_TP', axis=1, level='base'
                                #).multiply(scale_df['pixelArea'], axis=0
                                           ).rename(columns={'sum':'vol'}, level='metric'))
        #area
        l.append(
            dx1a.loc[:, idx[:, :, 'wse', :, 'real_count']].drop('s12_TP', axis=1, level='base'
                          # ).multiply(scale_df['pixelArea'], axis=0
                      ).rename(columns={'real_count':'real_area'}, level='metric'))
        
        dx1b = pd.concat(l, axis=1).sort_index(axis=1) #include these in the norm calcsl
        
        log.info(f'added area and volume w/ {str(dx1b.shape)}')
        l = [dx1b]
        #=======================================================================
        # residuals and norming
        #=======================================================================
        #baseline mean
        #base_ser = dx1a['s1']['direct'].loc[:, idx[:, 'full', 'mean']].droplevel(['dsc', 'metric'], axis=1).iloc[0, :].rename('base')
        
        #normalize aggregates
        """wse real_count not right
        base_ser = dx1a['s1']['direct']['wse'].loc[:, idx[:, 'full', 'mean']].droplevel(['dsc', 'metric'], axis=1).iloc[0, :].rename('base')
        
        dx1a['s12A']['filter']['wse'].loc[:, idx[:, 'real_count']]
        
        
        view(dx1b.T)
        
        """
        #norm globals
        l.append(self.get_normd(dx1b, to_be_normd='s12A'))
        
        
        #normalize locals
        """only these metrics make sense to normalize on the diffs"""
        l.append(
            self.get_normd(dx1b, to_be_normd='s12').drop(['RMSE', 'real_count'], axis=1, level='metric', errors='ignore'))
        #=======================================================================
        # merge
        #=======================================================================
        dx2 = pd.concat(l, axis=1).sort_index(axis=1)
        
        dx2.index = pd.MultiIndex.from_frame(scale_df.reset_index())
        
        
        #=======================================================================
        # write
        #=======================================================================
        if write: 
            dx2.to_pickle(ofp)             
            log.info(f'wrote {str(dx2.shape)} to \n    {ofp}')
        
        mdex = dx2.columns
        names_d = {name:mdex.unique(name).to_list() for name in mdex.names}
            
        log.info('assembled w/ \n%s'%pprint.pformat(names_d, width=10, indent=3, compact=True, sort_dicts =False))
        
        return dx2
    
 

 
    
    
