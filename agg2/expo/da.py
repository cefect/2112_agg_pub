'''
Created on Sep. 10, 2022

@author: cefect
'''
import os, copy, datetime, logging
import numpy as np
 
import pandas as pd
from pandas.testing import assert_index_equal

idx= pd.IndexSlice

#from definitions import max_cores
#from multiprocessing import Pool

#from agg2.haz.coms import coldx_d, cm_int_d
from hp.pd import append_levels

 


#from agg2.haz.da import UpsampleDASession
from agg2.expo.scripts import ExpoSession
 
from agg2.coms import Agg2DAComs, log_dxcol
from hp.plot import view


def now():
    return datetime.datetime.now()


class ExpoDASession(Agg2DAComs, ExpoSession):
 
    def __init__(self,scen_name='expo_da',  **kwargs): 
        super().__init__(scen_name=scen_name, **kwargs)
 
    
    def join_layer_samps(self,fp_lib,
                         dsc_df=None,
                         **kwargs):
        """assemble resample class of assets
        
        this is used to tag assets to dsc for reporting.
        can also compute stat counts from this
        """
        
        #=======================================================================
        # defaults
        #=======================================================================
        log, tmp_dir, out_dir, ofp, resname, write = self._func_setup('lsampsK',  subdir=True,ext='.pkl', **kwargs)
        
        res_d = dict()
        for method, d1 in fp_lib.items():
            d = dict()
            for layName, fp in d1.items():
                if not layName in ['wd', 'wse', 'dem']: continue        
 
                d[layName] = pd.read_pickle(fp)
                """
                pd.read_pickle(fp).hist()
                """
                
            #wrap method
            res_d[method] = pd.concat(d, axis=1).droplevel(0, axis=1) #already a dx
            
        #wrap
        dx1 =  pd.concat(res_d, axis=1, names=['method']).sort_index(axis=1)
 
        
        return dx1
    
    def build_samps(self, fp_lib, **kwargs):
        #=======================================================================
        # defaults
        #=======================================================================
        log, tmp_dir, out_dir, ofp, resname, write = self._func_setup('bsamps',  subdir=True,ext='.pkl', **kwargs)
        # get the rsc for each asset and scale
        
        # join the simulation results (and clean up indicides)
        samp_dx_raw = self.join_layer_samps(fp_lib)  #samples for each method and grid
        """
    
        view(samp_dx_raw.head(100))
        
        samp_dx.loc[:, idx['filter', 'wse', (1, 8)]].hist()
        
        """
        #=======================================================================
        # compute exposures
        #=======================================================================
        """1=wet"""
        wet_bxcol = samp_dx_raw.loc[:, idx[:, 'wse', :]].droplevel('layer', axis=1).notna().astype(int)
        #check false negatives
        wet_falseNegatives_bx = (wet_bxcol.subtract(
            wet_bxcol.loc[:, idx[:, 1]].droplevel('scale', axis=1)
            ) == -1)
        
        if wet_falseNegatives_bx.any().any():
            log.info('got %i False Negative exposures\n ' % (
                    wet_falseNegatives_bx.sum().sum(), 
                    #wet_falseNegatives[wet_falseNegatives != 0]
                    ))
        #merge
        samp_dx = pd.concat([
            samp_dx_raw, 
            pd.concat({'expo':wet_bxcol}, axis=1, names=['layer']).swaplevel('layer', 'method', axis=1)
            ], axis=1).sort_index(axis=1, sort_remaining=True)
            
        log.info(f'built {str(samp_dx.shape)}')
        return samp_dx
    
    def build_stats(self, samp_dx, dsc_df, **kwargs):
        """calc sample stats
        
        unlike Haz, we are computing the stats during data analysis
        
        s2 is computed against the matching samples
        s1 is computed against the baseline samples
        
        """
        
        #=======================================================================
        # defaults
        #=======================================================================
        log, tmp_dir, out_dir, ofp, resname, write = self._func_setup('bstats',  subdir=True,ext='.pkl', **kwargs)
        pdc_kwargs = dict(axis=1, names=['base'])
        idxn = self.idxn
        
        lvls = ['base', 'method', 'layer', 'metric', 'dsc']
        
        def sort_dx(dx):
            return dx.reorder_levels(lvls, axis=1).sort_index(axis=1, sort_remaining=True)
        
        #=======================================================================
        # baseline values
        #=======================================================================
        samp_base_dx = samp_dx.loc[:, idx[:, :, 1]].droplevel('scale', axis=1)
        ufunc_d = {'expo':['sum'], 'wd':['mean', 'sum'], 'wse':['mean']}
        
        #get baseline stats
        d = dict()
        for layName, stat_l in ufunc_d.items():
            dxi = samp_base_dx.loc[:, idx[:, layName]].droplevel('layer', axis=1)
            
            d[layName] = pd.concat({stat:getattr(dxi, stat)() for stat in stat_l}, axis=1, names='metric')
            
 
        #expand to match
        s1_sdxi = pd.concat(d, axis=1, names=['layer']).unstack().rename(1).to_frame().T.rename_axis('scale')        
        s1_sdx = sort_dx(pd.concat(
            {'full':pd.concat({'s1':s1_sdxi, 's2':s1_sdxi, 's12':pd.DataFrame(0.0, index=s1_sdxi.index, columns=s1_sdxi.columns)}, axis=1, names=['base'])
            }, axis=1, names=['dsc']))
        
 
        
        #=======================================================================
        #ZONAL stats
        #=======================================================================
        res_d = dict()
        for scale, col in dsc_df.items(): #loop rsc values for each scale
            """loop and compute stats with different masks"""
            d = dict()
            
            samp_scale_dx = samp_dx.loc[:, idx[:, :, scale]].droplevel('scale', axis=1)
            #===================================================================
            # #s1 (calc against base)
            #===================================================================
            #join the base values to this index
            mdex = pd.MultiIndex.from_frame(samp_base_dx.index.to_frame().reset_index(drop=True).join(col.rename('dsc')))
            dxi = pd.DataFrame(samp_base_dx.values, index=mdex, columns=samp_base_dx.columns)
            d['s1'] = self.get_dsc_stats2(dxi, ufunc_d=ufunc_d)
            
            #===================================================================
            # s2
            #===================================================================
            #agg values to this index
            dxi = pd.DataFrame(samp_scale_dx.values, index=mdex, columns=samp_base_dx.columns)
            d['s2'] = self.get_dsc_stats2(dxi, ufunc_d=ufunc_d)
            
            #===================================================================
            # s12
            #===================================================================
            #agg minus base
            dxi = pd.DataFrame(
                samp_scale_dx.sub(samp_base_dx).values, 
                index=mdex, columns=samp_base_dx.columns)
            
            #loop on each method to force exposure matching (because exposure changes between methods)
            meth_d = dict()
            for method, gdx in dxi.groupby('method', axis=1):
                
                #require exposure on both grids
                wet_bx = np.logical_and(
                    samp_scale_dx[method]['expo'].astype(bool),
                    samp_base_dx[method]['expo'].astype(bool)
                    )
                
                assert wet_bx.any(), f'none on {scale} {method}'
                meth_d[method] = self.get_dsc_stats2(gdx.loc[wet_bx.values, :], ufunc_d=ufunc_d)
                
            #join methods back together            
            d['s12'] = pd.concat(meth_d.values(), axis=1)
            """
            mdx = pd.concat(
                {'s2':samp_dx.loc[:, idx[:, :, scale]].droplevel('scale', axis=1),
                's1':samp_base_dx,
                's12':dxi.droplevel('dsc')}, axis=1)
            
 
            view(mdx.loc[:, idx[:, 'direct', 'wd']].head(100))
            """
            #===================================================================
            # wrap
            #===================================================================
            res_d[scale] = pd.concat(d, axis=1, names=['base'])
        
        #wrap
        sdx1 = sort_dx(pd.concat(res_d, axis=1, names=['scale']).stack('scale').unstack('dsc'))
        sdx2 = pd.concat([sdx1, s1_sdx]).sort_index() #add baseline (scale=1)
        
        """
        view(sdx2.drop('filter', level='method', axis=1).loc[:, idx[:, :, 'expo', :, 'full']])
        
        view(sdx2.loc[:, idx[:, 'direct', :,:, 'full']].T)
        """
        #=======================================================================
        # aggregate errors
        #=======================================================================
        srdx = sdx2['s2'].subtract(sdx2['s1']) #ggregated stats
        
        #=======================================================================
        # #normalize
        #=======================================================================
        base_serx1 = s1_sdxi.iloc[0, :].reorder_levels(['method', 'layer', 'metric']).sort_index()
        
        #=======================================================================
        # dsc counts
        #=======================================================================
        """
        add the classification counts onto s2
        """
 
        
        d = dict()
        for scale, col in dsc_df.items():
            d[scale] = col.value_counts()
        
        dx1 = pd.concat(d, axis=1).T
        dx1.columns.name = 'dsc'
        dx1.index.name=idxn
        
        dx1.columns = append_levels(dx1.columns, {'layer':'catMosaic', 'method':'direct', 'metric':'count', 'base':'s2'}
                                    ).reorder_levels(sdx2.columns.names)
                                    
        
        #=======================================================================
        # #combine everything
        #=======================================================================
        sdx3 = pd.concat([sdx2.join(dx1), 
                          pd.concat({
                                's12A':srdx, 's12AN':srdx.divide(base_serx1, axis=1), 's12N':sdx2['s12'].divide(base_serx1, axis=1)
                                }, **pdc_kwargs)
                          ], axis=1)
        

        
        #=======================================================================
        # write
        #=======================================================================
        log_dxcol(log, sdx3)
        if write: 
            sdx3.to_pickle(ofp)
            log.info(f'wrote {str(sdx3.shape)} to \n    {ofp}')
            
        return sdx3 
    

    
    def get_dsc_stats2(self, raw_dx,
                       ufunc_d = {'expo':['sum'], 'wd':['mean'], 'wse':['mean']},
                       **kwargs):
        log, tmp_dir, out_dir, ofp, resname, write = self._func_setup('dscStats',  subdir=True,ext='.pkl', **kwargs)
        
        gcols = ['method', 'layer']
        res_d = dict()
        for i, (gkeys, gdx) in enumerate(raw_dx.groupby(level=gcols, axis=1)):
            gkeys_d = dict(zip(gcols, gkeys))
            stat_l = ufunc_d[gkeys_d['layer']]
            
            d = dict()
            for stat in stat_l:
            
                #get zonal stats            
                grouper = gdx.groupby(level='dsc')            
                sdx = getattr(grouper, stat)()
                
                #get total stat
                sdx.loc['full', :] = getattr(gdx, stat)()
                
                d[stat] = sdx
            
            #wrap
            res_d[i] = pd.concat(d, axis=1, names=['metric'])
 
        return pd.concat(list(res_d.values()), axis=1).reorder_levels(list(raw_dx.columns.names) + ['metric'], axis=1)
    
 
   
    
    
        

    
    
    
    
