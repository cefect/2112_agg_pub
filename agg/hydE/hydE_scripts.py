'''
Created on May 12, 2022

@author: cefect

scripts for running exposure calcs on hydR outputs

loop on studyarea (load all the rasters)
    loop on finv
        loop on raster
'''

#===============================================================================
# imports-----------
#===============================================================================
import os, datetime, math, pickle, copy, sys
import qgis.core
from qgis.core import QgsRasterLayer, QgsMapLayerStore, QgsWkbTypes
import pandas as pd
import numpy as np
from pandas.testing import assert_index_equal, assert_frame_equal, assert_series_equal


idx = pd.IndexSlice
from hp.exceptions import Error
from hp.basic import set_info
from hp.pd import get_bx_multiVal
import hp.gdal

from hp.Q import assert_rlay_equal, vlay_get_fdf
from hp.err_calc import ErrorCalcs

from agg.hyd.hscripts import Model, StudyArea, view, RasterCalc
from agg.hydR.hydR_scripts import RastRun

class ExpoRun(RastRun):
    phase_l=['depth', 'diff', 'expo']
    index_col = list(range(8))
    
    agCn='aggLevel'
    ridn='rawid'
 
    def __init__(self,
                 name='expo',
                 data_retrieve_hndls={},
                 pick_index_map={},phase_d={},
                 **kwargs):
        

        
        data_retrieve_hndls = {**data_retrieve_hndls, **{
            # aggregating inventories
            'finv_agg_lib':{  # lib of aggrtevated finv vlays
                'compiled':lambda **kwargs:self.load_layer_lib(**kwargs),  # vlays need additional loading
                'build':lambda **kwargs: self.build_finv_agg2(**kwargs),
                },
            
            'faggMap':{  #map of aggregated to raw finvs
                'compiled':lambda **kwargs:self.load_pick(**kwargs),  # vlays need additional loading
                'build':lambda **kwargs: self.build_faggMap(**kwargs),
                },
            
            'finv_sg_lib':{  # sampling geometry
                'compiled':lambda **kwargs:self.load_layer_lib(**kwargs),  # vlays need additional loading
                'build':lambda **kwargs: self.build_sampGeo2(**kwargs),
                },
                        
            'rsamps':{  # lib of aggrtevated finv vlays
                'compiled':lambda **kwargs:self.load_pick(**kwargs),  # vlays need additional loading
                'build':lambda **kwargs: self.build_rsamps2(**kwargs),
                },
            
            'rsampStats':{  # lib of aggrtevated finv vlays
                'compiled':lambda **kwargs:self.load_pick(**kwargs),  # vlays need additional loading
                'build':lambda **kwargs: self.build_rsampStats(**kwargs),
                },
            
            'rsampErr':{  # lib of aggrtevated finv vlays
                'compiled':lambda **kwargs:self.load_pick(**kwargs),  # vlays need additional loading
                'build':lambda **kwargs: self.build_rsampErr(**kwargs),
                },
            
            'assetCat':{  # lib of aggrtevated finv vlays
                'compiled':lambda **kwargs:self.load_pick(**kwargs),  # vlays need additional loading
                'build':lambda **kwargs: self.build_assetCat(**kwargs),
                },
                        
            }}
        
        pick_index_map.update({
            'finv_agg_lib':(self.agCn, self.saCn),
            })
        pick_index_map['finv_sg_lib'] = copy.copy(pick_index_map['finv_agg_lib'])
        
        
        #=======================================================================
        # phase
        #=======================================================================
        phase_d.update({
 
            'expo':('rsampStats','rsampErr')
            
            })
        
 
        
        
        super().__init__( pick_index_map=pick_index_map,phase_d=phase_d,
                         data_retrieve_hndls=data_retrieve_hndls, name=name,
                         **kwargs)
        
        self.rcol_l.append(self.agCn)
        
    def runExpo(self):
        
        #build the inventory (polygons)
        #self.retrieve('finv_agg_lib')
        self.retrieve('faggMap')
        
        #build the sampling geometry
        #self.retrieve('finv_sg_lib')
        
        #sample all the rasters
        #self.retrieve('rsamps')
        
        #get per-run stats
        self.retrieve('rsampStats')
        
        
        self.retrieve('rsampErr')
        

        #reshape for catalog
        #self.retrieve('assetCat')
        
    def build_finv_agg2(self,  # build aggregated finvs
                       dkey=None,
                       
                       # control aggregated finv type 
                       aggType=None,
                       
                       #aggregation levels
                       aggLevel_l=None,
                       iters=3, #number of aggregations to perform
                       agg_scale=4,
                       
                       #defaults
                       proj_lib=None,
                       write=None, logger=None,**kwargs):
        """
        wrapper for calling more specific aggregated finvs (and their indexer)
            filepaths_dict and indexer are copied
            layer container is not
            
        only the compiled pickles for this top level are required
            (lower levels are still written though)
        """
        
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger=self.logger
        log = logger.getChild('build_finv_agg')
        if write is None: write=self.write
 
        assert dkey in ['finv_agg_lib',
                        #'faggMap', #makes specifycing keys tricky... 
                        ], 'bad dkey: \'%s\''%dkey
 
        gcn = self.gcn
        saCn=self.saCn
        log.info('building \'%s\' ' % (aggType))
        
        
        #=======================================================================
        # clean proj_lib
        #=======================================================================
        if proj_lib is None:
            proj_lib = copy.deepcopy(self.proj_lib)
        for sa, d in proj_lib.items():
            for k in ['wse_fp_d', 'dem_fp_d']:
                if k in d:
                    del d[k]
                    
        #=======================================================================
        # build aggLevels
        #=======================================================================
        #[1, 4, 16]
        if aggLevel_l is None:
            aggLevel_l = [(agg_scale)**i for i in range(iters)]
        
        assert max(aggLevel_l)<1e3
        
        #=======================================================================
        # build aggregated finvs------
        #=======================================================================
        """these should always be polygons
        StudyArea.get_finv_agg_d()
        """
 
        res_d = self.sa_get(meth='get_finv_agg_d', write=False, dkey=dkey, get_lookup=True,
                             aggLevel_l=aggLevel_l,aggType=aggType, **kwargs)
        
 
        
        # unzip
        finv_gkey_df_d, finv_agg_d = dict(), dict()
        for studyArea, d in res_d.items():
            finv_gkey_df_d[studyArea], finv_agg_d[studyArea] = d['faMap_dx'], d['finv_d']
            
        assert len(finv_gkey_df_d) > 0, 'got no links!'
        assert len(finv_agg_d) > 0, 'got no layers!'
 
        #=======================================================================
        # check
        #=======================================================================
        #invert to match others
        dnew = {i:dict() for i in aggLevel_l}
        for studyArea, d in finv_agg_d.items():
            for aggLevel, vlay in d.items():
                """relaxing this
                assert vlay.wkbType()==3, 'requiring singleParts'"""
                assert 'Polygon' in QgsWkbTypes().displayString(vlay.wkbType())
                dnew[aggLevel][studyArea] = vlay
                
        finv_agg_d = dnew
        #=======================================================================
        # handle layers----
        #=======================================================================
        """saving write till here to match other functions
        might run into memory problems though....
        consider writing after each studyArea"""
        dkey1 = 'finv_agg_lib'
        if write:
            self.store_lay_lib(finv_agg_d,dkey1,  logger=log)
        
        self.data_d[dkey1] = finv_agg_d
        #=======================================================================
        # handle mindex-------
        #=======================================================================
        """creating a index that maps gridded assets (gid) to their children (id)
        seems nicer to store this as an index
        
        """
        #=======================================================================
        # assemble
        #=======================================================================
        
        dkey1 = 'faggMap'
        df = pd.concat(finv_gkey_df_d, verify_integrity=True, names=[self.saCn, self.ridn]) 
        
        df.columns.name=self.agCn
        #=======================================================================
        # check
        #=======================================================================
        
        #retrieve directly from layres
        mindex = self.get_indexers_from_layers(lay_lib=finv_agg_d, dkey=dkey)

 
        gb = mindex.to_frame().groupby(level=[saCn, 'aggLevel'])
        #check against previous
        for aggLevel, col in df.items():
            for studyArea, gcol in col.groupby(level=self.saCn):
                layVals = gb.get_group((studyArea, aggLevel)).index.unique(gcn).values
                d = set_info(gcol.values, layVals, result='counts')
                if not d['symmetric_difference']==0:
                    raise Error('%s.%s \n    %s'%(aggLevel, studyArea, d))
 
            
 
 
        #=======================================================================
        # write
        #=======================================================================
        # save the pickle
        if write:
            self.ofp_d[dkey1] = self.write_pick(df,
                           os.path.join(self.wrk_dir, '%s_%s.pickle' % (dkey1, self.longname)), logger=log)
        
        # save to data
        self.data_d[dkey1] = copy.deepcopy(df)
 
        return finv_agg_d
    #===========================================================================
    #     if dkey=='finv_agg_lib':
    #         return finv_agg_d
    #     elif dkey=='faggMap':
    #         return df.copy()
    #     else:
    #         raise KeyError(dkey)
    # 
    #===========================================================================
    def build_faggMap(self, #construct the map linking 
                        dkey='faggMap',
                        #finv_agg_lib=None,
                     #defaults
                     logger=None,write=None,
                     ):
        """typically this is constructed by build_finv_agg2"""
        #'no way to retrieve faggMap. this... need to reconstruct')
        #=======================================================================
        # defauts
        #=======================================================================
        assert dkey == 'faggMap'
        if logger is None: logger=self.logger
        log = logger.getChild('build_faggMap')
        if write is None: write=self.write
        
        assert not 'finv_agg_lib' in self.compiled_fp_d, 'having finv_agg_lib compiled and faggMap NOT compiled is not supported'
        assert not 'finv_agg_lib' in self.data_d
        self.retrieve('finv_agg_lib')
            
        return self.data_d[dkey].copy()
    
    def build_sampGeo2(self,  # sampling geometry no each asset
                     dkey='finv_sg_lib',
                     sgType='centroids',
                     
                     finv_agg_d=None,
                     
                     #defaults
                     logger=None,write=None,
                     ):
        """
        see test_sampGeo
        """
        #=======================================================================
        # defauts
        #=======================================================================
        assert dkey == 'finv_sg_lib'
        if logger is None: logger=self.logger
        log = logger.getChild('build_sampGeo')
        if write is None: write=self.write
        
        if finv_agg_d is None: finv_agg_d = self.retrieve('finv_agg_lib', write=write)
 
        #=======================================================================
        # loop each polygon layer and build sampling geometry
        #=======================================================================
        
        #retrieve indexers from layer
        def func(poly_vlay, logger=None, meta_d={}):
            assert 'Polygon' in QgsWkbTypes().displayString(poly_vlay.wkbType())
            if sgType == 'centroids':
                """works on point layers"""
                sg_vlay = self.centroids(poly_vlay, logger=log)
                
            elif sgType == 'poly':
                assert 'Polygon' in QgsWkbTypes().displayString(poly_vlay.wkbType()
                                                                ), 'bad type on %s' % (meta_d)
                poly_vlay.selectAll()
                sg_vlay = self.saveselectedfeatures(poly_vlay, logger=log)  # just get a copy
                
            else:
                raise Error('not implemented')
            
            sg_vlay.setName('%s_%s'%(poly_vlay.name(), sgType))
            return sg_vlay        
        
        
        lay_lib = self.calc_on_layers(lay_lib=finv_agg_d, func=func, subIndexer='aggLevel', 
                                  format='dict',write=False, dkey=dkey)
        
 
        #=======================================================================
        # store layers
        #=======================================================================
        if write: 
            ofp_d = self.store_lay_lib(lay_lib,dkey,  logger=log)
        
 
        
        return lay_lib
        
    def build_rsamps2(self,  # sample all the rasters and all the finvs
                       dkey='rsamps',
                       finv_agg_lib=None, drlay_lib=None,
                       
                       #parameters
                       samp_method='points',  # method for raster sampling
                     
                       #defaults
                       proj_lib=None,
                       write=None, logger=None,**kwargs):
        """
        see self.build_rsamps()
 
        """
        
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger=self.logger
        log = logger.getChild('build_rsamps2')
        if write is None: write=self.write
 
        assert dkey =='rsamps'
 
 
        log.info('building \'%s\' ' % (samp_method))
        
        #=======================================================================
        # retrieve
        #=======================================================================
        if finv_agg_lib is None:
            finv_agg_lib = self.retrieve('finv_agg_lib')
            
        if drlay_lib is None:
            """consider allowing to load from library"""
            drlay_lib = self.retrieve('drlay_lib')
        
        
        #=======================================================================
        # clean proj_lib
        #=======================================================================
        if proj_lib is None:
            proj_lib = copy.deepcopy(self.proj_lib)
        for sa, d in proj_lib.items():
            for k in ['wse_fp_d', 'dem_fp_d', 'finv_fp', 'aoi']:
                if k in d:
                    del d[k]
 
        #=======================================================================
        # build aggregated finvs------
        #=======================================================================
        """these should always be polygons"""
 
        res_d = self.sa_get(meth='get_rsamps_d', write=False, dkey=dkey, 
                            samp_method=samp_method, 
                            drlay_lib=drlay_lib,finv_agg_lib=finv_agg_lib,
                            **kwargs)
 
        #=======================================================================
        # reshape results
        #=======================================================================
        rdx = pd.concat(res_d, names=[self.saCn])
        rdx.columns.name='resolution'
        rserx = rdx.stack().swaplevel().sort_index().rename(dkey)
        
        rserx = rserx.reorder_levels(self.rcol_l+[self.gcn]).sort_index()
 
        
        #join 
        #=======================================================================
        # write
        #=======================================================================
        if write:
            self.ofp_d[dkey] = self.write_pick(rserx,
                                   os.path.join(self.wrk_dir, '%s_%s.pickle' % (dkey, self.longname)),
                                   logger=log)
        return rserx
        

    def build_rsampStats(self, #compute the statistics on raster samples  
                       dkey='rsampStats',
                       npstats = ['mean', 'min', 'max', 'count'],
                       write=None, logger=None,**kwargs):
 
        
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger=self.logger
        log = logger.getChild('build_rsampStats')
        if write is None: write=self.write
        gcn = self.gcn
        agCn=self.agCn
        saCn=self.saCn
        reCn=self.resCn
        #=======================================================================
        # retrieve
        #=======================================================================
        serx = self.retrieve('rsamps')
 
 
        #=======================================================================
        # helpers
        #=======================================================================
        def statFunc(obj, stat_l=npstats, pfx=''):
            return {pfx+stat:getattr(obj, stat)() for stat in stat_l}
                
        #=======================================================================
        # basic stats
        #=======================================================================

        res_d = statFunc(serx.groupby(level=[saCn,agCn, reCn] ))
        
        #=======================================================================
        # wet stats
        #=======================================================================
 
        res_d.update(statFunc(serx[serx>0].groupby(level=[saCn,agCn, reCn] ), pfx='wet_'))
 
        #=======================================================================
        # compile
        #=======================================================================
        rserx = pd.concat(res_d, axis=1, names=['stat']).reorder_levels(self.rcol_l)
        
        
        #=======================================================================
        # meta stats
        #=======================================================================
        rserx['wet_pct'] = rserx['wet_count']/rserx['count']
        
        #=======================================================================
        # write
        #=======================================================================
 
        
        if write:
            self.ofp_d[dkey] = self.write_pick(rserx,
                                   os.path.join(self.wrk_dir, '%s_%s.pickle' % (dkey, self.longname)),
                                   logger=log)
        return rserx
    
    def build_rsampErr(self,#compute the error statistics of raster samples
                       dkey='rsampErr',
                       
                       #paramters
                       confusion=False,
                       
                       #defaults
                       write=None, logger=None,
                       #**kwargs,
                       ):
        
        """
        these are 'dry' errors as we include all zero samples
        """
 
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger=self.logger
        log = logger.getChild('build_rsampErr')
        if write is None: write=self.write
        gcn = self.gcn
        agCn=self.agCn
        saCn=self.saCn
        reCn=self.resCn
        ridn=self.ridn
        rcol_l = self.rcol_l.copy() #results columns for this worker
        #=======================================================================
        # retrieve
        #=======================================================================
        serx_raw = self.retrieve('rsamps').sort_index()
        fmindex1 = self.get_faMindex()
        
            
        #=======================================================================
        # prep
        #=======================================================================
        #expand resolutions onto fa_mindex
        
        fmindex2 = pd.concat({k:fmindex1.to_frame() for k in serx_raw.index.unique(reCn)}, axis=0, names=[reCn]
                             ).index
        
        fmindex2 = fmindex2.reorder_levels(rcol_l.copy() + [gcn, ridn]).sort_values()
 
        #disaggregate
        serx1 = self.get_aggregate(serx_raw, mindex=fmindex2, aggMethod='join', logger=log, agg_name=ridn)
        
        #=======================================================================
        # calc errors-----
        #=======================================================================
        res_d = dict()
        #gcoln_l = [reCn, agCn]
        drop_cols = list(set(serx1.index.names).difference([ridn]))
        base_d = dict()
        for gkeys, gserx in serx1.groupby(level=rcol_l):
            keys_d = dict(zip(rcol_l, gkeys))
            log.info('%s w/ %i'%(keys_d, len(gserx)))
            
            gserx = gserx.sort_index(level=ridn)
            #===================================================================
            # setup base
            #===================================================================
            if keys_d[reCn]==min(serx1.index.unique(reCn)) and keys_d[agCn]==min(serx1.index.unique(agCn)):
                """because we neeed to group by SA which has unique indexers
                alternatively, could just do a lookup"""
                base_d[keys_d[saCn]] = gserx
            
            #retrieve baseline/true
            bserx = base_d[keys_d[saCn]].copy()
                
            #===================================================================
            # compute errorsr
            #===================================================================
            assert np.array_equal(
                np.array(gserx.index.get_level_values(ridn)),
                np.array(bserx.index.get_level_values(ridn)),), keys_d
            
 
            with ErrorCalcs(pred_ser=gserx.droplevel(drop_cols), true_ser=bserx.droplevel(drop_cols), logger=log) as eW:
  
                err_d = eW.get_all(dkeys_l=['bias', 'bias_shift', 'meanError', 'meanErrorAbs', 'RMSE', 'pearson'])
 
                rser = pd.Series(err_d, name=gkeys) #convert remainers to a series
                #===============================================================
                # confusion
                #===============================================================
                if confusion:
     
                    #calc matrix
                    cm_df, cm_dx = eW.get_confusion()
                    rser = rser.append(cm_dx.droplevel(['pred', 'true']).iloc[:, 0])
                    
            res_d[gkeys] = rser
 
        #=======================================================================
        # wrap
        #=======================================================================
        rdx = pd.concat(res_d, axis=1, names=rcol_l).T.sort_index().reorder_levels(rcol_l)
 
        
        #=======================================================================
        # write
        #=======================================================================
 
        
        if write:
            self.ofp_d[dkey] = self.write_pick(rdx,
                                   os.path.join(self.wrk_dir, '%s_%s.pickle' % (dkey, self.longname)),
                                   logger=log)
        return rdx
    
    def get_faMindex(self, #get raw-agg keys mapped as a multiindex
 
                     finv_agg_mindex=None,
                     logger=None):
        
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger=self.logger
        log = logger.getChild('append_rawid')
        
        
        ridn=self.ridn
        gcn = self.gcn
        agCn=self.agCn
 
        reCn=self.resCn
        
        if finv_agg_mindex is None:
            finv_agg_mindex=self.retrieve('faggMap')
            
        #=======================================================================
        # precheck
        #=======================================================================        
        assert finv_agg_mindex.columns.name == agCn
        assert ridn in finv_agg_mindex.index.names        
        #=======================================================================
        # join
        #=======================================================================
        fa_mindex = finv_agg_mindex.stack().rename(gcn).to_frame().set_index(gcn, append=True).sort_index().index
        
        col_l = self.rcol_l.copy() + [gcn, ridn]
        col_l.remove(reCn) 
        
        return fa_mindex.reorder_levels(col_l).sort_values()
    
    def get_indexers_from_layers(self, #retrieve indexers from layers
                     #defaults
                     lay_lib=None,
                     dkey='get_indexers'):
        """only retrives gids (not rawids)"""
        
        #=======================================================================
        # defaults
        #=======================================================================
    
        gcn=self.gcn
        agCn=self.agCn
        #=======================================================================
        # retrival
        #=======================================================================
        if lay_lib is None:
            lay_lib = self.retrieve('finv_agg_lib')
 
        
        
        #retrieve indexers from layer
        def func(vlay, logger=None, meta_d={}):
            df = vlay_get_fdf(vlay)
            assert len(df.columns)==1
            df1 = df.set_index(gcn).sort_index()
            df1.index.name=vlay.name()
            return df1           
        
        
        index_lib = self.calc_on_layers(lay_lib=lay_lib, func=func, subIndexer=agCn, 
                                  format='dict',
                            write=False, dkey=dkey)
        
        d = dict()
        for k,v in index_lib.items():
            d[k] = pd.concat(v, names=[self.saCn, gcn])
            
            
        mindex = pd.concat(d, names=[agCn]).reorder_levels([self.saCn, agCn, gcn]).sort_index().index
        
        return mindex
        
 
        
    
    def build_assetCat(self, #reshape asset data to match raster catalog
                       dkey='assetCat',
                       
                       #data
                       dx=None,finv_agg_mindex=None,
                       
                       #defaults 
                       write=None, logger=None,**kwargs):
        """
        see self.build_rsamps()
        
        I think its easier to store this uncomopressed
            then we dont have to deal with the finv_agg_d anymore
            
        need to mirror the same structure we're using for layers
            1 pickle per raster
 
        """
        
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger=self.logger
        log = logger.getChild('build_assetCat')
        if write is None: write=self.write
        gcn = self.gcn
        agCn=self.agCn
        saCn=self.saCn
        reCn=self.resCn
        ridn = self.ridn
        
        
        #=======================================================================
        # retrieve
        #=======================================================================
        if dx is None:
            dx = self.retrieve('rsamps')
            
        serx = dx.stack().swaplevel().sort_index().rename('rsamps')
        if finv_agg_mindex is None:
            finv_agg_mindex = self.retrieve('faggMap')
            
            
        #=======================================================================
        # write pickles
        #=======================================================================
        return
        out_dir = os.path.join(lib_dir, 'data', *list(id_params.values()))
            
        """
        view(serx)
        """
            
            
            
def aggLevel_remap(l):
    return ['aL%03i'%i for i in l]
    
    
def cat_reindex(serx):
    agCn=ExpoRun.agCn
    dx1 = serx.unstack(level=agCn).swaplevel(axis=1).copy().swaplevel(axis=0).sort_index()
    
    #add prefix to aggLevel names
    idx_raw = dx1.columns.get_level_values(agCn)
    if 'int' in idx_raw.dtype.name:
        dx1.columns.set_levels(level=agCn, levels=aggLevel_remap(idx_raw),
                               verify_integrity=False, inplace=True)
    
    return dx1.sort_index(axis=1)
 
            
        
        
        
        
        
        
        
