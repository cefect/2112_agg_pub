'''
Created on May 15, 2022

@author: cefect
'''


#===============================================================================
# imports-----------
#===============================================================================
import os, datetime, math, pickle, copy, sys, pprint
from pathlib import Path

import qgis.core
from qgis.core import QgsRasterLayer, QgsMapLayerStore
import pandas as pd
import numpy as np
 
#from pandas.testing import assert_index_equal, assert_frame_equal, assert_series_equal

idx = pd.IndexSlice
from hp.exceptions import Error
from hp.pd import get_bx_multiVal, view, assert_index_equal
 
from hp.Q import assert_rlay_equal, QgsMapLayer
from hp.basic import set_info, get_dict_str
from agg.hyd.hscripts import Model

class RRcoms(Model):
    resCn='resolution'
    
    agCn='' #placeholder for build_dataExport
    saCn='studyArea'
    
 
    
    id_params=dict()
    
    dkey_from_cat=list() #built by compileFromCat to inform build_dataExport not to write files to library again
    
    def __init__(self,
                  lib_dir=None,
                  data_retrieve_hndls={},
                  phase_d={},phase_l=None,
                 **kwargs):
        
        data_retrieve_hndls = {**data_retrieve_hndls, **{
                        #combiners
            'res_dx':{
                'compiled':lambda **kwargs:self.load_pick(**kwargs),
                'build':lambda **kwargs:self.build_resdx(**kwargs), #
                },
            
            'dataExport':{
                'compiled':lambda **kwargs:self.load_pick(**kwargs),
                'build':lambda **kwargs:self.build_dataExport(**kwargs), #
                },
            }}
        
            
        
        #=======================================================================
        # handle phase keys
        #=======================================================================
        if phase_l is None:
            phase_l=list(phase_d.keys())
            
        for p in phase_l:
            assert p in phase_d
            
        self.phase_d={k:v for k,v in phase_d.items() if k in phase_l}
        self.phase_l=phase_l
        
        
        super().__init__(data_retrieve_hndls=data_retrieve_hndls, **kwargs)
                
                
        if lib_dir is None:
            lib_dir = os.path.join(self.work_dir, 'lib', self.name)
        #assert os.path.exists(lib_dir), lib_dir
        self.lib_dir=lib_dir
        
        
    #===========================================================================
    # COMPILEERS----
    #===========================================================================
    def compileFromCat(self, #construct pickle from the catalog and add to compiled
                       catalog_fp='',
                       #dkey_l = ['drlay_lib'], #dkeys to laod
                       
                       id_params={}, #index values identifying this run
                       
                       logger=None,
                       pick_index_map=None,
                       studyArea_l=None, #for checks
                       #phase_d=None,
                       index_col=None,
                       dkey_skip_l=[],
                       ):
        """
        because we generally execute a group of parameterizations (id_params) as 1 run (w/ a batch script)
            then compile together in the catalog for analysis
            
        loading straight from the catalog is nice if we want to add one calc to the catalog set
        
        our framework is still constrained to only execute 1 parameterization per call
            add a wrapping for loop to execute on the whole catalog
        """
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger=self.logger
        if pick_index_map is None: pick_index_map=self.pick_index_map
        log=logger.getChild('compileFromCat')
        saCn=self.saCn
        if index_col is None: index_col=self.index_col
        if studyArea_l is None:
            studyArea_l=list(self.proj_lib.keys())
            
        #if phase_d is None: phase_d=self.phase_d
            
        id_params = id_params.copy()
        
        #dkey_l = {e for k,v in phase_d.items() for e in v}
        #=======================================================================
        # for dkey in dkey_l:
        #     assert dkey in pick_index_map
        #     assert not dkey in self.compiled_fp_d, dkey
        #=======================================================================
        #=======================================================================
        # load the catalog
        #=======================================================================
        
        with Catalog(catalog_fp=catalog_fp, logger=log, overwrite=False,
                       index_col=index_col ) as cat:
            
            #===================================================================
            # data prep
            #===================================================================
            dx_raw=cat.get()
            
            for k in id_params.copy().keys():
                if not k in dx_raw.index.names:
                    log.warning('passed indexer \'%s\' not in cattalog... ignoring'%k)
                    del id_params[k]
            
            bx = get_bx_multiVal(dx_raw, id_params, matchOn='index', log=log)
            
            #===================================================================
            # check
            #===================================================================
            assert bx.any(), id_params
            
            miss_l = set(dx_raw.index.unique(saCn)).symmetric_difference(studyArea_l)
            assert len(miss_l)==0, 'got %i studyAreas conflicting between proj_lib and catalog'%len(miss_l)
 
            #===================================================================
            # loop and build
            #===================================================================
            meta_d = dict()
            cnt=0
            for dkey, gdx in dx_raw.loc[bx, :].droplevel(list(id_params.keys())).groupby(level='dkey', axis=1):
                if dkey.startswith('_'): continue #special meta
                
                if dkey in dkey_skip_l: continue #skip those not specified
                
                if dkey in ['finv_agg_lib']:
                    log.warning('%s not impleented... skipping'%dkey)
                    continue
            
 
                log.debug('on %s\n\n'%dkey)
                
                if gdx.isna().all().all():
                    log.warning('no data for %s... skipping'%dkey)
                    continue
                
                assert gdx.notna().all().all()
                #===============================================================
                # filepaths
                #===============================================================
                if 'fp' in gdx.columns.unique(1):
                
                    #pull the filepaths from the catalog
                    assert dkey in pick_index_map, dkey
                    res = cat.get_dkey_fp(dkey=dkey, pick_indexers=pick_index_map[dkey], dx_raw=gdx)
                    
                    #mark this dkey so we dont write to library again
                    self.dkey_from_cat.append(dkey)
                    
                #===============================================================
                # data frames
                #===============================================================
                else:
                    res = gdx.droplevel(0, axis=1).sort_index()
                    
                
                #save as a pickel
                """writing to temp as we never store these"""
                cnt+=1
                meta_d[dkey] = '%s (%i)'%(type(res).__name__, len(res))
                assert not dkey in self.compiled_fp_d
                self.compiled_fp_d[dkey] = self.write_pick(res, 
                                    os.path.join(self.temp_dir, '%s_%s.pickle' % (dkey, self.longname)), 
                                    logger=log.getChild(dkey))
                
            #===================================================================
            # wrap
            #===================================================================
            cat.df=None #avoid writing
                
        log.info('finished on %i:\n    %s'%(len(meta_d), pprint.PrettyPrinter(indent=4).pformat(meta_d)))
        
        return
    
    
    def load_layer_lib(self,  # generic retrival for layer librarires
                  fp=None, dkey=None,
                  **kwargs):
        """not the most memory efficient..."""
        #=======================================================================
        # defaults
        #=======================================================================
        log = self.logger.getChild('load.%s' % dkey)
        assert dkey in ['drlay_lib', 'difrlay_lib', 'finv_agg_lib', 'finv_sg_lib'], dkey
        
        #=======================================================================
        # load the filepaths
        #=======================================================================
        fp_lib = self.load_pick(fp=fp, dkey=dkey)   
        
        #=======================================================================
        # # load layers
        #=======================================================================
        lay_lib = dict()
        cnt = 0
        for k0, d0 in fp_lib.items():
            lay_lib[k0] = dict()
            for k1, fp in d0.items(): #usualy StudyArea
     
                log.info('loading %s.%s from %s' % (k0, k1, fp))
                
                assert isinstance(fp, str), 'got bad type on %s.%s: %s'%(k0, k1, type(fp))
                assert os.path.exists(fp), fp
                ext = os.path.splitext(os.path.basename(fp))[1]
                #===================================================================
                # vectors
                #===================================================================
                if ext in ['.gpkg', '.geojson']:
                
                    lay_lib[k0][k1] = self.vlay_load(fp, logger=log, 
                                                   #set_proj_crs=False, #these usually have different crs's
                                                           **kwargs)
                elif ext in ['.tif']:
                    lay_lib[k0][k1] = self.rlay_load(fp, logger=log, 
                                                   #set_proj_crs=False, #these usually have different crs's
                                                           **kwargs)
                else:
                    raise IOError('unrecognized filetype: %s'%ext)
                cnt+=1
        
        log.info('finished loading %i'%cnt)
        return lay_lib
        
    def store_lay_lib(self,  res_lib,dkey,
                      out_dir=None, 
                      logger=None):
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger=self.logger
        log=logger.getChild('store_lay_lib')
        if out_dir is None:
            out_dir = os.path.join(self.wrk_dir, dkey)
        ofp_lib = dict()
        
        #=======================================================================
        # #write each to file
        #=======================================================================
        for resolution, layer_d in res_lib.items():
            out_dir=os.path.join(out_dir, 'r%i' % resolution)
            ofp_lib[resolution] = self.store_layer_d(layer_d, dkey, logger=log, 
                write_pick=False, #need to write your own
                out_dir=out_dir)
        
        #=======================================================================
        # #write the pick
        #=======================================================================
        self.ofp_d[dkey] = self.write_pick(ofp_lib, 
            os.path.join(self.wrk_dir, '%s_%s.pickle' % (dkey, self.longname)), logger=log)
        
    #============================================================================
    # COMBINERS------------
    #============================================================================
    def build_resdx(self, #just combing all the results
        dkey='res_dx',
        
        phase_l=None,
        phase_d = None,

        logger=None,write=None,
         ):
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger=self.logger
        log=logger.getChild('build_resdx')
        if write is None: write=self.write
        assert dkey=='res_dx'
 
        if phase_l is None: phase_l=self.phase_l
        if phase_d is None: phase_d=self.phase_d.copy()
 
        #clean out
        #phase_d = {k:v for k,v in phase_d.items() if k in phase_l}
        
        phase_d = {p:phase_d[p] for p in phase_l} 
        
        #=======================================================================
        # reindexing functions
        #=======================================================================
        #from agg.hydE.hydE_scripts import cat_reindex as hydE_reindexer
        
        reindex_d = {'expo':lambda x:x.sort_index(sort_remaining=True),
                     'depth':lambda x:x.sort_index(sort_remaining=True),
                     'diff':lambda x:x.sort_index(sort_remaining=True)}
        
        #=======================================================================
        # retrieve and check all
        #=======================================================================
        
        res_d=dict()
        for phase, dki_l in phase_d.items():
            d = dict()
            first = True
            for dki in dki_l:
                raw = self.retrieve(dki)
                
                #use reindexer func for this pahse
                dx = reindex_d[phase](raw) 
                
                #assert np.array_equal(dx.index.names, np.array([resCn, saCn])), dki
                 
                #check consistency within phase
                if first:
                    mindex_last = dx.index
                    first = False
                else:      
                    assert_index_equal(dx.index, mindex_last, msg='%s.%s'%(phase, dki))
                    
    
                d[dki] = dx.sort_index()
            
            res_d[phase] = pd.concat(d, axis=1, names=['dkey', 'stat'])
 
 
        #=======================================================================
        # assemble by type
        #=======================================================================
        first=True
        for phase, dxi in res_d.items():
            #get first
            if first:
                rdx=dxi.copy()
                first=False
                continue
            
            el_d = set_info(dxi.index.names, rdx.index.names)
            #===================================================================
            # simple joins
            #===================================================================
            if len(el_d['symmetric_difference'])==0: 
                dxi = dxi.reorder_levels(rdx.index.names).sort_index()
                assert_index_equal(dxi.index, rdx.index)
 
                rdx = rdx.join(dxi)
                
            #===================================================================
            # expanding join
            #===================================================================
            elif len(el_d['diff_left'])==1:
 
                new_name=list(el_d['diff_left'])[0]
                
                #check the existing indexers match
                skinny_mindex = dxi.index.droplevel(new_name).to_frame().drop_duplicates(
                        ).sort_index().index.reorder_levels(rdx.index.names).sort_values()
                        
                assert_index_equal(skinny_mindex,rdx.index)
                
                #simple join seems to work
                rdx = rdx.join(dxi).sort_index()
                
            #===================================================================
            # contraction join
            #===================================================================
            elif len(el_d['diff_right'])==1:
                """trying to join a smaller index onto the results which have already been expanded"""
                #raise IOError('not implmeented')
                rdx = rdx.join(dxi)
 
            
            else:
                raise IOError(el_d)        
        
 
 
        """
        view(rdx)
        """
        
        rdx = rdx.reorder_levels(self.rcol_l, axis=0).sort_index()
        #=======================================================================
        # wrap
        #=======================================================================
        log.info('finished on %s'%str(rdx.shape))
        if write:
            self.ofp_d[dkey] = self.write_pick(rdx,
                                   os.path.join(self.wrk_dir, '%s_%s.pickle' % (dkey, self.longname)),
                                   logger=log)
            
        return rdx
    
    def build_dataExport(self, #export layers to library
                      dkey='dataExport',
                    lib_dir = None, #library directory
                      overwrite=None,
                      compression='med', #keepin this separate from global compression (which applys to all ops)
                      
                      id_params = {}, #additional parameter values to use as indexers in teh library
                      debug_max_len = None,
                      
                      #defaults
                      phase_l=None,
                      write=None, logger=None):
        """no cleanup here
        setup for one write per parameterization"""
        #=======================================================================
        # defaults
        #=======================================================================
        assert dkey=='dataExport'
        if logger is None: logger=self.logger
        log = logger.getChild(dkey)
        if overwrite is None: overwrite=self.overwrite
        if write is None: write=self.write
        if lib_dir is None:
            lib_dir = self.lib_dir
        
        if phase_l is None: phase_l=self.phase_l
            
        assert os.path.exists(lib_dir), lib_dir
        resCn=self.resCn
        saCn=self.saCn
        agCn=self.agCn
        
 
        #=======================================================================
        # setup filepaths4
        #=======================================================================
        
        rlay_dir = os.path.join(lib_dir, 'data', *list(id_params.values()))
 
 
        #=======================================================================
        # re-write raster layers
        #=======================================================================
        """todo: add filesize"""
        ofp_lib = dict()
        cnt0=0
        for phase, (dki_l, icoln) in  {
            'depth':(['drlay_lib'], resCn),
             'diff':(['difrlay_lib'], resCn),
             'expo':(['finv_agg_lib'],agCn),
            }.items():
            
            #phase selector
            if not phase in phase_l:continue
            
            #===================================================================
            # loop on each key
            #===================================================================
            for dki in dki_l:
                #===================================================================
                # #retrieve
                #===================================================================
                lay_lib = self.retrieve(dki)
                assert_lay_lib(lay_lib, msg=dki)
                
                
                #===============================================================
                # handle catalog loaded layers
                #===============================================================
                if dki in self.dkey_from_cat:
                    log.warning('\'%s\' was loaded from the catalog... skipping'%dki)
                    
                    d = self.load_pick(fp=self.compiled_fp_d[dki], dkey=dkey) 
                    cnt = len(d) 
                    
                else:
                    #===================================================================
                    # #write each layer to file
                    #===================================================================
                    d=dict()
                    cnt=0
                    for indx, layer_d in lay_lib.items():
     
                        d[indx] = self.store_layer_d(layer_d, dki, logger=log,
                                           write_pick=False, #need to write your own
                                           out_dir = os.path.join(rlay_dir,dki, '%s%04i'%(icoln[0], indx)),
                                           compression=compression, add_subfolders=False,overwrite=overwrite,                               
                                           )
                        
                        #debug handler
                        cnt+=1
                        if not debug_max_len is None:
                            if cnt>=debug_max_len:
                                log.warning('cnt>=debug_max_len (%i)... breaking'%debug_max_len)
                                break
                cnt0+=cnt
                #===================================================================
                # compile
                #===================================================================
                #dk_clean = dki.replace('_lib','')
                fp_serx = pd.DataFrame.from_dict(d).stack().swaplevel().rename('fp')
                fp_serx.index.set_names([icoln, saCn], inplace=True)
                
                #===================================================================
                # filesizes
                #===================================================================
                dx = fp_serx.to_frame()
                dx['fp_sizeMB'] = np.nan
                for gkeys, fp in fp_serx.items():
                    dx.loc[gkeys, 'fp_sizeMB'] = Path(fp).stat().st_size*1e-6
                
                dx.columns.name='stat'
                
                #===============================================================
                # wrap
                #===============================================================
                assert len(dx)>0
                
                assert not dx.index.to_frame().isna().any().any(), dki
                
                ofp_lib[dki] = pd.concat({dki:dx}, axis=1, names=['dkey'])
            
        #=======================================================================
        # #concat by indexer
        #=======================================================================
        """because phases have separate indexers but both use this function:
            depths + diffs: indexed by resolution
            expo: indexed by aggLevel
            
        """
        rdx =None
        for dki, dxi in ofp_lib.items():
            if rdx is None: 
                rdx = dxi.copy()

            else:
                rdx = rdx.merge(dxi, how='outer', left_index=True, right_index=True, sort=True)
                
            assert rdx.notna().all().all(), dki
            assert rdx.index.to_frame().notna().all().all(), dki
                
            """
            view(rdx.merge(dxi, how='outer', left_index=True, right_index=True))
            """
 
        rdx = rdx.reorder_levels(self.rcol_l, axis=0).sort_index()
        #=======================================================================
        # write
        #=======================================================================
        log.info('finished writing %i'%cnt0)
        if write:
            self.ofp_d[dkey] = self.write_pick(rdx,
                                   os.path.join(self.wrk_dir, '%s_%s.pickle' % (dkey, self.longname)),
                                   logger=log)
            
        return rdx
        
   

    def write_lib(self,
                  catalog_fp=None,
                  lib_dir = None, #library directory
                  res_dx = None, ldx=None,
                  overwrite=None, logger=None,
                  id_params={},
                  ):
        
        #=======================================================================
        # defautls
        #=======================================================================
        if overwrite is None: overwrite=self.overwrite
        if logger is None: logger=self.logger
        log=logger.getChild('write_lib')
        
 
        
        
        if lib_dir is None:
            lib_dir = self.lib_dir
            
        if catalog_fp is None: 
            catalog_fp = os.path.join(lib_dir, '%s_run_index.csv'%self.name)
            
        #=======================================================================
        # retrieve
        #=======================================================================
        if res_dx is None:
            res_dx=self.retrieve('res_dx')
           
        if ldx is None: 
            ldx = self.retrieve('dataExport', id_params=id_params, lib_dir=lib_dir)
        
        assert_index_equal(res_dx.index, ldx.index)    
        #=======================================================================
        # build catalog--------
        #=======================================================================
        cdx = res_dx.join(ldx)
 
        
        
        #=======================================================================
        # #add additional indexers
        #=======================================================================
        #add singgle value levels from a dictionary
        mdex_df = cdx.index.to_frame().reset_index(drop=True)
        for k,v in id_params.items():
            mdex_df[k] = v
            
            #remove from cols
            if k in cdx.columns.get_level_values(1):
                """TODO: check the values are the same"""
                cdx = cdx.drop(k, level=1, axis=1)
            
        cdx.index = pd.MultiIndex.from_frame(mdex_df)
        
        #=======================================================================
        # add metadata
        #=======================================================================
        meta_d = {'tag':self.tag, 'date':datetime.datetime.now().strftime('%y-%m-%d.%H%M'),
                  'runtime_mins':(datetime.datetime.now() - self.start).total_seconds()/60.0
                  }
        cdx = cdx.join(pd.concat({'_meta':pd.DataFrame(meta_d, index=cdx.index)}, axis=1))
 
 
        
        #=======================================================================
        # write catalog-----
        #=======================================================================
        miss_l = set(cdx.index.names).difference(Catalog.keys)
        assert len(miss_l)==0, 'key mistmatch with catalog worker: %s'%miss_l
        
        with Catalog(catalog_fp=catalog_fp, overwrite=overwrite, logger=log, 
                     index_col=list(range(len(cdx.index.names)))
                                    ) as cat:
            for rkeys, row in cdx.iterrows():
                keys_d = dict(zip(cdx.index.names, rkeys))
                cat.add_entry(row, keys_d, logger=log.getChild(str(rkeys)))
        
        
        """
        rdx.index.names.to_list()
        view(cdx)
        """
        log.info('finished')
        return catalog_fp

class Catalog(object): #handling the simulation index and library
    df=None
    keys = ['resolution', 'studyArea', 'downSampling', 'dsampStage', 'severity','sequenceType',
            'aggType', 'samp_method', 'aggLevel']
    cols = ['dkey', 'stat']
 

    
    def __init__(self, 
                 catalog_fp='fp', 
                 logger=None,
                 overwrite=True,
                 index_col=list(range(5)),
                 ):
        
        if logger is None:
            import logging
            logger = logging.getLogger()
 
        #=======================================================================
        # attachments
        #=======================================================================
        self.logger = logger.getChild('cat')
        self.overwrite=overwrite
        self.catalog_fp = catalog_fp
        
        
        #mandatory keys
        """not using this anymore"""
        self.cat_colns = []
        
        """
        self.logger.info('test')
        """
        #=======================================================================
        # load existing
        #=======================================================================
        if os.path.exists(catalog_fp):
            self.df = pd.read_csv(catalog_fp, 
                                  index_col=index_col,
                                  header = [0,1],
                                  )
            self.check(df=self.df.copy())
            self.df_raw = self.df.copy() #for checking
            
        else:
            self.df_raw=pd.DataFrame()
        
    def clean(self):
        raise Error('check consitency between index and library contents')
    
    def check(self,
              df=None,
              ):
        #=======================================================================
        # defai;lts
        #=======================================================================
        log = self.logger.getChild('check')
        if df is None: df = self.df.copy()
        log.debug('on %s'%str(df.shape))
        
        
        
        #=======================================================================
        # #check columns
        #=======================================================================
        assert not None in df.columns.names
        miss_l = set(self.cols).symmetric_difference(df.columns.names)
        assert len(miss_l)==0, miss_l
        
        assert isinstance(df.columns, pd.MultiIndex)
        
 
        #=======================================================================
        # #check index
        #=======================================================================
        assert not None in df.index.names
        #=======================================================================
        # assert df[self.idn].is_unique
        # assert 'int' in df[self.idn].dtype.name
        #=======================================================================
        assert isinstance(df.index, pd.MultiIndex)
        
        miss_l = set(df.index.names).difference(self.keys)
        assert len(miss_l)==0, miss_l
 
        
        #check filepaths
        errs_d = dict()
        
        bx_col = df.columns.get_level_values(1).str.endswith('fp')
        assert bx_col.any()
        
        for coln, cserx in df.loc[:, bx_col].items():
 
            
            for id, path in cserx.items():
                if pd.isnull(path):
                    log.warning('got null filepath on %s'%str(id))
                    continue
                if not os.path.exists(path):
                    errs_d['%s_%s'%(coln, id)] = path
                    
        if len(errs_d)>0:
            log.error(get_dict_str(errs_d))
            raise Error('got %i/%i bad filepaths'%(len(errs_d), len(df)))
 
 
        
    
    def get(self):
        assert os.path.exists(self.catalog_fp), self.catalog_fp
        self.check()
        return self.df.copy().sort_index()
    
    
    def remove(self, keys_d,
               logger=None): #remove an entry
        #=======================================================================
        # defaults 
        #=======================================================================
        if logger is None: logger=self.logger
        log=logger.getChild('remove')
        df_raw = self.df.copy()
        
        #identify this row
        bx = pd.Series(True, index=df_raw.index)
        for k,v in keys_d.items():
            bx = np.logical_and(bx,
                                df_raw.index.get_level_values(k)==v,
                                )
            
        assert bx.sum()==1
        
        #remove the raster
        """no... this should be handled by the library writer
        rlay_fp = df_raw.loc[bx, 'rlay_fp'].values[0]
        
        os.remove(rlay_fp)"""
            
        
        #remove the row
        self.df = df_raw.loc[~bx, :]
        
        log.info('removed %s'%(keys_d))
        
 

    def add_entry(self,
                  serx,
                  keys_d={},
                  logger=None,
                  ):
        """
        cat_d.keys()
        """
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger=self.logger
        log=logger.getChild('add_entry')
        cat_df = self.df
        
        """making this more flexible
        keys = self.keys.copy()"""
        keys = list(keys_d.keys())

        log.debug('w/ %i'%len(serx))
        #check mandatory columns are there
        miss_l = set(self.cat_colns).difference(serx.index.get_level_values(1))
        assert len(miss_l)==0, 'got %i missing cols: %s'%(len(miss_l), miss_l)
        
        for k in keys: 
            assert k in keys_d
        
        #=======================================================================
        # prepare new 
        #=======================================================================
        
        #new_df = pd.Series(cat_d).rename(keys_d.values()).to_frame().T
 
        new_df = serx.to_frame().T
        new_df.index = pd.MultiIndex.from_tuples([keys_d.values()], names=keys_d.keys())
        """
        view(new_df)
        """
        #=======================================================================
        # append
        #=======================================================================
        if cat_df is None:
            #convet the series (with tuple name) to a row of a multindex df
            cat_df=new_df
        else:
            #check if present
            
            cdf = cat_df.index.to_frame().reset_index(drop=True)
            ndf = new_df.index.to_frame().reset_index(drop=True)
 
            bxdf = cdf.apply(lambda s:ndf.eq(s).iloc[0,:], axis=1)
            bx = bxdf.all(axis=1)
 
            
            
            if bx.any():
                
                #===============================================================
                # remomve an existing entry
                #===============================================================
                assert bx.sum()==1
                assert self.overwrite
                bx.index = cat_df.index
                self.remove(dict(zip(keys, cat_df.loc[bx, :].iloc[0,:].name)))
                
                cat_df = self.df.copy()
                
            
            #===================================================================
            # append
            #===================================================================
            cat_df = cat_df.append(new_df,  verify_integrity=True)
            
        #=======================================================================
        # wrap
        #=======================================================================
        self.df = cat_df
 
        
        log.info('added %s'%len(keys_d))
        
    def get_dkey_fp(self, #build a pickle by dkey
                   dkey='', 
                   dx_raw=None, #catalog frame
                   
                   #parmaeters
                   pick_indexers=tuple(), #map of how the pickels are indexed
                   #id_params={}, #additional indexers identfying this run
                   #defaults
                   logger=None,
                   **kwargs):
        
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger=self.logger
        log=logger.getChild('build_pick_%s'%dkey)
        if dx_raw is None: dx_raw=self.get()
        ln1 = pick_indexers[0]
        #=======================================================================
        # precheck
        #=======================================================================
        assert dkey in dx_raw.columns.get_level_values(0), dkey
        
        #=======================================================================
        # #indexers implied by this selection
        # keys_s = set(pick_indexers).union(id_params.keys())
        # 
        # #compoared to indexers found on the catalog
        # miss_l = set(keys_s).difference(dx_raw.index.names)
        # assert len(miss_l)==0
        #=======================================================================
        
        #=======================================================================
        # prep columns
        #=======================================================================
        
        #slice to just this data
        serx1 = dx_raw.loc[:,idx[dkey, 'fp']]
        """
        view(serx1)
        """
        
 
        #=======================================================================
        # prep index
        #=======================================================================
        #remove any unecessary indexers
        drop_l = list(set(serx1.index.names).difference(pick_indexers))
        if len(drop_l)>0:
            serx1 = serx1.droplevel(drop_l).drop_duplicates()
        
        
        #=======================================================================
        # bx = get_bx_multiVal(serx0, id_params, matchOn='index', log=log)
        # 
        # serx1 = serx0[bx].droplevel(list(id_params.keys()))
        #=======================================================================
        
        if not serx1.is_unique:
            """this happens during later analysis where we duplicate filepaths"""
            raise IOError('got duplicate values on %s'%dkey)
        """
        view(serx1)
        """
        
        assert len(serx1)>0
        
        #=======================================================================
        # collapse to dict
        #=======================================================================
        res_d = dict()
        for studyArea, gserx in serx1.groupby(level=ln1):
            d =  gserx.droplevel(level=ln1).to_dict()
            
            #check these
            for k,fp in d.items():
                assert os.path.exists(fp), 'bad fp on %s.%s: \n    %s'%(studyArea, k, fp)
                
            res_d[studyArea] = d
            
            
        log.info('got %i'%len(serx1))
        
        return res_d
        
 
 
        
    def __enter__(self):
        return self
    def __exit__(self, *args, **kwargs):
        if self.df is None: 
            return
        #=======================================================================
        # write if there was a change
        #=======================================================================
        if not np.array_equal(self.df, self.df_raw):
            log = self.logger.getChild('__exit__')
            df = self.df
            
            #===================================================================
            # delete the old for empties
            #===================================================================
 
            if len(df)==0:
                try:
                    os.remove(self.catalog_fp)
                    log.warning('got empty catalog... deleteing file')
                except Exception as e:
                    raise Error(e)
            else:
                #===============================================================
                # write the new
                #===============================================================
                """should already be consolidated... just overwrite"""
 
                df.to_csv(self.catalog_fp, mode='w', header = True, index=True)
                self.logger.info('wrote %s to %s'%(str(df.shape), self.catalog_fp))
        
#===============================================================================
# funcs
#===============================================================================
def assert_lay_lib(lib_d, msg=''):
    if __debug__:
        assert isinstance(lib_d, dict)
        for k0,d0 in lib_d.items():
            assert_lay_d(d0, msg=msg+' '+str(k0))

def assert_lay_d(d0, msg=''):
    if __debug__:
        if not isinstance(d0, dict):
            raise AssertionError('bad type on  : %s\n'%(
                type(d0))+msg)
        
        for k1, lay in d0.items():
            if k1 is None:
                raise AssertionError('bad key on  %s: \n'%(
                     k1 )+msg)
            if not isinstance(lay, QgsMapLayer):
                raise AssertionError('bad type on  %s: %s\n'%(
                     k1, type(lay))+msg)
                    
                    
                    
                    
                    
                    