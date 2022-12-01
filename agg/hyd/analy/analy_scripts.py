'''
Created on Feb. 21, 2022

@author: cefect
'''
#===============================================================================
# imports------
#===============================================================================
import os, datetime, math, pickle, copy, random, pprint, gc, math
import matplotlib
from matplotlib.colors import Normalize 
import scipy.stats
from scipy.interpolate import interpn

import pandas as pd
import numpy as np
from pandas.testing import assert_index_equal, assert_frame_equal, assert_series_equal

idx = pd.IndexSlice




import matplotlib.pyplot as plt


#===============================================================================
# custom imports
#===============================================================================
from hp.basic import set_info, get_dict_str
from hp.exceptions import Error, assert_func

from hp.plot import Plotr

from hp.Q import Qproj, QgsCoordinateReferenceSystem, QgsMapLayerStore, view, \
    vlay_get_fdata, vlay_get_fdf, Error, vlay_dtypes, QgsFeatureRequest, vlay_get_geo, \
    QgsWkbTypes

from hp.err_calc import ErrorCalcs
from hp.gdal import rlay_to_array, getRasterMetadata
    
from agg.coms.scripts import Catalog
from agg.hyd.hscripts import HydSession



def get_ax(
        figNumber=0,
        figsize=(4,4),
        tight_layout=False,
        constrained_layout=True,
        ):
    
    if figNumber in plt.get_fignums():
        plt.close()
    
    fig = plt.figure(figNumber,figsize=figsize,
                tight_layout=tight_layout,constrained_layout=constrained_layout,
                )
            
    return fig.add_subplot(111)
 
 

 
class ModelAnalysis(HydSession, Qproj, Plotr): #analysis of model results
    
    
    
    
    #colormap per data type
    colorMap_d = {
        'aggLevel':'cool',
        'dkey_range':'winter',
        'studyArea':'Dark2',
        'modelID':'Pastel1',
        'dsampStage':'Set1',
        'downSampling':'Set2',
        'aggType':'Pastel2',
        'tval_type':'Set1',
        'resolution':'copper',
        'vid':'Set1',
        'dkey':'tab20',
        'density':'viridis',
        'confusion':'RdYlGn',
        }
    
    def __init__(self,
                 catalog_fp=r'C:\LS\10_OUT\2112_Agg\lib\hyd2\model_run_index.csv',
                 plt=None,
                 name='analy',
                 modelID_l=None, #optional list for specifying a subset of the model runs
                 #baseID=0, #mase model ID
                 exit_summary=False,
                 **kwargs):
        
        data_retrieve_hndls = {
            'catalog':{
                #probably best not to have a compliled version of this
                'build':lambda **kwargs:self.build_catalog(**kwargs), #
                },
            'outs':{
                'compiled':lambda **kwargs:self.load_pick(**kwargs),
                'build':lambda **kwargs: self.build_outs(**kwargs),
                },
            'agg_mindex':{
                'compiled':lambda **kwargs:self.load_pick(**kwargs),
                'build':lambda **kwargs: self.build_agg_mindex(**kwargs),
                },
            'trues':{
                'compiled':lambda **kwargs:self.load_pick(**kwargs), #consider checking the baseID
                'build':lambda **kwargs: self.build_trues(**kwargs),
                },
            'deltas':{
                'compiled':lambda **kwargs:self.load_pick(**kwargs), #consider checking the baseID
                'build':lambda **kwargs: self.build_deltas(**kwargs),
                },
            'finv_agg_fps':{
                'compiled':lambda **kwargs:self.load_pick(**kwargs), #only filepaths?
                'build':lambda **kwargs: self.build_finv_agg(**kwargs),
                },
            'drlay_fps':{
                'compiled':lambda **kwargs:self.load_pick(**kwargs), 
                'build':lambda **kwargs: self.build_drlay_fps(**kwargs),
                }
            
            
            }
        
        super().__init__(data_retrieve_hndls=data_retrieve_hndls,name=name,init_plt_d=None,
                         work_dir = r'C:\LS\10_OUT\2112_Agg', exit_summary=exit_summary,
                         **kwargs)
        self.plt=plt
        self.catalog_fp=catalog_fp
        #self.baseID=baseID
        self.modelID_l=modelID_l
    
    #===========================================================================
    # RESULTS ASSEMBLY---------
    #===========================================================================
    def runCompileSuite(self, #conveneince for compilling all the results in order
                        ):
 
        
        cat_df = self.retrieve('catalog')
               
        dx_raw = self.retrieve('outs')
        
        agg_mindex = self.retrieve('agg_mindex')        
 
        true_d = self.retrieve('trues')
        
        self.retrieve('drlay_fps')
        
        
        
    
    def build_catalog(self,
                      dkey='catalog',
                      catalog_fp=None,
                      logger=None,
                      **kwargs):
        if logger is None: logger=self.logger
        assert dkey=='catalog'
        if catalog_fp is None: catalog_fp=self.catalog_fp
        
        return Catalog(catalog_fp=catalog_fp, logger=logger, overwrite=False, **kwargs).get()
    
    def build_agg_mindex(self,
                         dkey='agg_mindex',
    
                        write=None,
                        idn=None, logger=None,
                     **kwargs):
        """
        todo: check against loaded outs?
        """
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger=self.logger
        log=logger.getChild('build_agg_mindex')
        assert dkey=='agg_mindex'
        if write is None: write=self.write
        if idn is None: idn=self.idn
        #=======================================================================
        # pull from piciles
        #=======================================================================
        #get each finv_agg_mindex from the run pickels
        data_d= self.assemble_model_data(dkey='finv_agg_mindex', 
                                         logger=log, write=write, idn=idn,
                                         **kwargs)
        
        #=======================================================================
        # combine
        #=======================================================================
        log.debug('on %s'%data_d.keys())
        
        d = {k: mdex.to_frame().reset_index(drop=True) for k, mdex in data_d.items()}
        
        dx1 = pd.concat(d, names=[idn, 'index'])
        
        mdex = dx1.set_index(['studyArea', 'gid', 'id'], append=True).droplevel('index').index
        
        #=======================================================================
        # wrap
        #=======================================================================
        log.info('finished on %s'%str(mdex.to_frame().shape))
        if write:
            self.ofp_d[dkey] = self.write_pick(mdex,
                                   os.path.join(self.wrk_dir, '%s_%s.pickle' % (dkey, self.longname)),
                                   logger=log)
 
        return mdex
    
    def build_outs(self, #collecting outputs from multiple model runs
                    dkey='outs',
                         write=None,
                         idn=None,
                         cat_df=None,
                         logger=None,
                         **kwargs):
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger=self.logger
        log=logger.getChild('build_outs')
        assert dkey=='outs'
        if cat_df is None: 
            cat_df = self.retrieve('catalog')
        if write is None: write=self.write
        if idn is None: idn=self.idn
        #=======================================================================
        # pull from piciles
        #=======================================================================
        
        data_d= self.assemble_model_data(dkey='tloss', 
                                         logger=log, write=write, cat_df=cat_df, idn=idn,
                                         **kwargs)
    
    
        #=======================================================================
        # combine
        #=======================================================================
        dx = pd.concat(data_d).sort_index(level=0)
 
            
        dx.index.set_names(idn, level=0, inplace=True)
        
        #join tags
        dx.index = dx.index.to_frame().join(cat_df['tag']).set_index('tag', append=True
                          ).reorder_levels(['modelID', 'tag', 'studyArea', 'event', 'gid'], axis=0).index
        
        """
        view(dx)
        """
        
        #=======================================================================
        # wrap
        #=======================================================================
        log.info('finished on %s'%str(dx.shape))
        if write:
            self.ofp_d[dkey] = self.write_pick(dx,
                                   os.path.join(self.wrk_dir, '%s_%s.pickle' % (dkey, self.longname)),
                                   logger=log)
 
        return dx
    

    def build_finv_agg(self,
                         dkey='finv_agg_fps',
    
                        write=None,
                        idn=None,
                        logger=None,
                     **kwargs):
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger=self.logger
        log=logger.getChild('build_finv_agg')
        assert dkey=='finv_agg_fps'
        if write is None: write=self.write
        if idn is None: idn=self.idn
        #=======================================================================
        # pull from piciles
        #=======================================================================
        
        fp_lib= self.assemble_model_data(dkey='finv_agg_d', 
                                         logger=log, write=write, idn=idn,
                                         **kwargs)
        
        #=======================================================================
        # check
        #=======================================================================
        cnt = 0
        for mid, d in fp_lib.items():
            for studyArea, fp in d.items():
                assert os.path.exists(fp)
                cnt+=1
                
        
        
        
        #=======================================================================
        # wrap
        #=======================================================================
        log.info('retrieved %i finv_agg fps'%cnt)
        if write:
            self.ofp_d[dkey] = self.write_pick(fp_lib,
                                   os.path.join(self.wrk_dir, '%s_%s.pickle' % (dkey, self.longname)),
                                   logger=log)
 
        return fp_lib
        
    def build_drlay_fps(self, #collect depth rlay filepaths from the results lib
                         dkey='drlay_fps',
    
                        write=None,
                        idn=None,
                        logger=None,
                     **kwargs):
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger=self.logger
        log=logger.getChild('drlay_fps')
        assert dkey=='drlay_fps'
        if write is None: write=self.write
        if idn is None: idn=self.idn
        #=======================================================================
        # pull directories from piciles
        #=======================================================================
        
        dir_lib= self.assemble_model_data(dkey='rlay_dir', 
                                         logger=log, write=write, idn=idn,
                                         **kwargs)
        
        #=======================================================================
        # collect filepaths
        #=======================================================================
        cnt = 0
        fp_lib = {e:dict() for e in dir_lib.keys()}
        for mid, rlay_dir in dir_lib.items():
            assert os.path.exists(rlay_dir)
            for sub_fn in next(os.walk(rlay_dir))[1]: #should be study areas
                sub_dir = os.path.join(rlay_dir, sub_fn)
                #all the files here
                fns = [e for e in os.listdir(sub_dir) if e.endswith('.tif')]
                
                """each should only have 1 depth rlay... not capturing other rlays"""
                assert len(fns)==1, 'unexpected match in %s\n    %s'%(sub_dir, fns)
 
                #add the result
                fp_lib[mid][sub_fn] = os.path.join(sub_dir, fns[0])
                cnt +=1
                
            
        assert cnt/len(fp_lib) == math.floor(cnt/len(fp_lib)), 'loaded uneven count'
        #=======================================================================
        # wrap
        #=======================================================================
        log.info('from %i modelIDs got %i rlays'%(len(fp_lib), cnt))
        if write:
            self.ofp_d[dkey] = self.write_pick(fp_lib,
                                   os.path.join(self.wrk_dir, '%s_%s.pickle' % (dkey, self.longname)),
                                   logger=log)
 
        return fp_lib #{modelID:{studyArea:depth_rlay_fp}}
    
    def assemble_model_data(self, #collecting outputs from multiple model runs
                   modelID_l=None, #set of modelID's to include (None=load all in the catalog)
                   dkey='outs',
                    
                    cat_df=None,
                     idn=None,
                     write=None,
                     logger=None,
                     ):
        """
        loop t hrough each pickle, open, then retrive the requested dkey
        just collecting into a dx for now... now meta
        """
        #=======================================================================
        # defaults
        #=======================================================================
        log = logger.getChild('build_%s'%dkey)
        
        if write is None: write=self.write
        if idn is None: idn=self.idn
        if cat_df is None: cat_df = self.retrieve('catalog')
        if modelID_l is None: modelID_l=self.modelID_l
        
        assert idn==Catalog.idn
        #=======================================================================
        # retrieve catalog
        #=======================================================================
        
        
        if modelID_l is None:
            modelID_l= cat_df.index.tolist()
        
        assert len(modelID_l)>0
        #check
        miss_l = set(modelID_l).difference(cat_df.index)
        assert len(miss_l)==0, '%i/%i requested %s not found in catalog:\n    %s'%(
            len(miss_l), len(modelID_l), idn, miss_l)
        
        log.info('on %i'%len(modelID_l))
        #=======================================================================
        # load data from modle results
        #=======================================================================
        data_d = dict()
        for modelID, row in cat_df.loc[cat_df.index.isin(modelID_l),:].iterrows():
            log.info('    on %s.%s'%(modelID, row['tag']))
            
            #check
            pick_keys = eval(row['pick_keys'])
            assert dkey in pick_keys, 'requested dkey not stored in the pickle: \'%s\''%dkey
            
            #load pickel            
            with open(row['pick_fp'], 'rb') as f:
                data = pickle.load(f) 
                assert dkey in data, 'modelID=%i has no \'%s\' in pickle \n    %s'%(modelID, dkey, row['pick_fp'])
                
            
                #handle filepaths
                if isinstance(data[dkey], str):
                    data_d[modelID] = data[dkey]
                else:
 
                    data_d[modelID] = data[dkey].copy()
                #===============================================================
                # except Exception as e:
                #     log.warning('failed to copy %i.%s (taking raw) w/ \n    %s'%(modelID, dkey, e))
                #===============================================================
                    
                
                del data
                
        #=======================================================================
        # wrap
        #=======================================================================
        gc.collect()
        return data_d
                

    
    def build_trues(self, #map 'base' model data onto the index of all the  models 
                         
                     baseID_l=[0], #modelID to consider 'true'
                     #modelID_l=None, #optional slicing to specific models
                     dkey='trues',
                     dx_raw=None, agg_mindex=None,
                     
                     idn=None, write=None, logger=None,
                     ):
        """
        
        for direct comparison w/ aggregated model results,
            NOTE the index is expanded to preserve the trues
            these trues will need to be aggregated (groupby) again (method depends on what youre doing)
            
        TODO: check the specified baseID run had an aggLevel=0
        """
        
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger=self.logger
        log = logger.getChild('trues')
        if idn is None: idn=self.idn
        if write is None: write=self.write
        assert dkey == 'trues'
        
        #combined model outputs (on aggregated indexes): {modelID, tag, studyArea, event, gid}
        if dx_raw is None: 
            dx_raw = self.retrieve('outs')
            
        #index relating aggregated ids to raw ids {modelID, studyArea, gid, id}
        if agg_mindex is None:
            agg_mindex = self.retrieve('agg_mindex')
        
        
        
        log.info('on %s'%str(dx_raw.shape))
        
        #=======================================================================
        # check
        #=======================================================================
        #clean out extra index levels for checking
        l = set(dx_raw.index.names).difference(agg_mindex.names)
        chk_mindex = dx_raw.index.droplevel(list(l))
        
        
        assert_func(lambda: self.check_mindex_match_cats(agg_mindex,chk_mindex, glvls = [self.idn, 'studyArea']), 'agg_mindex')
        
        #check the base indicies are there
        miss_l = set(baseID_l).difference(dx_raw.index.unique(idn))
        assert len(miss_l)==0, 'requested baseIDs not loaded: %s'%miss_l
 
        
        miss_l = set(dx_raw.index.unique(idn)).symmetric_difference(agg_mindex.unique(idn))
        assert len(miss_l)==0, '%s mismatch between outs and agg_mindex... recompile?'%idn
        
        #=======================================================================
        # build for each base
        #=======================================================================
        log.info('building %i true sets'%len(baseID_l))
        res_d = dict()
        for i, baseID in enumerate(baseID_l):
            log.info('%i/%i on baseID=%i'%(i+1, len(baseID_l), baseID))
            #=======================================================================
            # get base
            #=======================================================================
            #just the results for the base model
            base_dx = dx_raw.loc[idx[baseID, :, :, :], :].droplevel([idn, 'tag', 'event']).dropna(how='all', axis=1)
            base_dx.index = base_dx.index.remove_unused_levels().set_names('id', level=1)
     
            #=======================================================================
            # expand to all results
            #=======================================================================
            #add the events and tags
            amindex1 = agg_mindex.join(dx_raw.index).reorder_levels(dx_raw.index.names + ['id'])
            
     
            #create a dummy joiner frame
            """need a 2d column index for the column dimensions to be preserved during the join"""
            jdx = pd.DataFrame(index=amindex1, columns=pd.MultiIndex.from_tuples([('a',1)], names=base_dx.columns.names))
            
            #=======================================================================
            # #loop and add base values for each Model
            #=======================================================================
            d = dict()
            err_d = dict()
            for modelID, gdx0 in jdx.groupby(level=idn):
                """
                view(gdx0.sort_index(level=['id']))
                """
                log.debug(modelID)
                #check indicides
                try:
                    assert_index_equal(
                        base_dx.index,
                        gdx0.index.droplevel(['modelID', 'tag', 'gid', 'event']).sortlevel()[0],
                        #check_order=False, #not in 1.1.3 yet 
                        #obj=modelID #not doing anything
                        )
                    
                    #join base data onto this models indicides
                    gdx1 =  gdx0.join(base_dx, on=base_dx.index.names).drop('a', axis=1, level=0)
                    
                    #check
                    assert gdx1.notna().all().all(), 'got %i/%i nulls'%(gdx1.isna().sum().sum(), gdx1.size)
                    d[modelID] = gdx1.copy()
                except Exception as e:
                    err_d[modelID] = e
            
            #report errors
            if len(err_d)>0:
                for mid, msg in err_d.items():
                    log.error('%i: %s'%(mid, msg))
                raise Error('failed join on %i/%i \n    %s'%(
                    len(err_d), len(jdx.index.unique(idn)), list(err_d.keys())))
                    
     
                
            #combine
            dx1 = pd.concat(d.values())
                
     
            #=======================================================================
            # check
            #=======================================================================
            assert dx1.notna().all().all()
            #check columns match
            assert np.array_equal(dx1.columns, base_dx.columns)
            
            #check we still match the aggregated index mapper
            assert np.array_equal(
                dx1.index.sortlevel()[0].to_frame().reset_index(drop=True).drop(['tag','event'], axis=1).drop_duplicates(),
                agg_mindex.sortlevel()[0].to_frame().reset_index(drop=True)
                ), 'result failed to match original mapper'
            
            assert_series_equal(dx1.max(axis=0), base_dx.max(axis=0))
     
            assert_func(lambda: self.check_mindex_match(dx1.index, dx_raw.index), msg='raw vs trues')
            
            res_d[baseID] = dx1.copy()
        #=======================================================================
        # wrap
        #=======================================================================
        log.info('finished on  %i'%len(res_d))
 
 
        if write:
            self.ofp_d[dkey] = self.write_pick(res_d,
                                   os.path.join(self.wrk_dir, '%s_%s.pickle' % (dkey, self.longname)),
                                   logger=log)
 
        return res_d
    
 
        
 
    def xxxwrite_loss_smry(self,  # write statistcs on total loss grouped by grid_size, studyArea, and event
                    
                   # data control   
                    dkey='tloss',
                    # lossType='tl', #loss types to generate layers for
                    gkeys=[ 'studyArea', 'event', 'grid_size'],
                    
                    # output config
                    write=True,
                    out_dir=None,
                    ):
 
        """not an intermediate result.. jsut some summary stats
        any additional analysis should be done on the raw data
        """
        #=======================================================================
        # defaults
        #=======================================================================
        scale_cn = self.scale_cn
        log = self.logger.getChild('write_loss_smry')
        assert dkey == 'tloss'
        if out_dir is None:
            out_dir = os.path.join(self.out_dir, 'errs')
 
        #=======================================================================
        # #retrieve child data
        #=======================================================================
        # errors
        dx_raw = self.retrieve(dkey)
        
        """
        view(self.retrieve('errs'))
        view(dx_raw)
        """
 
        log.info('on %i for \'%s\'' % (len(dx_raw), gkeys))
        #=======================================================================
        # calc group stats-----
        #=======================================================================
        rlib = dict()
        #=======================================================================
        # loss values
        #=======================================================================
        
        for lossType in dx_raw.columns.unique('lossType'):
            if lossType == 'expo':continue
            dxind1 = dx_raw.loc[:, idx[lossType,:]].droplevel(0, axis=1)
            # mdex = dxind1.index
            
            gbo = dxind1.groupby(level=gkeys)
            
            # loop and get each from the grouper
            d = dict()
            for statName in ['sum', 'mean', 'min', 'max']:
                d[statName] = getattr(gbo, statName)()
                
            # collect
            
            #=======================================================================
            # errors
            #=======================================================================
            """could also do this with the 'errs' data set... but simpler to just re-calc the totals here"""
            err_df = None
            for keys, gdf in gbo:
                keys_d = dict(zip(gkeys, keys))
                
                if keys_d['grid_size'] == 0: continue
                
                # get trues
                """a bit awkward as our key order has changed"""
                true_gdf = dxind1.loc[idx[0, keys_d['studyArea'], keys_d['event']],:]
     
                # calc delta (gridded - true)
                eser1 = gdf.sum() - true_gdf.sum()
     
                # handle results
                """couldnt figure out a nice way to handle this... just collecting in frame"""
                ival_ser = gdf.index.droplevel('gid').to_frame().reset_index(drop=True).iloc[0,:]
                
                eser2 = pd.concat([eser1, ival_ser])
                
                if err_df is None:
                    err_df = eser2.to_frame().T
                    
                else:
                    err_df = err_df.append(eser2, ignore_index=True)
            
            # collect
            d['delta'] = pd.DataFrame(err_df.loc[:, gdf.columns].values,
                index=pd.MultiIndex.from_frame(err_df.loc[:, gkeys]),
                columns=gdf.columns)
            
            rlib[lossType] = pd.concat(d, axis=1).swaplevel(axis=1).sort_index(axis=1)
        
        #=======================================================================
        # meta stats 
        #=======================================================================
        meta_d = dict()
        d = dict()
        dindex2 = dx_raw.loc[:, idx['expo',:]].droplevel(0, axis=1)
        
        d['count'] = dindex2['depth'].groupby(level=gkeys).count()
        
        #=======================================================================
        # depth stats
        #=======================================================================
        gbo = dindex2['depth'].groupby(level=gkeys)
        
        d['dry_cnt'] = gbo.agg(lambda x: x.eq(0).sum())
        
        d['wet_cnt'] = gbo.agg(lambda x: x.ne(0).sum())
 
        # loop and get each from the grouper
        for statName in ['mean', 'min', 'max', 'var']:
            d[statName] = getattr(gbo, statName)()
            
        meta_d['depth'] = pd.concat(d, axis=1)
        #=======================================================================
        # asset count stats
        #=======================================================================
        gbo = dindex2[scale_cn].groupby(level=gkeys)
        
        d = dict()
        
        d['mode'] = gbo.agg(lambda x:x.value_counts().index[0])
        for statName in ['mean', 'min', 'max', 'sum']:
            d[statName] = getattr(gbo, statName)()
 
        meta_d['assets'] = pd.concat(d, axis=1)
        
        #=======================================================================
        # collect all
        #=======================================================================
        rlib['meta'] = pd.concat(meta_d, axis=1)
        
        rdx = pd.concat(rlib, axis=1, names=['cat', 'var', 'stat'])
        
        #=======================================================================
        # write
        #=======================================================================
        log.info('finished w/ %s' % str(rdx.shape))
        if write:
            ofp = os.path.join(self.out_dir, 'lossSmry_%i_%s.csv' % (
                  len(dx_raw), self.longname))
            
            if os.path.exists(ofp): assert self.overwrite
            
            rdx.to_csv(ofp)
            
            log.info('wrote %s to %s' % (str(rdx.shape), ofp))
        
        return rdx
            
        """
        view(rdx)
        mindex.names
        view(dx_raw)
        """
    




    def get_performance(self,
                 agg_dx, true_dx,
                 aggMethod='mean',
                 confusion=False,
                 logger=None,
                  ):
        if logger is None: logger=self.logger
        log=logger.getChild('get_perf')
        #===============================================================
        # #aggregate
        #===============================================================
        #collapse trues
        tgdx2 = self.get_aggregate(true_dx, mindex=agg_dx.index, aggMethod=aggMethod, logger=log)
        #===============================================================
        # standard errors
        #===============================================================
        eW = ErrorCalcs(pred_ser=agg_dx, true_ser=tgdx2, logger=log)
        """
        
        eW.data_retrieve_hndls.keys()
        
        """
        err_d = eW.get_all(dkeys_l=['bias', 'bias_shift', 'meanError', 'meanErrorAbs', 'RMSE', 'pearson'])
        stats_d = eW.get_stats()
        rser = pd.Series({**stats_d, **err_d}) #convert remainers to a series
        #===============================================================
        # confusion
        #===============================================================
        if confusion:
            """I think it makes more sense to report confusion on the expanded trues"""
            #disaggregate
            gdx1_exp = self.get_aggregate(agg_dx, mindex=true_dx.index, 
                aggMethod=self.agg_inv_d[aggMethod], logger=log)
            #calc matrix
            cm_df, cm_dx = ErrorCalcs(pred_ser=gdx1_exp, true_ser=true_dx, logger=log).get_confusion()
            rser = rser.append(cm_dx.droplevel(['pred', 'true']).iloc[:, 0])
        #===============================================================
        # wrap
        #===============================================================
        return rser.astype(float).round(3)

    def write_suite_smry(self, #write a summary of the model suite
                         
                         #data
                         dx_raw=None,
                         true_dx_raw=None,
                         modelID_l=None, #models to include
                         baseID=0,
                         
                         #control
                         agg_d = {'rloss':'mean', 'rsamps': 'mean', 'tvals':'sum', 'tloss':'sum'}, #aggregation methods
                         ):
        #=======================================================================
        # defaults
        #=======================================================================
        log = self.logger.getChild('suite_smry')
        
        
        #=======================================================================
        # retrieve
        #=======================================================================
        idn = self.idn
        #if baseID is None: baseID=self.baseID
 
        if dx_raw is None:
            dx_raw = self.retrieve('outs')
            
        if true_dx_raw is None:
            true_d = self.retrieve('trues')
            true_dx_raw = true_d[baseID]
            
        if modelID_l is None:
            modelID_l = dx_raw.index.unique(0).tolist()
            
        #assert baseID == modelID_l[0]
        
        
        
        #=======================================================================
        # prep data
        #=======================================================================
 
        #=======================================================================
        # dx = dx_raw.loc[idx[modelID_l, :, :, :, :], :]
        # true_dx = true_dx_raw.loc[idx[modelID_l, :,:, :, :, :], :]
        #=======================================================================
        
        
        
        pars_df = self.retrieve('model_pars')
        
        dx = self.join_meta_indexers(dx_raw = dx_raw, 
                                #meta_indexers =  set(pars_df.columns.tolist()),
                                modelID_l=modelID_l)
        
        true_dx = self.join_meta_indexers(dx_raw = true_dx_raw, 
                                #meta_indexers =  set(pars_df.columns.tolist()),
                                modelID_l=modelID_l)
        
        #=======================================================================
        # collapse iters
        #=======================================================================
        dx = dx.groupby(level=0, axis=1).first()
        true_dx = true_dx.groupby(level=0, axis=1).first()
        

        
        #=======================================================================
        # define func
        #=======================================================================
        def calc(grp_cols):
            rdx = None
            
            true_gb = true_dx.groupby(level=grp_cols, axis=0)
            for i, (gkeys, gdx0) in enumerate(dx.groupby(level=grp_cols, axis=0)): #loop by axis data
                #===================================================================
                # setup
                #===================================================================            
                #keys_d = dict(zip(grp_cols, gkeys))
                tgdx0 = true_gb.get_group(gkeys)
                log.info('on %s'%str(gkeys))
                #===================================================================
                # by column
                #===================================================================
                res_d = dict()
                for coln, aggMethod in agg_d.items():
    
                    res_d[coln] = self.get_performance(gdx0[coln], tgdx0[coln],
                                                       confusion = coln=='rsamps',
                                        aggMethod=aggMethod, logger=log.getChild(str(gkeys)))
     
                    
                #===================================================================
                # combine
                #===================================================================
                rserx = pd.concat(res_d, names=['var', 'metric']).rename(gkeys)
                
                if rdx is None:
                    rdx =rserx.to_frame().T
                else:
                    rdx = rdx.append(rserx)
                
            #=======================================================================
            # join meta
            #=======================================================================
            rdx.index.set_names(grp_cols, inplace=True)
 
            
            return rdx.join(pd.concat({'parameters':pars_df}, axis=1), on=idn)
        
        #=======================================================================
        # per study area
        #=======================================================================
        log.info('on %i models w/ %s'%(len(modelID_l), str(dx_raw.shape)))
        rdx_sa = calc([idn, 'studyArea'])
        
        #=======================================================================
        # total model
        #=======================================================================
        rdx_tot = calc([idn])
        
        """
        view(rdx)
        """
        
        #=======================================================================
        # write
        #=======================================================================
        ofp = os.path.join(self.out_dir, 'suite_smry_%s.xls'%self.longname)
        
        with pd.ExcelWriter(ofp) as writer:
            for tabnm, df in {idn:rdx_tot, 'studyAreas':rdx_sa}.items():
                df.reset_index(drop=False).to_excel(writer, sheet_name=tabnm, index=True, header=True)
        
        log.info('wrote %i tabs to %s'%(2, ofp))
        
        return ofp
                
 
        
        
    
    def write_resvlay(self,  #attach some data to the gridded finv
                  # data control   
                  modelID_l=None,
                  dx_raw=None,finv_agg_fps=None,
                  dkey='tloss',
                  stats = ['mean', 'min', 'max', 'var'], #stats for compressing iter data
                    
                    # output config
 
                    out_dir=None,
                    ):
        """
        folder: tloss_spatial
            sub-folder: studyArea
                sub-sub-folder: event
                    file: one per grid size 
        """
            
        #=======================================================================
        # defaults
        #=======================================================================
        log = self.logger.getChild('write_resvlay')
 
        if out_dir is None:out_dir = self.out_dir
        idn=self.idn
        gcn = 'gid'
        #=======================================================================
        # #retrieve child data
        #=======================================================================
        # errors
        if dx_raw is None:
            dx_raw = self.retrieve('outs')
 
        
        # geometry
        if finv_agg_fps is None:
            finv_agg_fps = self.retrieve('finv_agg_fps')
            
        #=======================================================================
        # slice
        #=======================================================================
        if modelID_l is None:
            modelID_l = dx_raw.index.unique(idn).tolist()
            
 
        
        
        dxind = dx_raw.loc[idx[modelID_l, :, :, :, :], idx[dkey, :]].droplevel(0, axis=1)
        
 
        
        #=======================================================================
        # loop and write--------
        #=======================================================================
        log.info('on \'%s\' w/ %i' % (dkey, len(dx_raw)))
        #=======================================================================
        # loop on study area
        #=======================================================================
        ofp_l = list()
        gnames = [idn, 'studyArea', 'event']
        for keys, gdx0 in dxind.groupby(level=gnames):
 
            keys_d = dict(zip(gnames, keys))
            mstore = QgsMapLayerStore()
            log.info(keys_d)
            
            
            #===================================================================
            # prep
            #===================================================================
            assert len(gdx0.index.unique('tag'))==1
            tag = gdx0.index.unique('tag')[0]
            keys_d['tag'] = tag
            
            #drop extra indicies
            gdf1 = gdx0.droplevel([0,1,2,3])
                
            #===================================================================
            # retrieve spatial data
            #===================================================================
            # get vlay
            finv_fp = finv_agg_fps[keys_d[idn]][keys_d['studyArea']]
            finv_vlay = self.vlay_load(finv_fp, logger=log, set_proj_crs=True)
            mstore.addMapLayer(finv_vlay)
            
            #get geo
            geo_d = vlay_get_geo(finv_vlay)
            
            #===================================================================
            # re-key
            #===================================================================
            fid_gid_d = vlay_get_fdata(finv_vlay, fieldn=gcn)
            
            #chekc keys
            miss_l = set(gdf1.index).difference(fid_gid_d.values())
            if not len(miss_l)==0:
                raise Error('missing %i/%i keys on %s'%(len(miss_l), len(gdf1), keys_d))
            
            #rekey data
            fid_ser = pd.Series({v:k for k,v in fid_gid_d.items()}, name='fid')
            gdf2 = gdf1.join(fid_ser).set_index('fid')
            
            """
            gdf2.columns
            """
            
            #===================================================================
            # compute stats
            #===================================================================
            d = {'mean':gdf2.mean(axis=1)}
            
            if len(gdf2.columns)>0:
                d = {k:getattr(gdf2, k)(axis=1) for k in stats}
                
            gdf3 = pd.concat(d.values(), keys=d.keys(), axis=1).sort_index()
            """
            view(gdx1)
            """
 
            #===================================================================
            # build layer
            #===================================================================
            layname = 'm' + '_'.join([str(e).replace('_', '') for e in keys_d.values()])
            vlay = self.vlay_new_df(gdf3, geo_d=geo_d, layname=layname, logger=log,
                                    crs=finv_vlay.crs(),  # session crs does not match studyAreas
                                    )
            mstore.addMapLayer(vlay)
            #===================================================================
            # write layer
            #===================================================================
            # get output directory
            od = os.path.join(out_dir, tag)
 
            if not os.path.exists(od):
                os.makedirs(od)
                
            s = '_'.join([str(e) for e in keys_d.values()])
            ofp = os.path.join(od, self.longname + '_%s_'%dkey + s + '.gpkg') 
 
            ofp_l.append(self.vlay_write(vlay, ofp, logger=log))
            
            #===================================================================
            # wrap
            #===================================================================
            mstore.removeAllMapLayers()
            
        #=======================================================================
        # write meta
        #=======================================================================
        log.info('finished on %i' % len(ofp_l))
        
 
        return ofp_l
            

    
    #===========================================================================
    # PLOTTERS-------------
    #===========================================================================
    def plot_model_smry(self, #plot a summary of data on one model
                        modelID,
                        dx_raw=None,
                        
                        #plot config
                        plot_rown='dkey',
                        plot_coln='event',
                        #plot_colr = 'event',
                        xlims_d = {'rsamps':(0,5)}, #specyfing limits per row
                        
                        #errorbars
                        #qhi=0.99, qlo=0.01,
                        
                         #plot style
                         drop_zeros=True,
                         colorMap=None,
                        ):
        #=======================================================================
        # defaults
        #=======================================================================
        log = self.logger.getChild('plot_model_smry_%i'%modelID)
        if dx_raw is None: dx_raw = self.retrieve('outs')
        if colorMap is None: colorMap=self.colorMap_d['dkey_range']
        
        #=======================================================================
        # data prep
        #=======================================================================
        dx = dx_raw.loc[idx[modelID, :, :, :, :], :].droplevel([0,1])
        mdex = dx.index
        log.info('on %s'%str(dx.shape))
        
        tag = dx_raw.loc[idx[modelID, :, :, :, :], :].index.remove_unused_levels().unique('tag')[0]
        #=======================================================================
        # setup the figure
        #=======================================================================
        plt.close('all')
        col_keys =mdex.unique(plot_coln).tolist()
        row_keys = dx.columns.unique(plot_rown).tolist() 
 
        fig, ax_d = self.get_matrix_fig(row_keys,col_keys, 
                                    figsize_scaler=4,
                                    constrained_layout=True,
                                    sharey='none', sharex='row',  # everything should b euniform
                                    fig_id=0,
                                    set_ax_title=True,
                                    )
        fig.suptitle('Model Summary for \'%s\''%(tag))
        
        # get colors
        #cvals = dx_raw.index.unique(plot_colr)
        cvals = ['min', 'mean', 'max']
        cmap = plt.cm.get_cmap(name=colorMap) 
        newColor_d = {k:matplotlib.colors.rgb2hex(cmap(ni)) for k, ni in dict(zip(cvals, np.linspace(0, 1, len(cvals)))).items()}
        
        #===================================================================
        # loop and plot
        #===================================================================
        for col_key, gdx1 in dx.groupby(level=[plot_coln]):
            keys_d = {plot_coln:col_key}
            
            for row_key, gdx2 in gdx1.groupby(level=[plot_rown], axis=1):
                keys_d[plot_rown] = row_key
                ax = ax_d[row_key][col_key]
                
                #===============================================================
                # prep data
                #===============================================================
                gb = gdx2.groupby('dkey', axis=1)
                
                d = {k:getattr(gb, k)() for k in cvals}
                err_df = pd.concat(d, axis=1).droplevel(axis=1, level=1)
                
                bx = err_df==0
                if drop_zeros:                    
                    err_df = err_df.loc[~bx.all(axis=1), :]
                    
                if keys_d['dkey'] in xlims_d:
                    xlims = xlims_d[keys_d['dkey']]
                else:
                    xlims=None
                #ax.set_xlim(xlims)
                
                #===============================================================
                # loop and plot bounds
                #===============================================================
                for boundTag, col in err_df.items():
                    ax.hist(col.values, 
                            color=newColor_d[boundTag], alpha=0.3, 
                            label=boundTag, 
                            density=False, bins=40, 
                            range=xlims,
                            )
                    
                    if len(gdx2.columns.get_level_values(1))==1:
                        break #dont bother plotting bounds
                    
                    
                #===============================================================
                # #label
                #===============================================================
                # get float labels
                meta_d = {'cnt':len(err_df), 'zero_cnt':bx.all(axis=1).sum(), 'drop_zeros':drop_zeros,
                          'min':err_df.min().min(), 'max':err_df.max().max()}
 
                ax.text(0.4, 0.9, get_dict_str(meta_d), transform=ax.transAxes, va='top', fontsize=8, color='black')
                    
                
                #===============================================================
                # styling
                #===============================================================   
                ax.set_xlabel(row_key)                 
                # first columns
                if col_key == col_keys[0]:
                    """not setting for some reason"""
                    ax.set_ylabel('count')
 
                # last row
                if row_key == row_keys[-1]:
                    ax.legend()
 
        #=======================================================================
        # wrap
        #=======================================================================
        log.info('finsihed')
        
        return self.output_fig(fig,
                               out_dir = os.path.join(self.out_dir, tag), 
                               fname='model_smry_%03d_%s' %(modelID, self.longname))
        
 
    """
    plt.show()
    """
 
 
        
        


    def plot_dkey_mat2(self, #flexible plotting of model results (one dkey)
                  
                    #data control
                    ax_d=None,
                    dkey='tvals',#column group w/ values to plot
 
                    xlims = None,ylims=None,
                    modelID_l = None, #optinal sorting list
 
                    #qhi=0.99, qlo=0.01, #just taking the mean
                    drop_zeros=True, 
                    slice_d = {}, #special slicing

                    
                    #data
                    dx_raw=None, #combined model results                    
                    
                    #plot config
                    plot_type='hist', 
                    plot_rown='aggLevel',
                    plot_coln='resolution',
                    plot_colr=None,                    
 
                    plot_bgrp=None, #grouping (for plotType==bars)

                     
                    #histwargs
                    bins=20, rwidth=0.9, 
                    mean_line=True, #plot a vertical line on the mean
                    density=False,
                    baseID=None, #for gaussian_kde... duplicate the baseline
 
 
                    #meta labelling
                    meta_txt=True, #add meta info to plot as text
                    meta_func = lambda meta_d={}, **kwargs:meta_d, #lambda for calculating additional meta information (add_meta=True)        
                    write_meta=False, #write all the meta info to a csv            
                    
                    #plot style                    
                    colorMap=None, title=None, val_lab=None, grid=False,
                    sharey='none',sharex='none',
                    
                    #output
                    fmt='svg',
 
                    **kwargs):
        """"
        similar to plot_dkey_mat (more flexible)
        similar to plot_err_mat (1 dkey only)
        
        
        """
        
        #=======================================================================
        # defaults
        #=======================================================================
        log = self.logger.getChild('plot_dkey_mat2')
 
        idn = self.idn
 
        #retrieve data
        if dx_raw is None:
            dx_raw = self.retrieve('outs')
 
            
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
        if not baseID is None: assert plot_type=='gaussian_kde'
        assert not plot_rown==plot_coln
            
        #plot style
                 
        if title is None:
            title = '\'%s\' values'%dkey
                
            for name, val in slice_d.items(): 
                title = title + ' %s=%s'%(name, val) 
                
        if colorMap is None: colorMap = self.colorMap_d[plot_colr]
        
        if val_lab is None: val_lab=dkey
        
        if plot_type in ['violin', 'bar']:
            assert xlims is None
        
 
        log.info('on \'%s\' (%s x %s)'%(dkey, plot_rown, plot_coln))
        #=======================================================================
        # data prep
        #=======================================================================

        
        meta_indexers = set([plot_rown, plot_coln])
        if not plot_bgrp is None:
            meta_indexers.add(plot_bgrp)
        
        for name in slice_d.keys(): meta_indexers.add(name)
        
        #add requested indexers
        dx = self.join_meta_indexers(dx_raw = dx_raw.loc[:, idx[dkey, :]], 
                                meta_indexers = meta_indexers.copy(),
                                modelID_l=modelID_l)
 
        log.info('on %s'%str(dx.shape))
        
        
 
        #=======================================================================
        # subsetting
        #=======================================================================
        for name, val in slice_d.items():
            assert name in dx.index.names
            bx = dx.index.get_level_values(name) == val
            assert bx.any()
            dx = dx.loc[bx, :]

            log.info('w/ %s=%s slicing to %i/%i'%(
                name, val, bx.sum(), len(bx)))
 
        #=======================================================================
        # #collapse iters
        #=======================================================================
        
        ser_dx = dx.mean(axis=1)
        
        #=======================================================================
        # drop zeros
        #=======================================================================
        if drop_zeros:
            bx = ser_dx==0
            if bx.any():
                log.warning('dropping %i/%i zeros'%(bx.sum(), len(bx)))
                ser_dx = ser_dx.loc[~bx]
                
        #=======================================================================
        # set base
        #=======================================================================
        if not baseID is None:
            """
            todo: set up for a plot w/ multiple baseline
                (e.g., keyed by lookup values rather than modelID)
                be careful of studyArea
            """
            bx = ser_dx.index.get_level_values('modelID')==baseID
            base_sx = ser_dx.loc[bx].copy()
            
            if 'studyArea' in [plot_coln, plot_rown]:
                base_gb = base_sx.groupby(level='studyArea')
            else:
                b_ar = base_sx.values
            
            base_mods_d = dict()
        else:
            base_sx= None
        
        
        meta_d1 = {'iters':len(dx.columns), 'drop_zeros':drop_zeros}
        meta_d1.update(slice_d)
        #=======================================================================
        # setup the figure
        #=======================================================================
        mdex = dx.index
        plt.close('all')
 
        col_keys =mdex.unique(plot_coln).tolist()
        row_keys = mdex.unique(plot_rown).tolist()
 
        if ax_d is None:
            fig, ax_d = self.get_matrix_fig(row_keys, col_keys,
                                        figsize_scaler=4,
                                        constrained_layout=True,
                                        sharey=sharey, 
                                        sharex=sharex,  
                                        fig_id=0,
                                        set_ax_title=True,
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
        color_d = self.get_color_d(ckeys, colorMap=colorMap)
        
        #=======================================================================
        # loop and plot
        #=======================================================================
        first=True
        meta_dx=None
        for gkeys, gdx0 in ser_dx.groupby(level=[plot_coln, plot_rown]): #loop by axis data
            
            #===================================================================
            # setup
            #===================================================================
            keys_d = dict(zip([plot_coln, plot_rown], gkeys))
            ax = ax_d[gkeys[1]][gkeys[0]]
            log.info('on %s'%keys_d)
            
            #===================================================================
            # data prep----------
            #===================================================================

 
            meta_d = {**{ 'modelIDs':str(list(gdx0.index.unique(idn)))}, **meta_d1}
 
            
            if not plot_bgrp is None:
                data_d = {k:df.values for k,df in gdx0.groupby(level=plot_bgrp)}
            else:
                data_d = {dkey:gdx0.values}
                
                
            """
            fig.show()
            """
             
            #===================================================================
            # add plots--------
            #===================================================================
            if mean_line:
                mval =gdx0.mean().mean()
            else: mval=None 
            
            md1 = self.ax_data(ax, data_d,
                               plot_type=plot_type, 
                               bins=bins, rwidth=rwidth, mean_line=mval, hrange=xlims, density=density,
                               color_d=color_d, logger=log,label_key=plot_bgrp, **kwargs) 
 
            meta_d.update(md1)
            labels = ['%s=%s'%(plot_bgrp, k) for k in data_d.keys()]
            
            #===================================================================
            # base plot
            #===================================================================
            if not base_sx is None:
                if 'studyArea' in keys_d: studyArea = keys_d['studyArea']
                else: studyArea = 'all'
                #set the base
                if not studyArea in base_mods_d:
                    if 'studyArea' in keys_d:
                        b_ar = base_gb.get_group(keys_d['studyArea']).values
                    assert len(b_ar)>0
                    kde = scipy.stats.gaussian_kde(b_ar,bw_method='scott',weights=None)
                    
                    xvals = np.linspace(b_ar.min()+.01, b_ar.max(), 1000)
                    yvals = kde(xvals)
                    
                    base_mods_d[studyArea] = (xvals, yvals)
                
                
                #plot it
                xvals, yvals = base_mods_d[studyArea]
                ax.plot(xvals, yvals, color='black', label='baseline', linestyle='dashed')
                    
                
            
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
                ax.text(0.1, 0.9, get_dict_str(meta_d), transform=ax.transAxes, va='top', fontsize=8, color='black',
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", lw=0.0,alpha=0.5 ),
                        )
            
            
            #===================================================================
            # collect meta 
            #===================================================================
            meta_serx = pd.Series(meta_d, name=gkeys)
            
            if meta_dx is None:
                meta_dx = meta_serx.to_frame().T
                meta_dx.index.set_names(keys_d.keys(), inplace=True)
            else:
                meta_dx = meta_dx.append(meta_serx)
                
            first=False
                
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
 
                
                # first row
                if row_key == row_keys[0]:
                    #last col
                    if col_key == col_keys[-1]:
                        if plot_type in ['hist', 'gaussian_kde']:
                            ax.legend()
                
                        
                # first col
                if col_key == col_keys[0]:
                    if plot_type in ['hist', 'gaussian_kde']:
                        if density:
                            ax.set_ylabel('density')
                        else:
                            ax.set_ylabel('count')
                    elif plot_type in ['box', 'violin']:
                        ax.set_ylabel(val_lab)
                
                #last row
                if row_key == row_keys[-1]:
                    if plot_type in ['hist', 'gaussian_kde']:
                        ax.set_xlabel(val_lab)
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
        plt.show()
        """
 
        fname = 'values_%s_%s_%sX%s_%s_%s' % (
            title.replace(' ','').replace('\'',''),
             plot_type, plot_rown, plot_coln, val_lab, self.longname)
                
        fname = fname.replace('=', '-')
        
        if write_meta:
            ofp =  os.path.join(self.out_dir, fname+'_meta.csv')
            meta_dx.to_csv(ofp)
            log.info('wrote meta_dx %s to \n    %s'%(str(meta_dx.shape), ofp))
               
        
        return self.output_fig(fig, fname=fname, fmt=fmt)
 
 
    def plot_dkeyS_mat(self, #flexible plotting of model results (dkeys as dimension)
                  
                    #data control
                    ax_d=None,
                    dkey_l=['rloss', 'rsamps'],#column group w/ values to plot
 
 
                    modelID_l = None, #optinal sorting list
 
                    #qhi=0.99, qlo=0.01, #just taking the mean
                    drop_zeros=True, 
                    slice_d = {}, #special slicing

                    
                    #data
                    dx_raw=None, #combined model results                    
                    
                    #plot config
                    plot_type='hist', 
                    plot_rown='aggLevel',
                    plot_coln='dkey',
                    plot_colr=None,                    
 
                    plot_bgrp=None, #grouping (for plotType==bars)

                     
                    #histwargs
                    bins=20, rwidth=0.9, 
                    mean_line=True, #plot a vertical line on the mean
                    density=False,
 
 
                    #meta labelling
                    meta_txt=True, #add meta info to plot as text
                    meta_func = lambda meta_d={}, **kwargs:meta_d, #lambda for calculating additional meta information (add_meta=True)        
                    write_meta=False, #write all the meta info to a csv            
                    
                    #plot style                    
                    colorMap=None, title=None, val_lab=None,
                    sharey=None,sharex=None,xlims=None,
                    
                    #output
                    fmt='svg',
 
                    **kwargs):
        """"
         This is pretty strange... 
         upon reflection, doesnt seem like a good idea squash multiple dkeys like this
        
        
        """
        
        #=======================================================================
        # defaults
        #=======================================================================
        log = self.logger.getChild('plot_dkeyS_mat')
 
        idn = self.idn
 
        #retrieve data
        if dx_raw is None:
            dx_raw = self.retrieve('outs')
 
            
        #plot keys
        if plot_colr is None: 
            plot_colr=plot_bgrp
        
        if plot_colr is None: 
            plot_colr=plot_rown
            
        if plot_bgrp is None:
            plot_bgrp = plot_colr
            
        

        assert not plot_rown==plot_coln
        assert not plot_bgrp=='dkey', 'doesnt make sense to plot muttiple dkeys on the same axis'
        
        if plot_type in ['hist', 'gaussian_kde']:
            assert plot_coln=='dkey', 'need to plot data dimensin on columns'
            
        if not xlims is None: assert sharex=='all'
            
        #plot style
                 
        if title is None:
            title = '\'%s\' values'%' & '.join(dkey_l)
                
            for name, val in slice_d.items(): 
                title = title + ' %s=%s'%(name, val) 
                
        if colorMap is None: colorMap = self.colorMap_d[plot_colr]
        
        if val_lab is None: val_lab=' & '.join(dkey_l)
 
        log.info('on \'%s\' (%s x %s)'%(dkey_l, plot_rown, plot_coln))
        #=======================================================================
        # data prep
        #=======================================================================

        
        meta_indexers = set([plot_rown, plot_coln])
        if not plot_bgrp is None:
            meta_indexers.add(plot_bgrp)
            
        assert 'dkey' in meta_indexers, 'must include dkey in dimensions'
        meta_indexers.remove('dkey')
        
        
        #add requested indexers
        dx0 = self.join_meta_indexers(dx_raw = dx_raw.loc[:, idx[dkey_l, :]], 
                                meta_indexers = meta_indexers.copy(),
                                modelID_l=modelID_l)
        
        #clear iters
        dx1 = dx0.mean(level=0, axis=1)
        
        #stack
        ser_dx = dx1.stack().rename('values')
 
        log.info('on %i'%len(ser_dx))
 
        #=======================================================================
        # subsetting
        #=======================================================================
        for name, val in slice_d.items():
            assert name in ser_dx.index.names
            bx = ser_dx.index.get_level_values(name) == val
            assert bx.any()
            ser_dx = ser_dx.loc[bx]

            log.info('w/ %s=%s slicing to %i/%i'%(
                name, val, bx.sum(), len(bx)))
 
 
        
        #=======================================================================
        # drop zeros
        #=======================================================================
        if drop_zeros:
            bx = ser_dx==0
            if bx.any():
                log.warning('dropping %i/%i zeros'%(bx.sum(), len(bx)))
                ser_dx = ser_dx.loc[~bx]
        
        meta_d1 = {'iters':len(dx0.columns)/len(dkey_l), 'drop_zeros':drop_zeros}
        meta_d1.update(slice_d)
        #=======================================================================
        # setup the figure
        #=======================================================================
        mdex = ser_dx.index
        plt.close('all')
 
        col_keys =mdex.unique(plot_coln).tolist()
        row_keys = mdex.unique(plot_rown).tolist()
 
        if ax_d is None:
            fig, ax_d = self.get_matrix_fig(row_keys, col_keys,
                                        figsize_scaler=4,
                                        constrained_layout=True,
                                        sharey=sharey, 
                                        sharex=sharex,  
                                        fig_id=0,
                                        set_ax_title=True,
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
        color_d = self.get_color_d(ckeys, colorMap=colorMap)
        
        #=======================================================================
        # loop and plot
        #=======================================================================
        meta_dx=None
        for gkeys, gdx0 in ser_dx.groupby(level=[plot_coln, plot_rown]): #loop by axis data
            
            #===================================================================
            # setup
            #===================================================================
            keys_d = dict(zip([plot_coln, plot_rown], gkeys))
            ax = ax_d[gkeys[1]][gkeys[0]]
            log.info('on %s'%keys_d)
            
            #===================================================================
            # data prep----------
            #===================================================================

 
            meta_d = {**{ 'modelIDs':str(list(gdx0.index.unique(idn)))}, **meta_d1}
            data_d = {k:df.values for k,df in gdx0.groupby(level=plot_bgrp)}
 
                

             
            #===================================================================
            # add plots--------
            #===================================================================
            if mean_line:
                mval =gdx0.mean().mean()
            else: mval=None 
            
            md1 = self.ax_data(ax, data_d,
                               plot_type=plot_type, 
                               bins=bins, rwidth=rwidth, mean_line=mval, hrange=xlims, density=density,
                               color_d=color_d, logger=log, **kwargs) 
 
            meta_d.update(md1)
            labels = ['%s=%s'%(plot_bgrp, k) for k in data_d.keys()]
            #===================================================================
            # post format 
            #===================================================================
            
            ax.set_title(' & '.join(['%s:%s' % (k, v) for (k, v) in keys_d.items()]))
            
            if not xlims is None:
                ax.set_xlim(xlims)
            #===================================================================
            # meta  text
            #===================================================================
            """for bars, this ignores the bgrp key"""
            meta_d = meta_func(logger=log, meta_d=meta_d, pred_ser=gdx0)
            if meta_txt:
                ax.text(0.1, 0.9, get_dict_str(meta_d), transform=ax.transAxes, va='top', fontsize=8, color='black')
            
            
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
                

 
                
                # first row
                if row_key == row_keys[0]:
                    #last col
                    if col_key == col_keys[-1]:
                        if plot_type in ['hist', 'gaussian_kde']:
                            ax.legend()
                
                        
                # first col
                if col_key == col_keys[0]:
                    if plot_type in ['hist', 'gaussian_kde']:
                        if density:
                            ax.set_ylabel('frequency')
                        else:
                            ax.set_ylabel('count')
                    elif plot_type in ['box', 'violin']:
                        ax.set_ylabel(col_key)
                
                #last row
                if row_key == row_keys[-1]:
                    if plot_type in ['hist', 'gaussian_kde']:
                        ax.set_xlabel(col_key)
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
        plt.show()
        """
 
        fname = 'values_%s_%s_%sX%s_%s_%s' % (
            title.replace(' ','').replace('\'',''),
             plot_type, plot_rown, plot_coln, val_lab, self.longname)
                
        fname = fname.replace('=', '-').replace(' ','').replace('&','-')
        
        if write_meta:
            ofp =  os.path.join(self.out_dir, fname+'_meta.csv')
            meta_dx.to_csv(ofp)
            log.info('wrote meta_dx %s to \n    %s'%(str(meta_dx.shape), ofp))
               
        
        return self.output_fig(fig, fname=fname, fmt=fmt)
 
    


    def dprep_slice(self, #convenience for index slicing from matching values 
                    dx_raw, slice_d, logger=None):
        if logger is None: logger=self.logger
        log=logger.getChild('slice') 
        
 
        
        dx = dx_raw.copy()
        for name, val in slice_d.items():
            mdex = dx.index
            assert name in mdex.names, 'slice dimension (%s) not present' % name
            bx = mdex.get_level_values(name) == val
            assert bx.any(), 'failed to match any %s=%s' % (name, val)
            dx = dx.loc[bx, :]
 
            log.info('w/ %s=%s slicing to %i/%i' % (name, val, bx.sum(), len(bx)))
            
        return dx
        
 

    def plot_err_mat(self, #flexible plotting of model results vs. true in a matrix
                  
                    #data selection
                    dkey='tvals',#column group w/ values to plot
                    baseID=0,
                    modelID_l = None, #optinal sorting list
                    slice_d = {}, #special slicing
                    
                    #parameters [true vs. agg relation]
                    aggMethod='mean', #method to use for aggregating the true values (down to the gridded)
                    base_index='agg', #which index to treat as base values for calculating metrics
                        #generally we've been using the aggregated (collapsing trues)
                        #but maybe we should be using trues more? (expanding aggs)
                     
                    #drop_zeros=True, #must always be false for the matching to work
                    
                    
                    #data
                    true_dx_raw=None, #base values (indexed to raws per model)
                    dx_raw=None, #combined model results                    
                    
                    #plot config
                    plot_type='scatter', 
                    plot_rown='aggLevel',
                    plot_coln='resolution',
                    plot_colr=None,
                    
                    #plot config [bars and violin]
                    plot_bgrp=None, #grouping (for plotType==bars)
                    bar_labels=True,
                                        
                    #ErrorCalcs
                    err_type='absolute', #ErrorCalcs. what type of errors to calculate (for plot_type='bars')
                        #absolute: modelled - true
                        #relative: absolute/true
                    normed=True, #ErrorCalcs
                    
                    #plot config [density]
                    bins=50, vmin=None, vmax=None,
 
                    #meta labelling
                    meta_txt=True, #add meta info to plot as text
                    meta_func = lambda meta_d={}, **kwargs:meta_d, #lambda for calculating additional meta information (add_meta=True)        
                    write_meta=False, #write all the meta info to a csv            
                    
                    #plot style                    
                    colorMap=None, title=None,xlims = None,
                    sharey=None,sharex=None,zero_line=False,
 
                    
                    #plot output
                    fmt=None,
 
                    **kwargs):
        """"
        generally 1 modelId per panel
        TODO: 
        pass meta as a lambda
            gives more customization outside of this function (should simplify)
        
        
        """
        
        #=======================================================================
        # defaults----------
        #=======================================================================
        log = self.logger.getChild('plot_err_mat')
 
        idn = self.idn
 
        #=======================================================================
        # #retrieve data
        #=======================================================================
        if dx_raw is None:
            dx_raw = self.retrieve('outs')
            
        if true_dx_raw is None:
            true_d = self.retrieve('trues')
            true_dx_raw = true_d[baseID]
            
        #=======================================================================
        # #plot keys
        #=======================================================================
        if plot_colr is None:
            if err_type=='confusion':
                plot_colr='confusion'
            else:
                plot_colr=plot_bgrp
        
        if plot_colr is None and plot_type!='scatter_density': 
            plot_colr=plot_rown
            
        if plot_bgrp is None:
            plot_bgrp = plot_colr
            
        
        #=======================================================================
        # #logic checks
        #=======================================================================
        #assert isinstance(plot_bgrp, str) 
            
        assert not plot_rown==plot_coln
        
        if plot_type=='scatter_density':
            assert plot_colr is None
            assert plot_bgrp is None
            
        if err_type=='confusion':
            assert plot_colr ==err_type
        
        assert not plot_bgrp==err_type
        #=======================================================================
        # #plot style
        #=======================================================================
        if sharey is None:
            if 'scatter' in plot_type:
                sharey='none'
            else:
                sharey='all'
                
        if sharex is None:
            if 'scatter' in plot_type:
                sharex='none'
            else:
                sharex='all'
                
        if title is None:
            if not plot_type=='bars':
                title = '\'%s\' errors'%dkey
            else:
                title = '\'%s\' %s'%(dkey, err_type)
                
            for name, val in slice_d.items(): 
                title = title + ' %s=%s'%(name, val) 
                
        if colorMap is None:
            if plot_type in ['scatter_density', 'hist2d']: 
                colorMap = self.colorMap_d['density']
            if err_type=='confusion':
                colorMap = self.colorMap_d['confusion']
            else:
                colorMap = self.colorMap_d[plot_colr]
                
        
        #=======================================================================
        # #outputs 
        #=======================================================================
        if fmt is None:
            if 'scatter' in plot_type:
                fmt='png'
            else:
                fmt='svg'
 
        log.info('on \'%s\' (%s x %s)'%(dkey, plot_rown, plot_coln))
        #=======================================================================
        # data prep
        #=======================================================================
        assert_func(lambda: self.check_mindex_match(true_dx_raw.index, dx_raw.index), msg='raw vs trues')
        
        meta_indexers = set([plot_rown, plot_coln,])
        if not plot_bgrp is None:
            meta_indexers.add(plot_bgrp)
        
        for k in slice_d.keys(): meta_indexers.add(k) #add any required slicers
        
        #=======================================================================
        # #collaps iters
        #=======================================================================
        sx_raw = dx_raw.loc[:, idx[dkey, :]].groupby(level=0, axis=1).first()
        
        txs_raw = true_dx_raw.loc[:, idx[dkey, :]].groupby(level=0, axis=1).first()
        
        #=======================================================================
        # join meta inddexers
        #=======================================================================
        sx0r = self.join_meta_indexers(dx_raw = sx_raw,meta_indexers = meta_indexers,modelID_l=modelID_l).iloc[:,0]
 
        tsx0r = self.join_meta_indexers(dx_raw = txs_raw,meta_indexers = meta_indexers,modelID_l=modelID_l).iloc[:,0]
 
        #=======================================================================
        # subsetting
        #=======================================================================.
        sx1r = self.dprep_slice(sx0r, slice_d, logger=log)
        
        tsx1r = self.dprep_slice(tsx0r, slice_d, logger=log)
        
        #=======================================================================
        # collapse/expand
        #=======================================================================
        if base_index=='agg':
            tsx1 = self.get_aggregate(tsx1r, mindex=sx1r.index, aggMethod=aggMethod, logger=log)
            sx1 = sx1r.copy()
        elif base_index=='true':
            sx1 = self.get_aggregate(sx1r, mindex=tsx1r.index, aggMethod=aggMethod, logger=log)
            tsx1 = tsx1r.copy()
        else: raise IOError(base_index)
        
        mdex = sx1.index
        #=======================================================================
        # setup the figure
        #=======================================================================
        plt.close('all')
 
        col_keys =mdex.unique(plot_coln).tolist()
        row_keys = mdex.unique(plot_rown).tolist()
 
        fig, ax_d = self.get_matrix_fig(row_keys, col_keys,
                                    figsize_scaler=4,
                                    constrained_layout=True,
                                    sharey=sharey, 
                                    sharex=sharex,  
                                    fig_id=0,
                                    set_ax_title=True,
                                    )
 
        fig.suptitle(title)
        #=======================================================================
        # #get colors
        #=======================================================================
        if err_type=='confusion':
            ckeys = ['FP', 'FN', '1','2','3', 'TP','TN',] #adding some spacers
        else:
            ckeys = mdex.unique(plot_colr) 
        color_d = self.get_color_d(ckeys, colorMap=colorMap)
        
        #=======================================================================
        # loop and plot--------
        #=======================================================================
        meta_dx=None
        true_gb = tsx1.groupby(level=[plot_coln, plot_rown])
        for gkeys, gsx0 in sx1.groupby(level=[plot_coln, plot_rown]): #loop by axis data
            
            #===================================================================
            # setup
            #===================================================================
            keys_d = dict(zip([plot_coln, plot_rown], gkeys))
            ax = ax_d[gkeys[1]][gkeys[0]]
            log.info('on %s'%keys_d)
            tgsx0 = true_gb.get_group(gkeys) #raw, uncollaped
            mdex = gsx0.index
            assert np.array_equal(gsx0.index, tgsx0.index)
 
 
            meta_d = { 'modelIDs':str(list(mdex.unique(idn))),'drop_zeros':False,}
            meta_d.update(slice_d)
 
            
            #===================================================================
            # scatter-----
            #===================================================================
            """consider hist2d?"""
            if plot_type =='scatter':
 
                """only using mean values for now"""
 
                
                #===============================================================
                # #build colors
                #===============================================================
                cgserx0 = pd.Series(index=mdex, name='color', dtype=str)
                for gkey,gdx1 in gsx0.groupby(level=plot_colr):
                    cgserx0.loc[gdx1.index] = color_d[gkey]
 
                assert_index_equal(cgserx0.index,mdex)
                
                #===============================================================
                # #plot
                #===============================================================
                log.debug('    scatter on %i'%len(gsx0))
                stat_d = self.ax_corr_scat2(ax, gdx0.values.T[0], tgdx0.values.T[0], 
                                           colors_ar=cgserx0.values.T,
                                           #label='%s=%s'%(plot_colr, keys_d[plot_colr]),
                                           scatter_kwargs = {},
                                           logger=log, add_label=False, **kwargs)
 
                meta_d.update(stat_d)
                
            #===================================================================
            # scatter density----------
            #===================================================================
                
            elif plot_type=='scatter_density':
                raise Error('not implemeneted')
                #===============================================================
                # #plot
                #===============================================================
                log.debug('    scatter_density on %i'%len(gsx0))
                stat_d = self.ax_corr_scat2(ax, gdx0.values.T[0], tgdx0.values.T[0], 
                                            bins=bins,colorMap=colorMap,
                                           logger=log, add_label=False, **kwargs)
 
                meta_d.update(stat_d)
 
            #===================================================================
            # hist2d----
            #===================================================================
            elif plot_type=='hist2d':
                log.debug('    %s on %i'%(plot_type, len(gsx0)))
 
                stat_d = self.ax_corr_scat2(ax, gsx0.values, tgsx0.values, 
                                             bins=bins,colorMap=colorMap,
                                           logger=log, add_label=False,
                                           plot_type=plot_type, vmin=vmin, vmax=vmax,
                                           xlims=xlims,colorBar=False,
                                            **kwargs)
                
                vmin, vmax = stat_d.pop('vmin'), stat_d.pop('vmax')
                meta_d.update(stat_d)
                
 
                

 
            #===================================================================
            # bar plot---------
            #===================================================================
            elif plot_type=='bars':
                """TODO: consolidate w/ plot_total_bars
                integrate with write_suite_smry errors"""
 
                #===============================================================
                # data setup
                #===============================================================
 
                #loop on sub-group and calc errors on each group
                d=dict()
                true_gb2 = tgsx0.groupby(level=plot_bgrp)
                for gkey1, gsx1 in gsx0.groupby(level=plot_bgrp):
                    tgsx1 =  true_gb2.get_group(gkey1)
                    
                    with ErrorCalcs(logger=log,pred_ser=gsx1, true_ser=tgsx1, normed=normed) as eW:
                        d[gkey1] = eW.retrieve(err_type)
                        
 
                #standard bar parameters
                xlocs = np.linspace(0, 1, num=len(d)) #spread out from 0 to 1
                bar_cnt = len(d)
                width = 0.9 / float(bar_cnt)
                
                tick_label = ['%s=%s'%(plot_bgrp, i) for i in d.keys()]
                """
                fig.show()
                ax.clear()
                ax.legend()
                """
                #===============================================================
                # stacked confusion bars
                #===============================================================
                if err_type=='confusion':
                    meta_d.update({'normed':normed, 'base_index':base_index, 'aggMethod':aggMethod})
                    #extract frames
                    bhsx1 = pd.concat({k:v[1] for k,v in d.items()}, names=[plot_bgrp]).iloc[:,0]
                    
                    assert len(bhsx1)==bar_cnt*4, keys_d
 
                    
                    bottom = np.full(bar_cnt,0.0)
                    for gkey2, bhgsx in bhsx1.groupby(level=['codes']):
 
                        ylocs = bhgsx.values   #values per aggLevel 
                        
                        bars = ax.bar(xlocs, ylocs, width=width, 
                                      color=color_d[gkey2],bottom=bottom,
                                      tick_label=tick_label, label=gkey2
                                      )
                        bottom = bottom + ylocs
                                      
 
                #===============================================================
                # normal bars
                #===============================================================
                else:
                    #if err_type in ['TP', 'FP', 'FN', 'TN']:
                    #barHeight_ser = pd.Series(d, name=gkeys)
                    ylocs = np.array(d.values())
 
                    bars = ax.bar(
                        xlocs,  # xlocation of bars
                        ylocs,  # heights
                        width=width,
                        align='center',
                        color=color_d.values(),
                        #label='%s=%s' % (plot_colr, ckey),
                        #alpha=0.5,
                        tick_label=tick_label,
                        )
                    
                #===============================================================
                # post bar
                #===============================================================
                if err_type=='bias':
                    split_val = 1.0
                else:
                    split_val = 0.0
 
                ax.axhline(split_val, color='black', linestyle='dashed') #draw in the axis
 
                
                #===============================================================
                # add bar labels
                #===============================================================
                if bar_labels:
                    d1 = {k:pd.Series(v, dtype=float) for k,v in {'yloc':ylocs, 'xloc':xlocs}.items()}
                    
                    for event, row in pd.concat(d1, axis=1).iterrows():
                        """TODO: use a lambda here"""
                        txt = '%+.2f' %(row['yloc'])
     
                        #get the color
      
                        if row['yloc']>=split_val: color='black'
                        else: color='red'
                            
                        ax.text(row['xloc'], row['yloc'] * 1.01, #shifted locations
                                    txt,
                                    bbox=dict(boxstyle="round,pad=0.05", fc="white", lw=0.0,alpha=0.9 ), #light background fill,
                                    ha='center', va='bottom', rotation='vertical',fontsize=10, color=color)
                    
            #===================================================================
            # violin plot-----
            #===================================================================
            elif plot_type in ['violin', 'hist']:
                
                #===============================================================
                # data setup
                #===============================================================
                
                #calc error metric
                err_gdx0 = gdx0.mean(axis=1) - tgdx0.mean(axis=1) #collapse iters and get delta
                if err_type=='error':
                    err_gdx1 = err_gdx0
                elif err_type == 'errorRelative':
                    err_gdx1 = err_gdx0/err_gdx0.max()
                else:
                    raise IOError('bad err_type=%s for plot_type=violin'%err_type)
                
                #group it
                d = {k:v for k, v in err_gdx1.groupby(level=plot_bgrp)}
                labels = ['%s=%s'%(plot_bgrp, k) for k in d.keys()]
                #===============================================================
                # plot
                #===============================================================
                md1 = self.ax_data(ax, d, plot_type=plot_type, color_d=color_d, logger=log,
                                   zero_line=zero_line, **kwargs)
                meta_d.update(md1)
 
 
            else:
                raise KeyError('unrecognized plot_type: %s'%plot_type)
 
            #===================================================================
            # post-format----
            #===================================================================
            ax.set_title(' & '.join(['%s:%s' % (k, v) for k, v in keys_d.items()]))
            #===================================================================
            # meta text
            #===================================================================
            log.debug('    meta_func')
            """for bars, this ignores the bgrp key"""
            meta_d = meta_func(logger=log, meta_d=meta_d,pred_ser=gsx0,true_ser=tgsx0)
                            
            if meta_txt: 
                ax.text(0.1, 0.9, get_dict_str(meta_d), transform=ax.transAxes, va='top', fontsize=8, color='black',
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", lw=0.0,alpha=0.5 ), #light background fill
                        )
                """
                fig.show()
                """
            #===================================================================
            # post-meta--------
            #===================================================================
            meta_serx = pd.Series(meta_d, name=gkeys)
            if meta_dx is None:
                meta_dx = meta_serx.to_frame().T
                meta_dx.index.set_names(keys_d.keys(), inplace=True)
            else:
                meta_dx = meta_dx.append(meta_serx)
                
        #===============================================================
        # post format subplot-------
        #===============================================================
        """best to loop on the axis container in case a plot was missed"""
        for row_key, d in ax_d.items():
            for col_key, ax in d.items():
                if plot_type in ['scatter', 'hist2d']:
                    ax.legend(loc=1)
                    
                # first row
                if row_key == row_keys[0]:
                    if col_key == col_keys[-1]:
                        if plot_type =='bars':
                            ax.legend(loc=1)
                #last col
                if col_key == col_keys[-1]:
                    if plot_type in ['hist2d']: 
                        norm = Normalize(vmin =vmin, vmax = vmax)
                        cbar = fig.colorbar(matplotlib.cm.ScalarMappable(norm = norm), ax=ax)
                        cbar.ax.set_ylabel('Density')
  
                # first col
                if col_key == col_keys[0]:
                    if plot_type in ['bars', 'violin']:
                        ax.set_ylabel('\'%s\' (%s)'%(dkey, err_type))
                        
                    elif plot_type in ['scatter', 'hist2d']:
                        ax.set_ylabel('\'%s\' (true)'%dkey)
                
                #last row
                if row_key == row_keys[-1]:
                    if plot_type == 'bars': 
                        pass
                        #ax.set_ylabel('\'%s\' (agg - true)'%dkey)
                    elif plot_type=='violin':
                        ax.set_xticks(np.arange(1, len(labels) + 1))
                        ax.set_xticklabels(labels)
                        
                    else:
                        ax.set_xlabel('\'%s\' (aggregated)'%dkey)
                        
                    
 
 
        #=======================================================================
        # wrap---------
        #=======================================================================
        log.debug('wrap')
        """
        plt.show()
        """
        if plot_type=='bar':
            fname = 'errMat_%s_%s_%s_%sX%s_%s' % (
            title.replace(' ','').replace('\'',''),
             plot_type, err_type, plot_rown, plot_coln, self.longname)
        else:
            fname='errMat_%s_%s_%sX%s_%s' % (
            title.replace(' ','').replace('\'',''),
             plot_type, plot_rown, plot_coln, self.longname)
        
        fname = fname.replace('=', '-')
        if write_meta:
            ofp =  os.path.join(self.out_dir, fname+'_meta.csv')
            meta_dx.to_csv(ofp)
            log.info('wrote meta_dx %s to \n    %s'%(str(meta_dx.shape), ofp))
               
        
        return self.output_fig(fig, fname=fname, fmt=fmt)
 
    
    def plot_vs_mat(self, #plot dkeys against eachother in a scatter matrix
                  
                    #data
                    dkey_y='rloss',#column group w/ values to plot
                    dkey_x='rsamps',
                     
 
                    dx_raw=None, #combined model results
                    modelID_l = None, #optinal sorting list
                    
                    #plot config
                    plot_type='scatter',
                    plot_rown='studyArea',
                    plot_coln='vid',
                    plot_colr=None,
                    #plot_bgrp='modelID',
                    slice_d = {}, #special slicing
                    drop_zeros=False,
                    
                    #meta labelling
                    meta_txt=True, #add meta info to plot as text
                    meta_func = lambda meta_d={}, **kwargs:meta_d, #lambda for calculating additional meta information (add_meta=True)        
                    write_meta=False, #write all the meta info to a csv  
 
                    
                    #plot style
                    title=None,
                     sharey='all', sharex='all',
                    colorMap=None, 
                    xlims=None,ylims=None,fmt='png',
                    **kwargs):
        """"
        generally 1 modelId per panel
        """
        
        #=======================================================================
        # defaults
        #=======================================================================
        log = self.logger.getChild('plot_compare_mat')
        
        if plot_colr is None: plot_colr=plot_rown
        idn = self.idn
        #if baseID is None: baseID=self.baseID
 
        if dx_raw is None:
            dx_raw = self.retrieve('outs')
            
        if not xlims is None:
            assert sharex=='all'
        if not ylims is None:
            assert sharey=='all'
        
        if title is None:
            title = '\'%s\' vs \'%s\' '%(dkey_x, dkey_y)
        
        log.info('on \'%s\' vs \'%s\' (%s x %s)'%(dkey_x, dkey_y, plot_rown, plot_coln))
        #=======================================================================
        # data prep------
        #=======================================================================
        #=======================================================================
        # #add requested indexers
        #=======================================================================
        
        meta_indexers = set([plot_rown, plot_coln])
        if not plot_colr is None:
            meta_indexers.add(plot_colr)
        
        
        dx = self.join_meta_indexers(dx_raw =  dx_raw.loc[:, idx[[dkey_x, dkey_y], :]], 
                                meta_indexers = meta_indexers.copy(),
                                modelID_l=modelID_l)
        
        
        
        #=======================================================================
        # subsetting
        #=======================================================================
        for name, val in slice_d.items():
            assert name in dx.index.names
            bx = dx.index.get_level_values(name) == val
            assert bx.any(), 'failed to get any matches for %s=%s'%(name, val)
            dx = dx.loc[bx, :]

            log.info('w/ %s=%s slicing to %i/%i'%(
                name, val, bx.sum(), len(bx)))
            
        #=======================================================================
        # collpase iters
        #=======================================================================
        """just taking the first iteration"""
        dx1 = dx.groupby(level=0, axis=1).first()
        
        #=======================================================================
        # dropping zeros
        #=======================================================================
        if drop_zeros:
            raise Error('dome')
 
        
        log.info('on %s'%str(dx.shape))
 
        mdex = dx.index
        #=======================================================================
        # setup the figure---------
        #=======================================================================
        plt.close('all')
 
        col_keys =mdex.unique(plot_coln).tolist()
        row_keys = mdex.unique(plot_rown).tolist()
 
        fig, ax_d = self.get_matrix_fig(row_keys, col_keys,
                                    figsize_scaler=4,
                                    constrained_layout=True,
                                    sharey=sharey, sharex=sharex,  # everything should b euniform
                                    fig_id=0,
                                    set_ax_title=True,
                                    )
        
        fig.suptitle(title)
        
        #=======================================================================
        # #get colors
        #=======================================================================
        if colorMap is None: colorMap = self.colorMap_d[plot_colr]
 
        ckeys = mdex.unique(plot_colr) 
        color_d = self.get_color_d(ckeys, colorMap=colorMap)
        
        #=======================================================================
        # loop and plot------
        #=======================================================================
 
        for gkeys, gdx0 in dx1.groupby(level=[plot_coln, plot_rown]): #loop by axis data
            
            #===================================================================
            # setup
            #===================================================================
            keys_d = dict(zip([plot_coln, plot_rown], gkeys))
            ax = ax_d[gkeys[1]][gkeys[0]]
            log.info('on %s'%keys_d)
            
            meta_d = { 'modelIDs':str(list(gdx0.index.unique(idn))),
                            'drop_zeros':False,'count':len(gdx0),
                            'zero_cnt':(gdx0==0).sum().sum()
                            }
            #===================================================================
            # scatter------
            #===================================================================
            if plot_type =='scatter':
 
                #===============================================================
                # #build colors
                #===============================================================
                cgserx0 = pd.Series(index=gdx0.index, name='color', dtype=str)
                for gkey,gdx1 in gdx0.mean(axis=1).groupby(level=plot_colr):
                    cgserx0.loc[gdx1.index] = color_d[gkey]
 
                assert_index_equal(cgserx0.index, gdx0.index)
                
                #===============================================================
                # #plot
                #===============================================================
                log.debug('    scatter on %i'%len(gdx0))
                stat_d = self.ax_corr_scat2(ax, gdx0[dkey_x].values, gdx0[dkey_y].values, 
                                           colors_ar=cgserx0.values,
                                           #label='%s=%s'%(plot_colr, keys_d[plot_colr]),
                                           scatter_kwargs = {},
                                           logger=log, add_label=False, **kwargs)
 
                meta_d.update(stat_d)
                
            else:
                raise IOError(plot_type)
                
 
            #===================================================================
            # post-format----
            #===================================================================
            ax.set_title(' & '.join(['%s:%s' % (k, v) for k, v in keys_d.items()]))
            
            if not xlims is None:
                ax.set_xlim(xlims)
                
            if not ylims is None:
                ax.set_ylim(ylims)
            #===================================================================
            # labels
            #===================================================================
            
            meta_d = meta_func(logger=log, meta_d=meta_d,xser=gdx0[dkey_x], yser=gdx0[dkey_y])
                        
            if meta_txt: 
                ax.text(0.1, 0.9, get_dict_str(meta_d), transform=ax.transAxes, va='top', fontsize=8, color='black',
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", lw=0.0,alpha=0.5 ), #light background fill
                        )
                
        #===============================================================
        # #wrap format subplot
        #===============================================================
        """best to loop on the axis container in case a plot was missed"""
        for row_key, d in ax_d.items():
            for col_key, ax in d.items():
 
                ax.legend(loc=1)
                # first row
                if row_key == row_keys[0]:
                    pass
                #last col
                if col_key == col_keys[-1]:
                    pass
                    
                
                        
                # first col
                if col_key == col_keys[0]:
                    ax.set_ylabel('\'%s\''%dkey_y)
                
                #last row
                if row_key == row_keys[-1]:
                    ax.set_xlabel('\'%s\''%dkey_x)
 
 
        #=======================================================================
        # wrap
        #=======================================================================
        log.info('finsihed')
        """
        plt.show()
        """
        
        fname='valuesXY_%s_%s-%s_%s'%(title, plot_rown, plot_coln, self.longname)
        
        fname = fname.replace('\'', '').replace(' ','')
        
        return self.output_fig(fig, fname=fname,fmt=fmt, **kwargs)
    
    def plot_rast(self, #add raster values to a plot
                
                #data control               
                xlims = None,
                modelID_l = None, #optinal sorting list
                drop_zeros=False, #zeros are effecitvely null on depth rasters 
                slice_d = {}, #special slicing
                debug_max_len=1e6, #max random sample
                
                #data
                rast_fp_lib=None,
                agg_mindex=None, #needed for adding meta indexers
                
                #plot config
                plot_type='hist', 
                plot_rown='resolution',
                plot_coln='studyArea',
                plot_colr=None,
                plot_bgrp=None,                
                ax_d=None, #inheriting preconfigured matrix plot
                
                #histwargs
                bins=20, rwidth=0.9, 
                mean_line=True, #plot a vertical line on the mean
                
                
                #meta labelling
                meta_txt=True, #add meta info to plot as text
                meta_func = lambda meta_d={}, **kwargs:meta_d, #lambda for calculating additional meta information (add_meta=True)        
                write_meta=False, #write all the meta info to a csv            
                
                #plot style                    
                colorMap=None, title=None,
                sharey='col',sharex='col',
                val_lab = 'depths (m)',
                
                logger=None,  write=None,  **kwargs):
 
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger=self.logger
        log = logger.getChild('plot_rast')
        
        idn = self.idn
        debug_max_len = int(debug_max_len)

            
        if write is None:write=self.write
            
            
        #plot keys
        if plot_colr is None: 
            plot_colr=plot_bgrp
        
        if plot_colr is None: 
            plot_colr=plot_rown
            
        if plot_bgrp is None:
            plot_bgrp = plot_colr
 
        assert not plot_rown==plot_coln
        assert not plot_bgrp is None
        
        assert plot_colr in [plot_coln, plot_rown, plot_bgrp], 'must specify a valid plot_colr'
        
        """othwerise... wed have to load the ras
        assert 'studyArea' in [plot_coln, plot_rown], 'must index on studyArea'"""
            
        #plot style
                 
        if title is None:
            title = 'depth raster values'
                
            for name, val in slice_d.items(): 
                title = title + ' %s=%s'%(name, val) 
                
        if colorMap is None: colorMap = self.colorMap_d[plot_colr]
        
        #=======================================================================
        # retrieve
        #=======================================================================
        if rast_fp_lib is None:
            rast_fp_lib= self.retrieve('drlay_fps') 
            
        if agg_mindex is None:
            agg_mindex = self.retrieve('agg_mindex')
 
        log.info('on (%s x %s)'%(plot_rown, plot_coln))
        #=======================================================================
        # data prep
        #=======================================================================
        """a bit strange here... but originally we set these up to handle data"""
        meta_indexers = set([plot_rown, plot_coln, plot_bgrp])
        #=======================================================================
        # if not plot_bgrp is None:
        #     meta_indexers.add(plot_bgrp)
        #=======================================================================
        
        dx_raw = pd.DataFrame(index=agg_mindex.droplevel('id'))
        dx_raw['dummy']=0.0
        
        #add requested indexers
        dx = self.join_meta_indexers(dx_raw = dx_raw, 
                                meta_indexers = meta_indexers.copy(),
                                modelID_l=modelID_l)
 
        log.info('on %s'%str(dx.shape))
        #=======================================================================
        # subsetting
        #=======================================================================
        for name, val in slice_d.items():
            assert name in dx.index.names
            bx = dx.index.get_level_values(name) == val
            assert bx.any()
            dx = dx.loc[bx, :]

            log.info('w/ %s=%s slicing to %i/%i'%(
                name, val, bx.sum(), len(bx)))
        
        #=======================================================================
        # get iteration index 
        #=======================================================================
        """beacuse we dont want the data... just the groupings"""
        mdex = dx.index.droplevel('gid').to_frame().drop_duplicates().index
        #=======================================================================
        # setup the figure
        #=======================================================================
        if ax_d is None:
            plt.close('all')
     
            col_keys =mdex.unique(plot_coln).tolist()
            row_keys = mdex.unique(plot_rown).tolist()
     
            fig, ax_d = self.get_matrix_fig(row_keys, col_keys,
                                        figsize_scaler=4,
                                        constrained_layout=True,
                                        sharey=sharey, 
                                        sharex=sharex,  
                                        fig_id=0,
                                        set_ax_title=True,
                                        )
     
            fig.suptitle(title)
            

        
        #=======================================================================
        # attach filepahts
        #=======================================================================
        dx1 = mdex.to_frame()
 
        for gkeys, gdf in dx1.copy().groupby(level=['modelID', 'studyArea']):
            keys_d = dict(zip(['modelID', 'studyArea'], gkeys))
            
            dx1.loc[gdf.index, 'fp'] = rast_fp_lib[keys_d['modelID']][keys_d['studyArea']]


        #=======================================================================
        # attach colors
        #=======================================================================
 
        ckeys = mdex.unique(plot_colr) 
        color_d = self.get_color_d(ckeys, colorMap=colorMap)
        
        for gkey, gdf in dx1.copy().groupby(level=plot_colr):
            dx1.loc[gdf.index, 'color'] = color_d[gkey]
        #=======================================================================
        # loop and plot
        #=======================================================================
        meta_dx=None
        
 
        
        for gkeys, gdx0 in dx1.groupby(level=[plot_coln, plot_rown]): #loop by axis data
            #===================================================================
            # setup
            #===================================================================
            keys_d = dict(zip([plot_coln, plot_rown], gkeys))
            ax = ax_d[gkeys[1]][gkeys[0]]
            log.info('on %s'%keys_d)
            
 
            #===================================================================
            # data setup-------
            #===================================================================
            #gropu the filepathts
            sg_fp_d0 = {k:df['fp'].tolist() for k,df in gdx0.groupby(level=plot_bgrp)}
 
 
                
            #===================================================================
            # #remove duplicates
            #===================================================================
            
            """because we copy identical rasters for each model run.. 
            here we just remove duplicates using the filename
            dont care about which modelID it comes from?
            maybe there's a simpler way? this seems complicated
            """
            sg_fp_d1 = dict()
            cnt = 0
            for gkey1, fp_l in sg_fp_d0.items():
                #divide dir from fp
                df0 = pd.DataFrame.from_dict({i:os.path.split(fp) for i, fp in enumerate(fp_l)},
                                             columns=['dir', 'fn'], orient='index')
                #remove duplicates
                df1 = df0.drop_duplicates(subset=['fn'])
                
                #get these filepaths 
                sg_fp_d1[gkey1] = [os.path.join(row['dir'], row['fn']) for i, row in df1.iterrows()]
                cnt+= len(sg_fp_d1[gkey1])
                    
 
            log.info('on %s w/ %i groups and %i rasters'%(keys_d, len(sg_fp_d1), cnt))
            

            meta_d = {'raster_cnt':cnt, 'drop_zeros':drop_zeros, 'modelIDs':str(list(gdx0.index.unique(idn)))}
            #===================================================================
            # loop on raster and collect data
            #===================================================================
 
            zcnt=0
            data_d = dict()
 
            for gkey1, fp_l in sg_fp_d1.items():
                rser = None
                for fp in fp_l:
                    """just collapse everything into 1 pile"""
                    log.debug(fp)
                    
                    #get values
                    ar_raw = rlay_to_array(fp)
                    
                    ser1 = pd.Series(ar_raw.reshape((1,-1))[0]).dropna()
                    
                    #remove zeros
                    bx = ser1==0.0
                    if drop_zeros:
                        
                        ser1 = ser1.loc[~bx]
                    
                    #counts
                    zcnt+=bx.sum()
 
                    
                    #collect
                    if rser is None:
                        rser = ser1
                    else:
                        """not checked"""
                        rser = ser1.append(ser1)
                        
 
                data_d[gkey1] = rser
                log.info('   %s w/ %i cells '%(gkey1, len(rser)))
            
            #post
            rserx = pd.concat(data_d)
            
            #reduce
            if not debug_max_len is None:
                if len(rserx)>debug_max_len:
                    log.warning('reducing from %i to %i'%(len(rserx), debug_max_len))
                    meta_d['raw_cnt'] = len(rserx)
                    rserx = rserx.sample(debug_max_len) #get a random sample of these
                    #rserx = rserx.iloc[0:debug_max_len]
                    
            #split into a dict
            data_d = {k:g.values for k,g in rserx.groupby(level=0)}
            
            
            meta_d.update({'cnt':len(rserx), 'zeros_cnt':zcnt})
            
 
            #===================================================================
            # add plots--------
            #===================================================================
            
            color_d1 =  dict(zip(data_d.keys(), gdx0['color'].unique()))
            
            if mean_line:
                mval =rserx.mean().mean()
            else: mval=None 
            
            md1 = self.ax_data(ax, data_d,
                               plot_type=plot_type, 
                               bins=bins, rwidth=rwidth, mean_line=mval, hrange=xlims,
                               color_d = color_d1, logger=log, **kwargs) 
 
            meta_d.update(md1)
            
            #===================================================================
            # post format------
            #===================================================================
            
            ax.set_title(' & '.join(['%s:%s' % (k, v) for (k, v) in keys_d.items()]))
            #===================================================================
            # meta  text
            #===================================================================
            """for bars, this ignores the bgrp key"""
            meta_d = meta_func(logger=log, meta_d=meta_d, pred_ser=rserx)
            if meta_txt:
                ax.text(0.9, 0.6, get_dict_str(meta_d), transform=ax.transAxes, va='top', ha='right', fontsize=8, color='black')
            
            
            #===================================================================
            # collect meta------
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
                
                if not xlims is None:
                    ax.set_xlim(xlims)
 
                
                # first row
                if row_key == row_keys[0]:
                    pass
                    #last col
                    #if col_key == col_keys[-1]:
                        #if plot_type=='hist':
                    #ax.legend()
                
                        
                # first col
                if col_key == col_keys[0]:
                    if plot_type == 'hist':
                        ax.set_ylabel('count')
                    elif plot_type=='gaussian_kde':
                        ax.set_ylabel('frequency')
                    elif plot_type in ['box', 'violin']:
                        ax.set_ylabel(val_lab)
                
                #last row
                if row_key == row_keys[-1]:
                    if plot_type in ['hist', 'gaussian_kde']:
                        ax.set_xlabel(val_lab)
                    elif plot_type in ['violin', 'box']:
                        
                        ax.set_xticks(np.arange(1, len(labels) + 1))
                        ax.set_xticklabels(labels)
                #last col
                if col_key == col_keys[-1]:
                    if write: ax.legend()
                
                
        #=======================================================================
        # wrap---------
        #=======================================================================
        log.info('finsihed')
        """
        plt.show()
        """
 
        fname = 'rastVals_%s_%s_%sX%s_%s' % (
            title.replace(' ','').replace('\'',''),
             plot_type, plot_rown, plot_coln, self.longname)        
        fname = fname.replace('=', '-')
        
        if write_meta:
            ofp =  os.path.join(self.out_dir, fname+'_meta.csv')
            meta_dx.to_csv(ofp)
            log.info('wrote meta_dx %s to \n    %s'%(str(meta_dx.shape), ofp))
               
               
        if write:
            ofp = self.output_fig(fig, fname=fname)
        
        return ax_d 
         
    def plot_perf_mat(self, #plot performance
                  
                    #data control
                    dkey_d = {#{dkey:groupby operation method to use for aggregating the true values}
                        'rsamps':{'trueAgg':'mean', 'err_type':'meanError'},
                        'rloss':{'trueAgg':'mean', 'err_type':'meanError'},
                        'tloss':{'trueAgg':'sum', 'err_type':'bias_shift'},
                        }, 
                    baseID=0,
                    modelID_l = None,
 
 
 
                    #drop_zeros=True, #must always be false for the matching to work
                    slice_d = {}, #special slicing
                    
                    #data
                    true_dx_raw=None, #base values (indexed to raws per model)
                    dx_raw=None, #combined model results                    
                    
                    #plot config
                    #plot_type='bars', always grouped bars
                    plot_rown='studyArea',
                    plot_coln='resolution',
                    plot_colr='dkey',
                    plot_bgrp='aggLevel', #clustering bars
 
 
                    #meta labelling
                    meta_txt=True, #add meta info to plot as text
                    write_meta=False, #write all the meta info to a csv            
                    
                    #plot style      
                    barLabel_func = lambda **kwargs:'{yloc:+.2f}'.format(**kwargs), #formatter func for bart lables
                    colorMap=None, title=None, baseWidth=0.7,
                    sharey='all', 
                    
                    #plot output
                    fmt='svg',
 
                    **kwargs):
        """"
        generally 1 modelId per panel
 
        
        
        """
 
        #=======================================================================
        # defaults----------
        #=======================================================================
        log = self.logger.getChild('plot_perf_mat')
 
        idn = self.idn
 
        #=======================================================================
        # #retrieve data
        #=======================================================================
        if dx_raw is None:
            dx_raw = self.retrieve('outs')
            
        if true_dx_raw is None:
            true_d = self.retrieve('trues')
            true_dx_raw = true_d[baseID]

            
        
        #=======================================================================
        # #logic checks
        #=======================================================================
        assert plot_colr=='dkey', 'only dkey supported for now'
        assert not plot_rown==plot_coln
        assert not plot_rown==plot_bgrp
 
            
        #=======================================================================
        # defaults. plot style
        #=======================================================================
        sharex='all' #alwasy for bars
                
       
        if title is None:
            title = '%i metric performance'%(len(dkey_d))
                
            for name, val in slice_d.items(): 
                title = title + ' %s=%s'%(name, val) 
                
        if colorMap is None:
            colorMap = self.colorMap_d[plot_colr]
                
 
        log.info('on \'%s\' (%s x %s)'%(dkey_d.keys(), plot_rown, plot_coln))
        #=======================================================================
        # data prep--------
        #=======================================================================
        assert_func(lambda: self.check_mindex_match(true_dx_raw.index, dx_raw.index), msg='raw vs trues')
        
        meta_indexers = set([plot_rown, plot_coln,plot_bgrp])
 
        
        for k in slice_d.keys(): meta_indexers.add(k) #add any required slicers
        
        #add requested indexers
        dx = self.join_meta_indexers(dx_raw = dx_raw.loc[:, idx[dkey_d.keys(), :]], 
                                meta_indexers = meta_indexers, modelID_l=modelID_l)
        
        
        
        
        #and on the trues
        true_dx = self.join_meta_indexers(dx_raw = true_dx_raw.loc[:, idx[dkey_d.keys(), :]], 
                                meta_indexers = meta_indexers, modelID_l=modelID_l)
 
        #=======================================================================
        # subsetting
        #=======================================================================
        for name, val in slice_d.items():
            assert name in dx.index.names, 'slice dimension (%s) not present'%name
            bx = dx.index.get_level_values(name) == val
            assert bx.any(), 'failed to match any %s=%s'%(name, val)
            dx = dx.loc[bx, :]
            
            bx = true_dx.index.get_level_values(name) == val
            assert bx.any()
            true_dx = true_dx.loc[bx, :]
            log.info('w/ %s=%s slicing to %i/%i'%(
                name, val, bx.sum(), len(bx)))
            
        
        #=======================================================================
        # collpase iters
        #=======================================================================
        """just taking the first iteration"""
        dx1 = dx.groupby(level=0, axis=1).first()
        true_dx1 = true_dx.groupby(level=0, axis=1).first()
        
        mdex = dx1.index
        
        log.info('on %s'%str(dx1.shape))
        #=======================================================================
        # setup the figure
        #=======================================================================
        plt.close('all')
 
        col_keys =mdex.unique(plot_coln).tolist()
        row_keys = mdex.unique(plot_rown).tolist()
 
        fig, ax_d = self.get_matrix_fig(row_keys, col_keys,sharey=sharey,sharex=sharex, 
                                    figsize_scaler=4,
                                    constrained_layout=True,fig_id=0,set_ax_title=True,)
 
        fig.suptitle(title)
        #=======================================================================
        # #get colors
        #=======================================================================
        if plot_colr=='dkey':
            ckeys = dkey_d.keys()
        else:
            ckeys = mdex.unique(plot_colr) 
        color_d = self.get_color_d(ckeys, colorMap=colorMap)
        
        #=======================================================================
        # plot loop-----------
        #=======================================================================
        meta_dx=None
        true_gb = true_dx1.groupby(level=[plot_coln, plot_rown])
        for gkeys, gdx0 in dx1.groupby(level=[plot_coln, plot_rown]): #loop by axis data
            
            #===================================================================
            # setup
            #===================================================================
            keys_d = dict(zip([plot_coln, plot_rown], gkeys))
            ax = ax_d[gkeys[1]][gkeys[0]]
            log.info('on %s'%keys_d)
            tgdx_raw = true_gb.get_group(gkeys)
            meta_d = { 'model ID':str(list(gdx0.index.unique(idn))),'drop_zeros':False, 'cnt':len(gdx0)}
            #===================================================================
            # collect error metrics-------
            #===================================================================
            elib=dict()
            for dkey, pars_d in dkey_d.items():
                gserx0 = gdx0[dkey]
 
                #===================================================================
                # aggregate the trues
                #===================================================================
                """because trues are mapped from the base model.. here we compress down to the model index
                using a different aggregation method for each gkey"""
                tgserx0 = getattr(tgdx_raw[dkey].groupby(level=[gserx0.index.names]), pars_d['trueAgg'])()
                tgserx0 = tgserx0.reorder_levels(tgserx0.index.names).sort_index()
                
                #check
                assert np.array_equal(gserx0.index, tgserx0.index)
 
                #===============================================================
                # loop on groups
                #===============================================================
                true_gb1 = tgserx0.groupby(level=plot_bgrp)
                d = dict()
                for gkey1, gserx1 in gserx0.groupby(level=plot_bgrp):
                    tgserx1 = true_gb1.get_group(gkey1)
                    #===============================================================
                    # calc metric
                    #===============================================================
                    d[gkey1] = ErrorCalcs(logger=log,pred_ser=gserx1,true_ser=tgserx1).retrieve(pars_d['err_type'])
                    
                elib[dkey] = pd.Series(d, name=dkey)
                
            #multindex series with all of the calculated error values
            bh_serx = pd.concat(elib, names=['dkey', plot_bgrp]).swaplevel().sort_index().rename('errs')
 
            bh_gb = bh_serx.groupby(level='dkey')
            #===================================================================
            # plot each dkey------
            #===================================================================
            """
            fig.show()
            ax.clear()
            """
            bar_cnt_group = len(bh_serx.index.unique(plot_bgrp)) #groups
            """lookping on dkey_d to preserve order"""
            for i, (dkey, pars_d) in enumerate(dkey_d.items()):
 
                gserx2 = bh_gb.get_group(dkey)
                
                #===============================================================
                # # calc bar props
                #===============================================================
                bar_cnt_dkey = len(gserx2)
                width = baseWidth / float(bar_cnt_dkey + bar_cnt_group)    
                
                xlocs = np.linspace(0, 1, num=bar_cnt_dkey)  + width*i #spread out from 0 to 1... shifted by iter       

                #===============================================================
                # # #add bars                
                #===============================================================
                bars = ax.bar(
                    xlocs,  # xlocation of bars
                    gserx2.values,  # heights
                    width=width,
                    align='center',
                    color=color_d[dkey],
                    label='%s (%s, %s)'%(dkey, dkey_d[dkey]['trueAgg'], dkey_d[dkey]['err_type'])
                    #alpha=0.5,
                    #tick_label=tick_label,
                    )
 
                #===============================================================
                # add bar labels
                #===============================================================
                d1 = {k:pd.Series(v, dtype=float) for k,v in {'yloc':gserx2.values, 'xloc':xlocs}.items()}
                
                for event, row in pd.concat(d1, axis=1).iterrows(): 
                    """looks like 3.5.0 has some native support"""
                    txt = barLabel_func(event=event, yloc=row['yloc'], **pars_d)
                    
                    #color
                    if row['yloc']>=0: color='black'
                    else: color='red'
                    
                    ax.text(row['xloc'], row['yloc'] * 1.01, #shifted locations
                                txt,
                                bbox=dict(boxstyle="round,pad=0.05", fc="white", lw=0.0,alpha=0.9 ), #light background fill
                                ha='center', va='bottom', rotation='vertical',fontsize=8, 
                                #color=color_d[dkey],
                                color=color, #color by value
                                )
                    
 
            #===============================================================
            # #formatters.
            #===============================================================
            """not sure how this will work when bar_cnt_dkey !=3"""
            tic_xlocs =np.linspace(0, 1, num=bar_cnt_group) + (bar_cnt_dkey/3)*width
 
            tick_labels = ['%s=%s'%(plot_bgrp, k) for k in bh_serx.index.unique(plot_bgrp)]
 

            ax.set_xticks(tic_xlocs, minor=False)
            
            """behavior depends on sharex"""
            ax.set_xticklabels(tick_labels)
            #===================================================================
            # post-format----
            #===================================================================
            ax.axhline(0, color='black', linewidth=0.5) #draw in the axis
            ax.set_title(' & '.join(['%s:%s' % (k, v) for k, v in keys_d.items()]))
            #===================================================================
            # meta text
            #===================================================================
            log.debug('    meta_func')
            """no great way to calc meta here
            meta_d = meta_func(logger=log, meta_d=meta_d, 
                                pred_ser=pd.Series(data_d['mean']), 
                                true_ser=pd.Series(true_data_d['mean']))"""
                            
            if meta_txt: 
                ax.text(0.1, 0.9, get_dict_str(meta_d), transform=ax.transAxes, va='top', fontsize=8, color='black',
                        #bbox=dict(boxstyle="round,pad=0.3", fc="white", lw=0.0,alpha=0.7 ), #light background fill
                        )
                """
                fig.show()
                """
            #===================================================================
            # post-meta--------
            #===================================================================
            meta_serx = pd.Series(meta_d, name=gkeys)
            if meta_dx is None:
                meta_dx = meta_serx.to_frame().T
                meta_dx.index.set_names(keys_d.keys(), inplace=True)
            else:
                meta_dx = meta_dx.append(meta_serx)
                
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
 
                    ax.set_ylabel('error')
 
                
                #last row
                if row_key == row_keys[-1]:
                    pass
                        
                    
 
 
        #=======================================================================
        # wrap---------
        #=======================================================================
        log.debug('wrap')
        """
        plt.show()
        """
 
        fname='perfMat_%s_%i_%sX%s_%s' % (
            title.replace(' ','').replace('\'',''),
             len(dkey_d), plot_rown, plot_coln, self.longname)
        
        fname = fname.replace('=', '-')
        if write_meta:
            ofp =  os.path.join(self.out_dir, fname+'_meta.csv')
            meta_dx.to_csv(ofp)
            log.info('wrote meta_dx %s to \n    %s'%(str(meta_dx.shape), ofp))
               
        
        return self.output_fig(fig, fname=fname, fmt=fmt)
 
    #===========================================================================
    # AX.PLOTTERS---------
    #===========================================================================
 

    def ax_corr_scat2(self,  # correlation scatter plots on an axis
                ax,
                xar, yar,
                
                #data control
                xlims=None,
                plot_type='scatter',
                
                colors_ar=None,

                 #density plotting
                 bins=10,
                 sort=True,
                 colorMap=None,
                 vmin=None, vmax=None,
                
                # plot control
                plot_trend=True,
                plot_11=True,
                
               
                
                # lienstyles
                scatter_kwargs={  # default styles
                    
                    } ,
 
                #labelling
                label=None,                
                add_label=True,
                
                #style
                colorBar=True,
                
                logger=None,):
        """support for plt.scatter"""
        #=======================================================================
        # defaultst
        #=======================================================================
        if logger is None: logger = self.logger
        log = logger.getChild('ax_corr_scat2')
 
        # assert isinstance(stat_keys, list), label
        if not xar.shape == yar.shape:
            raise Error('data mismatch on %s'%(label))
        assert isinstance(scatter_kwargs, dict), label
        #assert isinstance(label, str)
        # log.info('on %s'%data.shape)
        
        if plot_type=='scatter':
            colorBar=False
        #=======================================================================
        # setup 
        #=======================================================================
        max_v = max(max(xar), max(yar))
        
        if xlims is None:
            xlims = (min(xar), max(xar))
 
        """only cropping xvals... may still get some yvals exceeding this"""
        bx =np.logical_and(xar>xlims[0], xar<=xlims[1])

 
        stat_d = dict()    
        z=None
        #=======================================================================
        # normal scatter--------
        #=======================================================================
        
        if plot_type=='scatter':
            assert colorMap is None
            #overwrite defaults with passed kwargs
            scatter_kwargs = {**{'s':3.0, 
                                 'marker':'o', 
                                 #'fillstyle':'full'
                                 },
                              **scatter_kwargs}

            
            """density color?
            plt.show()
            """
            log.debug('scatter')
            #ax.plot(xar, yar, linestyle='None', label=label, **scatter_kwargs)
            ax.scatter(xar, yar, 
                       c=colors_ar, **scatter_kwargs
                       )
        
        #=======================================================================
        # density scatter---------
        #=======================================================================
        elif plot_type=='scatter_density':
            """I dont really understand whats gonig on here... also doesnt look sio noce"""
            assert colors_ar is None
        
            scatter_kwargs = {**{'s':7.0, 
                                 'marker':'h', 
                                 'alpha':0.8,
                                 #'fillstyle':'full'
                                 },
                              **scatter_kwargs}
            """
            Scatter plot colored by 2d histogram
            fig.show()
            """
            if ax is None :
                fig , ax = plt.subplots()
            else:
                fig= ax.figure
                
            #build the pdist histogram
            data , x_e, y_e = np.histogram2d( xar, yar, bins = bins, density = True )
            
            #interpolate between these on the point locations?
            z = interpn( ( 0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ) , data , np.vstack([xar,yar]).T , method = "splinef2d", bounds_error = False)
        
            #fill in the nulls
            z[np.where(np.isnan(z))] = 0.0
        
            # Sort the points by density, so that the densest points are plotted last
            if sort :
                idx = z.argsort()
                x, y, z = xar[idx], yar[idx], z[idx]
        
            ax.scatter( x, y, c=z, cmap=colorMap, 
                        vmin=vmin, vmax=vmax, #boudn the color normalization
                        **scatter_kwargs )
        
 
        #=======================================================================
        # hist2d-------
        #=======================================================================
        elif plot_type=='hist2d':
            z, x_e, y_e, img = ax.hist2d(xar[bx], yar[bx], bins=bins, density=True,
                             cmap=colorMap, vmin=vmin, vmax=vmax, cmin=1e-6, alpha=0.8)
 
 
        #=======================================================================
        # add the 1:1 line
        #=======================================================================
        
        if plot_11:
            log.debug('plot_11')
            # draw a 1:1 line
            ax.plot([0, max_v * 10], [0, max_v * 10], color='black', linewidth=0.75, label='1:1')
        
        #=======================================================================
        # add the trend line
        #=======================================================================
        if plot_trend:
            log.debug('plot_trend (linregress) on %i'%len(xar))
            slope, intercept, rvalue, pvalue, stderr = scipy.stats.linregress(xar, yar)
            
            log.debug('plot_trend (pearsonr)')
            pearson, pval = scipy.stats.pearsonr(xar, yar)
            
            
            x_vals = np.array(xlims)
            y_vals = intercept + slope * x_vals
            log.debug('plot_trend (plot)')
            ax.plot(x_vals, y_vals, color='red', linewidth=0.75, label='r=%.3f'%rvalue)
 
        #=======================================================================
        # get stats
        #=======================================================================
        if not z is None: #collect color min/maxes
            if vmin is None:
                vmin = np.min(z)
            if vmax is None:
                vmax = np.max(z)
                
            stat_d.update({'vmin':vmin, 'vmax':vmax})
            
        
        stat_d.update({
                'count':len(xar),
                   'LR.slope':round(slope, 3),
                  # 'LR.intercept':round(intercept, 3),
                  # 'LR.pvalue':round(slope,3),
                  #'pearson':round(pearson, 3), #just teh same as rvalue
                  'r value':round(rvalue, 3),
                   # 'max':round(max_v,3),
                   })
            
        # dump into a string
        
        if add_label:
            annot = label + '\n' + get_dict_str(stat_d)
            
            anno_obj = ax.text(0.1, 0.9, annot, transform=ax.transAxes, va='center')
 
        #=======================================================================
        # post format
        #=======================================================================
        log.debug('wrap')
        
        #square it
        ax.set_xlim(xlims)
        ax.set_ylim(xlims)
        ax.grid()
        
        #build a normalized color bar
        if colorBar:
            fig = ax.figure
            norm = Normalize(vmin =vmin, vmax = vmax)
            cbar = fig.colorbar(matplotlib.cm.ScalarMappable(norm = norm), ax=ax)
            cbar.ax.set_ylabel('Density')
        
        return stat_d

    def ax_corr_scat(self,  # correlation scatter plots on an axis
                ax,
                xar, yar,
                label=None,
                
                # plot control
                plot_trend=True,
                plot_11=True,
                
                # lienstyles
                scatter_kwargs={  # default styles
                    
                    } ,
 
                logger=None,
                add_label=True,
                ):
        
        #=======================================================================
        # defaultst
        #=======================================================================
        if logger is None: logger = self.logger
        log = logger.getChild('ax_corr_scat')
 
        # assert isinstance(stat_keys, list), label
        if not xar.shape == yar.shape:
            raise Error('data mismatch on %s'%(label))
        assert isinstance(scatter_kwargs, dict), label
        #assert isinstance(label, str)
        # log.info('on %s'%data.shape)
        
        #=======================================================================
        # setup 
        #=======================================================================
        max_v = max(max(xar), max(yar))
        xlim = (min(xar), max(xar))
        #=======================================================================
        # add the scatter
        #=======================================================================
        #overwrite defaults with passed kwargs
        scatter_kwargs = {**{'markersize':3.0, 'marker':'.', 'fillstyle':'full'},
                          **scatter_kwargs}
        
        """density color?"""
        ax.plot(xar, yar, linestyle='None', label=label, **scatter_kwargs)
 
        #=======================================================================
        # add the 1:1 line
        #=======================================================================
        if plot_11:
            # draw a 1:1 line
            ax.plot([0, max_v * 10], [0, max_v * 10], color='black', linewidth=0.5, label='1:1')
        
        #=======================================================================
        # add the trend line
        #=======================================================================
        if plot_trend:
            slope, intercept, rvalue, pvalue, stderr = scipy.stats.linregress(xar, yar)
            
            pearson, pval = scipy.stats.pearsonr(xar, yar)
            
            
            x_vals = np.array(xlim)
            y_vals = intercept + slope * x_vals
            
            ax.plot(x_vals, y_vals, color='red', linewidth=0.5, label='r=%.3f'%rvalue)
 
        #=======================================================================
        # get stats
        #=======================================================================
        
        stat_d = {
                'count':len(xar),
                   'LR.slope':round(slope, 3),
                  # 'LR.intercept':round(intercept, 3),
                  # 'LR.pvalue':round(slope,3),
                  #'pearson':round(pearson, 3), #just teh same as rvalue
                  'r value':round(rvalue, 3),
                   # 'max':round(max_v,3),
                   }
            
        # dump into a string
        
        if add_label:
            annot = label + '\n' + get_dict_str(stat_d)
            
            anno_obj = ax.text(0.1, 0.9, annot, transform=ax.transAxes, va='center')
 
        #=======================================================================
        # post format
        #=======================================================================
        ax.set_xlim(xlim)
        ax.set_ylim(xlim)
        ax.grid()
        
        return stat_d
    
    
    #===========================================================================
    # HELPERS--------
    #===========================================================================
  

    def prep_ranges(self, #for multi-simulations, compress each entry using the passed stats 
                    qhi, qlo, drop_zeros, gdx_raw, logger=None):
        if logger is None: logger=self.logger
        log=logger.getChild('prep_ranges')
        #=======================================================================
        # check
        #=======================================================================
        
        assert not gdx_raw.isna().all(axis=1).any(), 'got some assets with all nulls'
        gdx0 = gdx_raw

        
        #check
        assert gdx0.notna().all().all()
        
        
        #===================================================================
        # prep data
        #===================================================================
        #handle zeros
        bx = (gdx0 == 0).all(axis=1)
        if drop_zeros:
            gdx1 = gdx0.loc[~bx, :]
        else:
            gdx1 = gdx0
            
        #collect ranges
        data_d = {'mean':gdx1.mean(axis=1).values}
        #add range for multi-dimensional
        if len(gdx1.columns) > 1:
            data_d.update({
                    'hi':gdx1.quantile(q=qhi, axis=1).values, 
                    'low':gdx1.quantile(q=qlo, axis=1).values})
        return data_d,bx
    

    
    
    def join_meta_indexers(self, #join some meta keys to the output data
                modelID_l = None, #optional sub-set of modelIDs
                meta_indexers = {'aggLevel', 'dscale_meth'}, #metadat fields to promote to index
                dx_raw=None,
                ):
        
        #=======================================================================
        # defaults
        #=======================================================================
        idn = self.idn
        log = self.logger.getChild('prep_dx')
        if dx_raw is None: 
            dx_raw = self.retrieve('outs')
                
        assert isinstance(meta_indexers, set)
        
        if modelID_l is None:
            modelID_l=dx_raw.index.unique(idn).tolist()
            
        #=======================================================================
        # checks
        #=======================================================================
        #duplicated modelID requests
        s = pd.Series(modelID_l, dtype=str)
        bx = s.duplicated()
        if bx.any():
            raise Error('got %i/%i duplicated modelIDs\n    %s'%(bx.sum(), len(bx), s[bx].tolist()))
 
        
        overlap_l = set(dx_raw.index.names).intersection(meta_indexers)
        if len(overlap_l)>0:
            log.warning('%i requested fields already in the index: %s'%(len(overlap_l), overlap_l))
            for e in overlap_l:
                meta_indexers.remove(e)
        
        #slice to these models
        """todo: allow this to handle series"""
        dx = dx_raw.loc[idx[modelID_l, :,:,:],:].dropna(how='all', axis=1).sort_index(axis=0)
        
        if len(meta_indexers) == 0:
            log.warning('no additional field srequested')
            return dx
        
        cat_df = self.retrieve('catalog')
        
        """
        view(dx.loc[idx[31, :,:,:],:].head(100))
        
        """
        null_cnt_ser = dx.isna().sum(axis=1)
        chk_index = dx.index.copy()
        #=======================================================================
        # check
        #=======================================================================
        miss_l = set(meta_indexers).difference(cat_df.columns)
        assert len(miss_l)==0, 'missing %i requested indexers: %s'%(len(miss_l), miss_l)
        
        miss_l = set(dx_raw.index.unique(idn)).difference(cat_df.index)
        assert len(miss_l)==0
        
        miss_l = set(modelID_l).difference(dx_raw.index.unique(idn))
        assert len(miss_l)==0, '%i/%i requested models not foundin data\n    %s'%(len(miss_l), len(modelID_l), miss_l)
        
 
        #=======================================================================
        # join new indexers
        #=======================================================================
        #slice to selection
        
        cdf1 = cat_df.loc[modelID_l,meta_indexers]
        
        #create expanded mindex from lookups
        assert cdf1.index.name == dx.index.names[0]
        dx.index = dx.index.join(pd.MultiIndex.from_frame(cdf1.reset_index()))
        
        
        
        
        #reorder a bit
        dx = dx.reorder_levels(list(dx_raw.index.names) + list(meta_indexers))
        
        assert_index_equal(dx.index.droplevel(list(meta_indexers)), chk_index)
        
        dx = dx.swaplevel(i=-1).sort_index()
        #=======================================================================
        # check
        #=======================================================================
        miss_l = set(meta_indexers).difference(dx.index.names)
        assert len(miss_l)==0
        
        assert np.array_equal(null_cnt_ser.values, dx.isna().sum(axis=1).values)
 
        
        return dx
    

            
            
            
        
