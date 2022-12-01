'''
Created on Jan. 18, 2022

@author: cefect



'''
#===============================================================================
# imports--------
#===============================================================================
import os, datetime, math, pickle, copy, random, pprint, shutil
import pandas as pd
import numpy as np

from pandas.testing import assert_index_equal, assert_series_equal

idx = pd.IndexSlice

from hp.basic import set_info

from hp.Q import Qproj, QgsCoordinateReferenceSystem, QgsMapLayerStore, view, \
    vlay_get_fdata, vlay_get_fdf, Error, vlay_dtypes, QgsFeatureRequest, vlay_get_geo, \
    QgsWkbTypes, QgsRasterLayer, RasterCalc, QgsVectorLayer, assert_rlay_equal

import hp.gdal
from hp.exceptions import assert_func
from hp.hyd import HQproj

from agg.coms.scripts import QSession, BaseSession
from agg.coms.scripts import Catalog


def serx_smry(serx):
 
    d = dict()
    
    for stat in ['count', 'min', 'mean', 'max', 'sum']:
        f = getattr(serx, stat)
        d['%s_%s'%(serx.name, stat)] = f() 
    return d
 
def get_all_pars(): #generate a matrix of all possible parameter combinations
    pars_lib = copy.deepcopy(Model.pars_lib)
    
    #separate values
    par_vals_d, par_dkey_d = dict(), dict()
    for varnm, d in pars_lib.items():
        par_vals_d[varnm] = d['vals']
        par_dkey_d[varnm] = d['dkey']
        
    
    df = pd.MultiIndex.from_product(par_vals_d.values(), names=par_vals_d.keys()).to_frame().reset_index(drop=True)
    
    lkp_ser = pd.Series({k:par_dkey_d[k] for k in df.columns}, name='dkey')
    lkp_ser.index.name = 'varnm'
    
    """assuming order matches"""
    df.columns = pd.MultiIndex.from_frame(lkp_ser.to_frame().reset_index())
 
    return df.swaplevel(axis=1).sort_index(axis=1)
    """
    view(df)
    """
class HydSession(BaseSession): #mostly shares between hyd.scripts and hyd.analy
    gcn = 'gid'
    scale_cn = 'tvals'
    idn = 'modelID'
    
    agg_inv_d = {'mean':'join', 'sum':'divide'} #reverse aggregation actions
    
    def __init__(self, 
                 data_retrieve_hndls={},
                 **kwargs):
        
        data_retrieve_hndls = {**data_retrieve_hndls, **{
            'model_pars':{
                'build':lambda **kwargs:self.load_modelPars(**kwargs)
                },
            }}
                
                
        
        super().__init__( data_retrieve_hndls=data_retrieve_hndls,
                         **kwargs)
        
        # checking container
        self.mindex_dtypes = {
                 'studyArea':np.dtype('object'),
                 'id':np.dtype('int64'),
                 self.gcn:np.dtype('int64'),  # both ids are valid
                 'grid_size':np.dtype('int64'),
                 'event':np.dtype('O'),
                 self.scale_cn:np.dtype('int64'),
                 self.idn:np.dtype('int64'),
                 'tag':np.dtype('object'),
                 'aggLevel':np.dtype('int64')
                         }
        

    def load_modelPars(self,
                       dkey='model_pars',
                       logger=None,
                       model_pars_fp=None,
                       ):
    
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger=self.logger
        log=logger.getChild('load_modelPars')
        
        if model_pars_fp is None:
            from definitions import model_pars_fp
            
        idn = self.idn
        #===========================================================================
        # load pars file
        #===========================================================================
        from numpy import dtype #needed.  not sure why though
        
        #pars_df_raw.dtypes.to_dict()
        #pars_df_raw = pd.read_csv(pars_fp, index_col=False, comment='#')
        pars_df_raw= pd.read_excel(model_pars_fp, comment='#')
        
        
        #remove notes comumns
        bxcol = pars_df_raw.columns.str.startswith('~')
        if bxcol.any():
            print('dropping %i/%i columns flagged as notes'%(bxcol.sum(), len(bxcol)))
            
            
        pars_df1 = pars_df_raw.loc[:, ~bxcol].dropna(how='all', axis=1).dropna(subset=['modelID'], how='any').infer_objects()
        
        #set types
        pars_df2 = pars_df1.astype(
            {'modelID': int, 'tag': str, 'tval_type': str, 
             'aggLevel': int, 'aggType': str, 'dscale_meth': dtype('O'), 'severity': dtype('O'), 
             'resolution': int, 'downSampling': dtype('O'), 'sgType': dtype('O'), 
             'samp_method': dtype('O'), 'zonal_stat': dtype('O'), 'vid': int}        
            ).set_index('modelID')
        
        #===========================================================================
        # check
        #===========================================================================
        pars_df = pars_df2.copy()
        
        assert pars_df.notna().all().all()
        assert pars_df.index.is_unique
        assert pars_df['tag'].is_unique
        assert pars_df.index.name == idn
     
 
        
        #=======================================================================
        # wrap
        #=======================================================================
        log.info('got %s'%str(pars_df.shape))
        return pars_df
    
    def check_mindex(self,  # check names and types
                     mindex,
                     chk_d=None,
                     ):
        """todo: return error messages"""
        #=======================================================================
        # defaults
        #=======================================================================
        # if logger is None: logger=self.logger
        # log=logger.getChild('check_mindex')
        if chk_d is None: chk_d = self.mindex_dtypes
        
        assert isinstance(mindex, pd.MultiIndex), 'bad type: %s'%type(mindex)
        #=======================================================================
        # check types and names
        #=======================================================================
        names_d = {lvlName:i for i, lvlName in enumerate(mindex.names)}
        
        assert not None in names_d, 'got unlabled name'
        
        for name, lvl in names_d.items():
 
            assert name in chk_d, 'name \'%s\' not recognized' % name
            assert mindex.get_level_values(lvl).dtype == chk_d[name], \
                'got bad type on \'%s\': %s' % (name, mindex.get_level_values(lvl).dtype.name)
                
        #=======================================================================
        # check index values
        #=======================================================================
        # totality is unique
        bx = mindex.to_frame().duplicated()
        assert not bx.any(), 'got %i/%i duplicated index entries on %i levels \n    %s' % (
            bx.sum(), len(bx), len(names_d), names_d)
        
        return True, ''
    
    def check_mindex_match(self, #special check of indexes
            mindex,
            mindex_short,
            sort=True,
            ):
        
        #=======================================================================
        # prep
        #=======================================================================
        if sort:
            #mindex = mindex.copy().sortlevel()[0]
            """not working for some reason"""
            mindex_short = mindex_short.copy().sort_values()
            
        assert_func(lambda: self.check_mindex(mindex), msg='left')
        assert_func(lambda: self.check_mindex(mindex_short), msg='right')
 
        
        
         
        
        
        #compress the midex
        chk_index = pd.MultiIndex.from_frame(mindex.droplevel('id').to_frame().reset_index(drop=True).drop_duplicates()).sort_values()
        
        
        #check names match
        miss_l = set(chk_index.names).symmetric_difference(mindex_short.names)
        if len(miss_l)>0:
            return False, 'names mismatch: %s'%miss_l
        
        #=======================================================================
        # #loop and check values on each level
        #=======================================================================
        err_d = dict()
        for lvlName in chk_index.names:
            left_vals = chk_index.unique(lvlName).sort_values()
            right_vals = mindex_short.unique(lvlName).sort_values()
            
            if not np.array_equal(left_vals, right_vals):
                set_d = set_info(left_vals, right_vals, result='counts')
                err_d[lvlName] = 'mismatch \'%s\' w/ %s'%(lvlName, set_d['symmetric_difference'])
                
        
        if len(err_d)>0:
            msg = pprint.PrettyPrinter(indent=4).pformat(err_d)
            print(msg)
            return False, err_d
        
        
        #=======================================================================
        # check lengths
        #=======================================================================
        """even though all the values are the same... teh lengths can be different"""
        if not len(chk_index)==len(mindex_short):
            return False, 'length mismatch mindex(%i) vs. R(%i) = %i'%(
                len(chk_index), len(mindex_short), abs(len(chk_index)-len(mindex_short)))
            
 
        assert_index_equal(chk_index, mindex_short)
        return True, ''
    
    def check_mindex_match_cats(self, #special categorical check on the mindex
                          mindex,
                          mindex_short,
                          glvls = ['studyArea'],#categories to check on 
                          ):
 
 
        #=======================================================================
        # defautls       
        #=======================================================================
        gcn = self.gcn
        assert_func(lambda: self.check_mindex(mindex), msg='mindex')
        assert_func(lambda: self.check_mindex(mindex_short), msg='mindex_short')
 
        #=======================================================================
        # prep
        #=======================================================================
        #clean up the short
        drop_l = set(mindex_short.names).difference(glvls + [gcn])
        if len(drop_l)>0:
            for coln in drop_l:
                assert len(mindex_short.unique(coln))==1
            drop_l = mindex_short.copy().droplevel(drop_l)
        
        #setup the true indexer
        mindex_gb = mindex.to_frame().groupby(level=glvls)
        
        #=======================================================================
        # loop by gruop and check
        #=======================================================================
        err_d = dict()
        for i, (gkeys, gdx) in enumerate(mindex_short.to_frame().groupby(level=glvls)):
            if isinstance(gkeys, str):
                keys_d = {glvls[0]:gkeys} 
            else: 
                keys_d = dict(zip(glvls, gkeys))
 
            #get the mindex for thsi model
            mindex_i = mindex_gb.get_group(gkeys).index
            
            #check this category
            result, err_d_i = self.check_mindex_match(mindex_i, gdx.index)
            if not result:
                err_d[i] = {**keys_d, **err_d_i}
                
        #=======================================================================
        # wrap
        #=======================================================================
        if len(err_d)>0:
            df = pd.DataFrame.from_dict(err_d).T.reset_index(drop=True)
            with pd.option_context('display.max_rows', None,'display.max_columns', None,'display.width',1000, 'display.max_colwidth', 200):
                print(df)
            
            return False, 'mismatch on %i combos of %s \n    %s'%(
                len(df),glvls, df)
            
        return True, ''
    
    
    def get_aggregate(self, #convenicnece for collapsing some data along an index
                      dx_raw, 
                      aggMethod='mean',
                      mindex=None,
                      logger=None,
                      agg_name=None, #index name which will be collapsed
                      ):
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger=self.logger
        log=logger.getChild('get_agg')
        
        log.debug('on %s %s'%(str(dx_raw.shape), type(dx_raw).__name__))
        
        #=======================================================================
        # precheck
        #=======================================================================
        assert isinstance(dx_raw.index, pd.MultiIndex)
        assert isinstance(dx_raw, pd.Series), 'only checked on series'
        
        #=======================================================================
        # prep
        #=======================================================================
        if agg_name is None:
            #get the index name which will be collapsed
            l = list(set(dx_raw.index.names).symmetric_difference(mindex.names))
            assert len(l)==1
            
            agg_name = l[0]
        #=======================================================================
        # aggregate (collapse)
        #=======================================================================
        if len(dx_raw)>len(mindex):
 
            log.debug('collapsing %i to %i on \'%s\' w/ \'%s\''%(len(dx_raw), len(mindex), agg_name, aggMethod))
            #===================================================================
            # checks
            #===================================================================
            #check the names
            miss_l = set(mindex.names).difference(dx_raw.index.names)
            if not len(miss_l)==0:
                raise IOError('missing %i index.names in the data: %s'%(len(miss_l), miss_l))
 
            
            #group on all levels passed in the mindex
            gb = dx_raw.groupby(level=[mindex.names])
            
            #peform the action
            adx1 = getattr(gb, aggMethod)()
 
            #clean up
            adx2 = adx1.reorder_levels(mindex.names).sort_index()
            
            res_dx = adx2
            
        #=======================================================================
        # disagg (expand)
        #=======================================================================
        elif len(dx_raw)<len(mindex):
            
            
            log.debug('expanding %i to %i on \'%s\' w/ \'%s\''%(len(dx_raw), len(mindex), agg_name, aggMethod))
            #check the names
            miss_l = set(dx_raw.index.names).difference(mindex.names)
            assert len(miss_l)==0, 'missing %i index.names in the mindex: %s'%(len(miss_l), miss_l)
            
            #===================================================================
            # prep
            #===================================================================
            #join onto the big index
            """always has to be a frame"""
            big_dx = pd.DataFrame(index=mindex).join(dx_raw)
            #===================================================================
            # simply replicate onto the big
            #===================================================================
            """maintain the mean per group"""
            if aggMethod=='join':
               
                res_dx = big_dx
                
                if isinstance(dx_raw, pd.Series):
                    """probably a nicer way..."""
                    res_dx = res_dx.iloc[:,0]
                    
            #===================================================================
            # divide equally onto the big
            #===================================================================
            #"""maintain the total per group"""
            elif aggMethod=='divide':
                
                #group size
                """only care bout the index really"""
                cnt_serx1 = big_dx.groupby(level=dx_raw.index.names).count().iloc[:,0].rename('group_size')
                
                #expand
                cnt_serx2 = pd.DataFrame(index=mindex).join(cnt_serx1).iloc[:,0] #join on
                
                #reduce
                res_dx = big_dx.divide(cnt_serx2, axis='index')
                

            
            else: 
                raise IOError(aggMethod)
            
            #===================================================================
            # check
            #===================================================================
            try:
                #do the reverse aggregation
                aggMethod_invert = {v:k for k,v in self.agg_inv_d.items()}[aggMethod]
                f = getattr(res_dx.groupby(level=dx_raw.index.names), aggMethod_invert)
                chk_serx = f().reorder_levels(dx_raw.index.names).sort_index()
                
                """not sure if we ever even use dataframes..."""
                assert_series_equal(chk_serx, dx_raw)
            except Exception as e:
                raise Error('failed disagg check w/ %s'%e)
            
        #=======================================================================
        # equialent
        #=======================================================================
        else:
            log.debug('equivalent')
            assert len(dx_raw)==len(mindex)
            
            #just add the level
            if agg_name in mindex.names:
                res_dx = pd.DataFrame(index=mindex).join(dx_raw).iloc[:,0]
 
            #drop the level
            elif agg_name in dx_raw.index.names:
                res_dx = dx_raw.droplevel(agg_name).reorder_levels(mindex.names).sort_index()
                
            else: raise IOError(agg_name)
            
            assert np.array_equal(res_dx.sort_index().values, dx_raw.sort_index().values)
                
 
        #=======================================================================
        # wrap
        #=======================================================================
        res_dx = res_dx.reorder_levels(mindex.names)
        try:
            assert_index_equal(mindex, res_dx.index)
        except Exception as e:
            raise Error(e)
   
        assert isinstance(res_dx, type(dx_raw))
        return res_dx
 
      
 

class Model(HydSession, QSession):  # single model run
 
    
    #supported parameter values
    pars_lib = {

        'aggType':{'vals':['none', 'gridded', 'convexHulls'],            'dkey':'finv_agg_d'},
        'aggLevel':{'vals':list(range(500)),              'dkey':'finv_agg_d'},
        'sgType':{'vals':['centroids', 'poly'],                         'dkey':'finv_sg_d'},
        
        'tval_type':{'vals':['uniform', 'rand', 'footprintArea'],        'dkey':'tvals_raw'},
        'dscale_meth':{'vals':['centroid', 'none', 'area_split'],       'dkey':'tvals'},
        
        'severity':{'vals':['hi', 'lo'],                                'dkey':'drlay_d'},
        'resolution':{'vals':list(range(10,500)),                         'dkey':'drlay_d'},
        'downSampling':{'vals':['none','Average', 'Mode', 'Nearest neighbour'],                    'dkey':'drlay_d'},
        'dsampStage':{'vals':['none', 'pre', 'post'],                  'dkey':'drlay_d'},
        
        'samp_method':{'vals':['points', 'zonal', 'true_mean'],         'dkey':'rsamps'},
        'zonal_stat':{'vals':['Mean', 'Minimum', 'Maximum', 'Median', 'none'], 'dkey':'rsamps'},
                
        'vid':{'vals':[49, 798,811, 0] + list(range(1000,1100)),                                 'dkey':'vfunc'},


        }
    

    
    def __init__(self,
                 name='hyd',
                 proj_lib={},
                 trim=True,  # whether to apply aois
                 data_retrieve_hndls={}, exit_summary=False,
                 **kwargs):
        
        #===========================================================================
        # HANDLES-----------
        #===========================================================================
        # configure handles
        data_retrieve_hndls = { **{

            # aggregating inventories
            'finv_agg_d':{  # lib of aggrtevated finv vlays
                'compiled':lambda **kwargs:self.load_layer_d(**kwargs),  # vlays need additional loading
                'build':lambda **kwargs: self.build_finv_agg(**kwargs),
                },
            
            'finv_agg_mindex':{  # map of aggregated keys to true keys
                'compiled':lambda **kwargs:self.load_pick(**kwargs),
                'build':lambda **kwargs:self.build_finv_agg_mindex(**kwargs),
                #'build':lambda **kwargs: self.build_finv_agg(**kwargs), 
                    #this function does build.. but calling it with this key makes for ambigious parameterization
                },
            
            'finv_sg_d':{  # sampling geometry
                'compiled':lambda **kwargs:self.load_layer_d(**kwargs),  # vlays need additional loading
                'build':lambda **kwargs: self.build_sampGeo(**kwargs),
                },
            
            'tvals_raw':{#total asset values (of the raw finv)
                 'compiled':lambda **kwargs:self.load_pick(**kwargs),
                'build':lambda **kwargs: self.build_tvals_raw(**kwargs),
                },
            
            'tvals':{  # total asset values aggregated
                'compiled':lambda **kwargs:self.load_pick(**kwargs),
                'build':lambda **kwargs: self.build_tvals(**kwargs),
                },
            
            'drlay_d':{ #depth raster layers
                'compiled':lambda **kwargs:self.load_layer_d(**kwargs),
                'build':lambda **kwargs: self.build_drlay(**kwargs),
                },
            
            'rsamps':{  # depth raster samples
                'compiled':lambda **kwargs:self.load_pick(**kwargs),
                'build':lambda **kwargs: self.build_rsamps(**kwargs),
                },
            
            'rloss':{
                'compiled':lambda **kwargs:self.load_pick(**kwargs),
                'build':lambda **kwargs:self.build_rloss(**kwargs),
                },
            'tloss':{  # total losses
                'compiled':lambda **kwargs:self.load_pick(**kwargs),
                'build':lambda **kwargs:self.build_tloss(**kwargs),
                },
 
            
            }, **data_retrieve_hndls}
        
        super().__init__(exit_summary=exit_summary,
                         data_retrieve_hndls=data_retrieve_hndls, name=name,
                         **kwargs)
        
        """
        data_retrieve_hndls['drlay_d']
        """
        
        #=======================================================================
        # simple attach
        #=======================================================================
        self.proj_lib = proj_lib
        self.trim = trim
 
        

        
 

    
    #===========================================================================
    # WRITERS---------
    #===========================================================================
    def write_summary(self, #write a nice model run summary 
                      dkey_l=['tloss', 'rloss', 'rsamps'],
                      out_fp=None,
                      ):
        
        #=======================================================================
        # defaults
        #=======================================================================
        log = self.logger.getChild('write_summary')
        if out_fp is None: out_fp = os.path.join(self.out_dir, 'summary_%s.xls'%self.longname)
        
        smry_lib = dict()
        log.info('preparing summary on %i dkeys: %s'%(len(dkey_l), dkey_l))
        #=======================================================================
        # retrieve
        #=======================================================================
        
        
        meta_d = self._get_meta()
        
        #=======================================================================
        # summary page
        #=======================================================================
        #clean out
        del meta_d['bk_lib']
        
        for studyArea in self.proj_lib.keys():
            del meta_d[studyArea]
        
        meta_d['dkey_l'] = dkey_l
        
        #force to strings
        d = {k:str(v) for k,v in meta_d.items()}
        
        smry_lib['smry'] = pd.Series(d).rename('').to_frame()
        
        #=======================================================================
        # parameter page
        #=======================================================================
        
        smry_lib['bk_lib'] = pd.DataFrame.from_dict(self.bk_lib).T.stack().to_frame()
        
        #=======================================================================
        # study Areas page
        #=======================================================================
        
        smry_lib['proj_lib'] = pd.DataFrame.from_dict(self.proj_lib).T
        
        #=======================================================================
        # data/results summary
        #=======================================================================
        d = dict()
 
        for dkey in dkey_l:
            
            data = self.retrieve(dkey) #best to call this before finv_agg_mindex
            log.info('collecting stats on \'%s\' w/ %i'%(dkey, len(data)))
            if isinstance(data, pd.DataFrame):
                serx = data.iloc[:, -1] #last one is usually total loss
            else:
                serx=data
                
                
            res_meta_d = dict()
            
            for stat in ['count', 'min', 'mean', 'max', 'sum']:
                f = getattr(serx, stat)
                res_meta_d[stat] = f() 
                
            res_meta_d['shape'] = str(serx.shape)
            
            d[dkey] = res_meta_d
        
 
            
        smry_lib['data_smry'] = pd.DataFrame.from_dict(d).T
        
        #=======================================================================
        # complete datasets
        #=======================================================================
        """only worth writing tloss as this holds all the data"""
        smry_lib = {**smry_lib, **{'tloss':self.retrieve('tloss')}}
        
        #=======================================================================
        # write
        #=======================================================================
        #write a dictionary of dfs
        with pd.ExcelWriter(out_fp) as writer:
            for tabnm, df in smry_lib.items():
                if len(df)>10000:continue
                df.to_excel(writer, sheet_name=tabnm, index=True, header=True)
                
        log.info('wrote %i tabs to %s'%(len(smry_lib), out_fp))
        
        return smry_lib
        
 

        

    
    #===========================================================================
    # DATA CONSTRUCTION-------------
    #===========================================================================
    def run_dataGeneration(self, #convenience for executing data generation calls insequence
                           ):
        #=======================================================================
        # defaults
        #=======================================================================
        log = self.logger.getChild('rDg')
        
        #=======================================================================
        # build the aggregated inventories
        #=======================================================================
        finv_agg_d = self.retrieve('finv_agg_d', logger=log)
 
        
        #retrieve the index linking back to the raw
        finv_agg_mindex = self.retrieve('finv_agg_mindex', logger=log)
 
        #=======================================================================
        # populate the 'total asset values' on each asset
        #=======================================================================
        #raw values
        finv_true_serx = self.retrieve('tvals_raw', logger=log)
 
        
        #aggregated values
        finv_agg_serx = self.retrieve('tvals', logger=log)
        
        #=======================================================================
        # build hazard data
        #=======================================================================
        drlay_d = self.retrieve('drlay_d', logger=log)
        
        #=======================================================================
        # wrap
        #=======================================================================
        log.info('done')
        
    def build_finv_agg(self,  # build aggregated finvs
                       dkey=None,
                       
                       # control aggregated finv type 
                       aggType=None,
                       aggLevel=None,
                       
                       #defaults
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
 
        assert dkey in ['finv_agg_d',
                        #'finv_agg_mindex', #makes specifycing keys tricky... 
                        ], 'bad dkey: \'%s\''%dkey
 
        gcn = self.gcn
        log.info('building \'%s\' ' % (aggType))
 
        assert not aggLevel is None
        aggLevel = int(aggLevel)
        #assert isinstance(aggLevel, int), 'got bad aggLevel type: %s (%s)'%(aggLevel, type(aggLevel))
        if not aggType == 'none': 
            assert aggLevel>0, 'got bad aggLevel: %s for aggType=%s'%(aggLevel, aggType)
        else:
            assert aggLevel==0, 'got bad aggLevel: %s'%aggLevel
        #=======================================================================
        # retrive aggregated finvs------
        #=======================================================================
        """these should always be polygons"""
        
        finv_agg_d, finv_gkey_df_d = dict(), dict()
        
        if aggType == 'none':  # see Test_p1_finv_none
            
            res_d = self.sa_get(meth='get_finv_clean', write=False, dkey=dkey, get_lookup=True, **kwargs)
 
        elif aggType == 'gridded':  # see Test_p1_finv_gridded
            res_d = self.sa_get(meth='get_finv_gridPoly', write=False, dkey=dkey, aggLevel=aggLevel, **kwargs)
            
        elif aggType=='convexHulls':
            res_d = self.sa_get(meth='get_finv_convexHull', write=False, dkey=dkey, aggLevel=aggLevel, **kwargs)
 
        else:
            raise Error('not implemented')
        
        # unzip
        for studyArea, d in res_d.items():
            finv_gkey_df_d[studyArea], finv_agg_d[studyArea] = d
            
        assert len(finv_gkey_df_d) > 0, 'got no links!'
        assert len(finv_agg_d) > 0, 'got no layers!'
        #=======================================================================
        # check
        #=======================================================================
        for studyArea, vlay in finv_agg_d.items():
            """relaxing this
            assert vlay.wkbType()==3, 'requiring singleParts'"""
            assert 'Polygon' in QgsWkbTypes().displayString(vlay.wkbType())
        
        #=======================================================================
        # handle layers----
        #=======================================================================
        dkey1 = 'finv_agg_d'
        if write:
            self.store_layer_d(finv_agg_d, dkey1, logger=log)
        
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
        
        dkey1 = 'finv_agg_mindex'
        serx = pd.concat(finv_gkey_df_d, verify_integrity=True).iloc[:, 0].sort_index()
 
        serx.index.set_names('studyArea', level=0, inplace=True)
        agg_mindex = serx.to_frame().set_index(gcn, append=True).swaplevel().sort_index().index
        
        #=======================================================================
        # check
        #=======================================================================
        """re-extracting indexes from layers"""
        d = dict()
        for studyArea, finv_vlay in finv_agg_d.items():
            
            df = vlay_get_fdf(finv_vlay)
            assert len(df.columns)==1
            
            d[studyArea] = df.set_index(gcn)
 
        
        chk_mindex = pd.concat(d, names=agg_mindex.names[0:2]).index.sortlevel()[0]
        """
        view(chk_mindex.to_frame())
        view(agg_mindex.to_frame())
        """
        
        assert_func(lambda: self.check_mindex_match_cats(agg_mindex, chk_mindex), msg='fing_df vs fing_agg_vlays for aggType=%s'%aggType)

        
        #=======================================================================
        # write
        #=======================================================================
        # save the pickle
        if write:
            self.ofp_d[dkey1] = self.write_pick(agg_mindex,
                           os.path.join(self.wrk_dir, '%s_%s.pickle' % (dkey1, self.longname)), logger=log)
        
        # save to data
        self.data_d[dkey1] = copy.deepcopy(agg_mindex)
 
        #=======================================================================
        # return requested data
        #=======================================================================
        """while we build two results here... we need to return the one specified by the user
        the other can still be retrieved from the data_d"""
 
        if dkey == 'finv_agg_d':
            result = finv_agg_d
        elif dkey == 'finv_agg_mindex':
            result = agg_mindex
        
        return result
    
    def build_finv_agg_mindex(self, #special wrapper for finv_agg_mindex requests
                              dkey=None,
                              **kwargs):
        """to force consistent variable handling on these siblings,
        here we just call the other"""
        
        assert dkey=='finv_agg_mindex'
        #assert len(kwargs)==0, 'specify kwargs on dkey=finv_agg_d'
        
        #call the sibling
        self.retrieve('finv_agg_d', **kwargs)
        
        return self.data_d[dkey]
        
    def build_tvals_raw(self,  # get the total values on each asset
                    dkey=None,
                    
                    #data
                    #finv_agg_d=None, #for consistency?
                    mindex=None,
                    
                    #parameters
                    tval_type='uniform',  # type for total values
                    
                    #parameters (default)
                    prec=2,
                    normed=True, #normalize per-studArea
                    norm_scale=1e2, #norm scalar
                        #for very big study areas we scale things up to avoid dangerously low values

                    write=None, logger=None,
                    **kwargs):
        """Warnning: usually called by
            ModelStoc.build_tvals_raw()
                ModelStoch.model_retrieve() #starts a new Model instance
                    
                    Model.build_tvals_raw()
                    """
        
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger=self.logger
        log = logger.getChild('build_tvals_raw')
        assert dkey == 'tvals_raw'
        if prec is None: prec = self.prec
        if write is None: write=self.write
 
        scale_cn = self.scale_cn + '_raw'
        
        #=======================================================================
        # retrieve
        #=======================================================================
        if mindex is None: 
            mindex = self.retrieve('finv_agg_mindex')  # studyArea, id : corresponding gid
 
        log.info('on %i w/ tval_type=%s'%(len(mindex), tval_type))
        #=======================================================================
        # get trues
        #=======================================================================
        vals = None
        if tval_type == 'uniform':
            vals = np.full(len(mindex), 1.0)
        elif tval_type == 'rand':
            vals = np.random.random(len(mindex))
            
        elif tval_type=='footprintArea':
 
            res_d = self.sa_get(meth='get_tvalsR_area', write=False, dkey=dkey, **kwargs)
            
            #join to mindex
            serx1 = pd.concat(res_d, names=[mindex.names[0], mindex.names[2]])            
            finv_true_serx = pd.DataFrame(index=mindex).join(serx1).iloc[:,0]
            
            finv_true_serx = finv_true_serx.reorder_levels(mindex.names)
 
            
        else:
            raise Error('unrecognized')

        if not vals is None:
            finv_true_serx = pd.Series(vals, index=mindex, name=scale_cn)
 
        #=======================================================================
        # check
        #=======================================================================
        assert finv_true_serx.notna().all()
        #assert finv_true_serx.min()>10.0
        self.check_mindex(finv_true_serx.index)
        
        #=======================================================================
        # normalize
        #=======================================================================
        """normalizng to the study area so the total value of all assets per study area always equals 100"""
        if normed:
            log.debug('on %i'%len(finv_true_serx))
            
            l = list()
            for studyArea, gserx in finv_true_serx.groupby(level='studyArea'):
                l.append(gserx.multiply(norm_scale).divide(gserx.sum()))
                
            serx = pd.concat(l)
            
            assert (serx.groupby(level='studyArea').sum().round(self.prec)==norm_scale).all()
                
            finv_true_serx = serx
            
        #=======================================================================
        # check
        #=======================================================================
        assert_index_equal(mindex, finv_true_serx.index)

 
        

        #=======================================================================
        # wrap
        #=======================================================================
        """todo: add writer for layers (join back normalized values)"""
        if write:
            self.ofp_d[dkey] = self.write_pick(finv_true_serx,
                                   os.path.join(self.wrk_dir, '%s_%s.pickle' % (dkey, self.longname)),
                                   logger=log)

        return finv_true_serx
    
    def build_tvals(self, #downscaling raw asset values onto the aggregated inventory
                    dkey = 'tvals',
                    #data
                    #mindex=None, 
                    tvals_raw_serx=None,
                    
                    #data (for area_split)                    
                    finv_agg_d=None,
                    #proj_lib=None,
                    
                    #parameterse
                    dscale_meth='centroid',
 
                    
                    #parameters (defualt)
                    write=None,logger=None,**kwargs):
        """Warnning: usually called by
            ModelStoc.build_tvals()
                ModelStoch.model_retrieve() #starts a new Model instance
                    
                    Model.build_tvals()
                    """
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger=self.logger
        log = logger.getChild('build_tvals')
        if write is None: write=self.write
        rcoln = self.scale_cn
        gcn = self.gcn
        assert dkey == 'tvals'
        
        #=======================================================================
        # if mindex is None: 
        #     mindex = self.retrieve('finv_agg_mindex')  # studyArea, id : corresponding gid
        #=======================================================================
            
        #generate asset values on the raw
        if tvals_raw_serx is None:
            assert self.session is None, 'need to pass this explicitly for nested runs'
            """changed this to be indenpendent of build_tvals"""
            tvals_raw_serx = self.retrieve('tvals_raw')
        
        mindex = tvals_raw_serx.index
        
        #=======================================================================
        # check 
        #=======================================================================
        self.check_mindex(mindex)
        
        assert isinstance(tvals_raw_serx, pd.Series)
        #if we are already a true finv
        if np.array_equal(mindex.get_level_values(1).values,mindex.get_level_values(2).values):
            assert dscale_meth=='none', 'finv is already aggregated'
            
        
        #=======================================================================
        # aggregate trues
        #=======================================================================
        log.info('on %i w/ dscale_meth=%s'%(len(tvals_raw_serx), dscale_meth))
        if dscale_meth == 'centroid':
            """because the finv_agg_mindex is generated using the centroid intersect
                see StudyArea.get_finv_gridPoly
                se can do a simple groupby to perform this type of downscaling"""
            
                
            finv_agg_serx = tvals_raw_serx.groupby(level=mindex.names[0:2]).sum().rename(rcoln)
            
        #no aggregation: base runs
        elif dscale_meth=='none': 
            """this should return the same result as the above groupby on 1:1"""
            finv_agg_serx = tvals_raw_serx.droplevel(2).rename(rcoln)
            
            assert len(mindex.unique(gcn)) == len(mindex.unique('id')), 'passed dscale_meth=none on an index that needs downscaling'

            
        elif dscale_meth == 'area_split':
            """this is tricky as we are usually executting a child run at this point"""
            
            #retreives
            if finv_agg_d is None:
                assert 'finv_agg_d' in self.data_d, 'problem with cascade... check ModelStoch.build_tvals()'
                finv_agg_d = self.retrieve('finv_agg_d')
                
#===============================================================================
#             if proj_lib is None:
# 
#                 if not self.session is None:
#                     proj_lib=self.session.proj_lib
#                     
#                 else: #test_model (all instances shoul dhave this attribute ... may be empty)
#                     proj_lib=self.proj_lib
#                     
#  
#                 
#                 assert len(proj_lib)>0, 'failed to get any proj_lib'
#===============================================================================
                
            #call for each study area
            d = self.sa_get(meth='get_tvals_aSplit', write=False, dkey=dkey, 
                                        tvals_raw_serx=tvals_raw_serx, finv_agg_d=finv_agg_d,
                                        #proj_lib=proj_lib,
                                        **kwargs)
            
            finv_agg_serx = pd.concat(d, names=tvals_raw_serx.index.names[0:2])
            
 
        else:
            raise Error('unrecognized dscale_meth=%s'%dscale_meth)
        
        #=======================================================================
        # checks
        #=======================================================================
        assert finv_agg_serx.name==rcoln, 'bad name on result: %s'%finv_agg_serx.name
        
        #collapsed gid index
        assert_func(lambda: self.check_mindex_match(mindex, finv_agg_serx.index), msg='tvals')
        #=======================================================================
        # chk_index = pd.MultiIndex.from_frame(mindex.droplevel(2).to_frame().reset_index(drop=True).drop_duplicates([gcn, 'studyArea']))  
        #    
        # assert np.array_equal(finv_agg_serx.index, chk_index)
        #=======================================================================
        #=======================================================================
        # wrap
        #=======================================================================
        if write:
            """usually false... written by modelStoch"""
            self.ofp_d[dkey] = self.write_pick(finv_agg_serx,
                                   os.path.join(self.wrk_dir, '%s_%s.pickle' % (dkey, self.longname)),
                                   logger=log)

        return finv_agg_serx
        
    def build_drlay(self, #buidl the depth rasters
                    dkey=None,
 
                   write=None, logger=None,
                    **kwargs):
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger=self.logger
        log = logger.getChild('build_drlay')
 
 
        if write is None: write=self.write
        assert dkey=='drlay_d'
        
        #=======================================================================
        # retrive rasters per StudyArea------
        #=======================================================================
        """leaving everything on the StudyArea to speed things up"""
        
        #execute
        res_d = self.sa_get(meth='get_drlay', logger=log, dkey=dkey, write=False, **kwargs)
 
        
        #=======================================================================
        # handle layers----
        #=======================================================================
 
        if write:
            self.store_layer_d(res_d, dkey, logger=log)
        
        """needed?"""
        self.data_d[dkey] = res_d
        
        return res_d
    
    #===========================================================================
    # INTERSECTION----------
    #===========================================================================
    def run_intersection(self, 
                         ):
        #=======================================================================
        # defaults
        #=======================================================================
        log = self.logger.getChild('rInt')
        
        #=======================================================================
        # build sampling geometry
        #=======================================================================
        finv_d = self.retrieve('finv_sg_d', logger=log)
        
        #=======================================================================
        # sample depth raseres
        #=======================================================================
        rsamp_serx = self.retrieve('rsamps', logger=log)
        
        #=======================================================================
        # wrap
        #=======================================================================
        log.info('done')
    
    def build_sampGeo(self,  # sampling geometry no each asset
                     dkey='finv_sg_d',
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
        assert dkey == 'finv_sg_d'
        if logger is None: logger=self.logger
        log = logger.getChild('build_sampGeo')
        if write is None: write=self.write
        
        if finv_agg_d is None: finv_agg_d = self.retrieve('finv_agg_d', write=write)
 
        #=======================================================================
        # loop each polygon layer and build sampling geometry
        #=======================================================================
        log.info('on %i w/ %s' % (len(finv_agg_d), sgType))
        res_d = dict()
        for studyArea, poly_vlay in finv_agg_d.items():
 
            log.info('on %s w/ %i feats' % (studyArea, poly_vlay.dataProvider().featureCount()))
            
            if sgType == 'centroids':
                """works on point layers"""
                sg_vlay = self.centroids(poly_vlay, logger=log)
                
            elif sgType == 'poly':
                assert 'Polygon' in QgsWkbTypes().displayString(poly_vlay.wkbType()), 'bad type on %s' % (studyArea)
                poly_vlay.selectAll()
                sg_vlay = self.saveselectedfeatures(poly_vlay, logger=log)  # just get a copy
                
            else:
                raise Error('not implemented')
            
            #===============================================================
            # wrap
            #===============================================================
            sg_vlay.setName('%s_%s' % (poly_vlay.name(), sgType))
            
            res_d[studyArea] = sg_vlay
        
        #=======================================================================
        # store layers
        #=======================================================================
        if write: ofp_d = self.store_layer_d(res_d, dkey, logger=log)
        
        return res_d
    
    def build_rsamps(self,  # get raster samples for all finvs
                     dkey=None,
                     samp_method='points',  # method for raster sampling
                     finv_sg_d=None, drlay_d=None,
                     
                     mindex=None, #special catch for test consitency
                     zonal_stat=None, 
                     
                     
                     #gen
                     prec=None, idfn=None,write=None, logger=None,
                     **kwargs):
        """
        keeping these as a dict because each studyArea/event is unique
        """
        #=======================================================================
        # defaults
        #=======================================================================
        assert dkey == 'rsamps', dkey
        if logger is None: logger=self.logger
        log = logger.getChild('build_rsamps')
 
        if prec is None: prec=self.prec
        if write is None: write=self.write
        
        #get the depth rasters
        if drlay_d is None:drlay_d = self.retrieve('drlay_d')

        #=======================================================================
        # generate depths------
        #=======================================================================
        #=======================================================================
        # simple point-raster sampling
        #=======================================================================
        if samp_method in ['points', 'zonal']:
            #deefaults
            gcn = self.gcn
            if idfn is None: idfn=gcn
            

            
            if finv_sg_d is None: finv_sg_d = self.retrieve('finv_sg_d')
            
            #execute
            res_d = self.sa_get(meth='get_rsamps', logger=log, dkey=dkey, write=False,
                                finv_sg_d=finv_sg_d, idfn=idfn, samp_method=samp_method,
                                zonal_stat=zonal_stat,drlay_d=drlay_d, **kwargs)
            
            #post
            dxind1 = pd.concat(res_d, verify_integrity=True)
 
            res_serx = dxind1.stack(
                dropna=True,  # zero values need to be set per-study area
                ).rename('depth').swaplevel().sort_index(axis=0, level=0, sort_remaining=True) 
                
            res_serx.index.set_names(['studyArea', 'event', gcn], inplace=True)
 
            

        #=======================================================================
        # use mean depths from true assets (for error isolation only)
        #=======================================================================
        elif samp_method == 'true_mean':
            res_serx = self.rsamp_trueMean(dkey,  logger=log, mindex=mindex, drlay_d=drlay_d, **kwargs)
        else:raise Error('bad key')
        
        #=======================================================================
        # post clean
        #=======================================================================
        res_serx = res_serx.round(prec).astype(float)
        #=======================================================================
        # checks
        #=======================================================================
        #type checks
        assert isinstance(res_serx, pd.Series)
        self.check_mindex(res_serx.index)
        
        #value checks
        assert res_serx.notna().all()
        bx = res_serx < 0.0
        if bx.any().any():
            raise Error('got some negative depths')
        
 
        assert res_serx.max()<=100
        
        
 
        #=======================================================================
        # write
        #=======================================================================
        if write:
            self.ofp_d[dkey] = self.write_pick(res_serx, os.path.join(self.wrk_dir, '%s_%s.pickle' % (dkey, self.longname)),
                            logger=log)
        
        return res_serx
    
    #===========================================================================
    # LOSS CALCS----------
    #===========================================================================
    def run_lossCalcs(self,
                      ):
        #=======================================================================
        # defaults
        #=======================================================================
        log=self.logger.getChild('rLoss')
        
        #=======================================================================
        # run depths through vfuncs
        #=======================================================================
        rloss_dx = self.retrieve('rloss', logger=log)
        
        #=======================================================================
        # multiply by total value to get total loss
        #=======================================================================
        tloss_dx = self.retrieve('tloss', logger=log)
        
        #=======================================================================
        # wrap
        #=======================================================================
        log.info('finished')
        
        
    
    def build_rloss(self,  # calculate relative loss from rsamps on each vfunc
                    dkey=None,
                    
                    dxser=None,
                    vid=798,
                    
                    #gens
                    write=None, prec=None,  # precision for RL
                    logger=None,
                    **kwargs):
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger=self.logger
        log = logger.getChild('build_rloss')
        assert dkey == 'rloss'
        if prec is None: prec = self.prec
        if write is None: write=self.write
        
        
        if dxser is None: dxser = self.retrieve('rsamps')
        log.debug('loaded %i rsamps' % len(dxser))
        #=======================================================================
        # #retrieve child data
        #=======================================================================
 
        # vfuncs
        vfunc = self.build_vfunc(vid=vid, **kwargs)
 
        #=======================================================================
        # loop and calc
        #=======================================================================
        log.info('getting impacts from vfunc %i and %i depths' % (
            vfunc.vid, len(dxser)))
 
 
        ar = vfunc.get_rloss(dxser.values, prec=prec)
        
        assert ar.max() <= 100, '%s returned some RLs above 100' % vfunc.name
 
        
        #=======================================================================
        # combine
        #=======================================================================
        rdxind = dxser.to_frame().join(pd.Series(ar, index=dxser.index, name='rl', dtype=float).round(prec)).astype(float)
        
 
        
        log.info('finished on %s' % str(rdxind.shape))
        
        self.check_mindex(rdxind.index)
 
        #=======================================================================
        # write
        #=======================================================================
        if write:
            self.ofp_d[dkey] = self.write_pick(rdxind,
                                   os.path.join(self.wrk_dir, '%s_%s.pickle' % (dkey, self.longname)),
                                   logger=log)
        
        return rdxind
    
    def build_tloss(self,  # get the total loss
                    #data retrieval
                    dkey=None,
                    tv_data = None,
                    rl_dxind = None,
                    
 
                    #gen
                    write=None, logger=None,
                    
                    ):
        
        #=======================================================================
        # defaults
        #=======================================================================
        #scale_cn = self.scale_cn
        if logger is None: logger=self.logger
        log = logger.getChild('build_tloss')
        assert dkey == 'tloss'
        if write is None: write=self.write
        #=======================================================================
        # retriever
        #=======================================================================
        
        if tv_data is None: 
            tv_data = self.retrieve('tvals')  # studyArea, id : grid_size : corresponding gid
        
        if rl_dxind is None: 
            #raise Error('the instancing in model_get() is corrupting the data_retrieve_hndls... self is now an empty Model isntance?')
            rl_dxind = self.retrieve('rloss')
        
        #rlnames_d = {lvlName:i for i, lvlName in enumerate(rl_dxind.index.names)}
        
        #assert isinstance(tv_data, pd.Series)
        if isinstance(tv_data, pd.Series):
            """reshaping tval data into multindex for backwards compatability w/ Model runs"""
            tv_data = tv_data.to_frame()
            tv_data = pd.concat({'tvals':tv_data}, axis=1, names=['dkey', 'iter'])
            
        assert np.array_equal(np.array(tv_data.columns.names), np.array(['dkey', 'iter']))
 
        #=======================================================================
        # join tval and rloss
        #=======================================================================
        #promote the rloss into dkeys
        rl_dx = pd.concat(
            {'rloss':rl_dxind.loc[:, ['rl']],
             'rsamps':rl_dxind.loc[:, ['depth']],
             }, axis=1, names = tv_data.columns.names) 
        
        dxind1 = rl_dx.join(tv_data, on=tv_data.index.names)
        
        #assert dxind1[scale_cn].notna().all()
        #=======================================================================
        # calc total loss (loss x scale)
        #=======================================================================
        tl_dxind = dxind1.loc[:, idx['tvals', :]].multiply(dxind1.loc[:, idx['rloss', :]].values, axis=0)
        
        #relable
        tl_dxind.columns = tl_dxind.columns.remove_unused_levels()
        tl_dxind.columns.set_levels([dkey], level=0, inplace=True)
 
        #=======================================================================
        # join back
        #=======================================================================
        #drop the scale to a singiel columns
        #=======================================================================
        # if not scale_cn in dxind1.columns:
        #     dxind1[scale_cn] = dxind1.loc[:, tv_data.columns].mean(axis=1)
        #     dxind1 = dxind1.drop(tv_data.columns, axis=1)
        #=======================================================================
            
        dxind2 = dxind1.join(tl_dxind)
 

        
        #=======================================================================
        # check
        #=======================================================================
        self.check_mindex(dxind2.index)
 
        #=======================================================================
        # wrap
        #=======================================================================
 
        serx1 = dxind2.loc[:, idx['tloss', :]].mean(axis=1).rename('tloss')
            
        mdf = pd.concat({
            'max':serx1.groupby(level='event').max(),
            'count':serx1.groupby(level='event').count(),
            'sum':serx1.groupby(level='event').sum(),
            }, axis=1)
        
        log.info('finished w/ %s and totalLoss: \n%s' % (
            str(dxind2.shape),
            # dxind3.loc[:,tval_colns].sum().astype(np.float32).round(1).to_dict(),
            mdf
            ))

        if write: 
            self.ofp_d[dkey] = self.write_pick(dxind2,
                                   os.path.join(self.wrk_dir, '%s_%s.pickle' % (dkey, self.longname)),
                                   logger=log)

        return dxind2
    
  
 
        
    

 
    #===========================================================================
    # HELPERS--------
    #===========================================================================
    
    def vectorize(self, #attach tabular results to vectors
                  dxind,
                  finv_agg_d=None,
                  logger=None):
        """
        creating 1 layer per event
            this means a lot of redundancy on geometry
        """
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger=self.logger
        log=logger.getChild('vectorize')
        
        if finv_agg_d is None: finv_agg_d = self.retrieve('finv_agg_d', write=False)
        gcn = self.gcn
        
        log.info('on %i in %i studyAreas'%(len(dxind), len(finv_agg_d)))
        
        assert gcn in dxind.index.names
        #=======================================================================
        # loop and join for each studyArea+event
        #=======================================================================
        cnt = 0
        res_lib = {k:dict() for k in dxind.index.unique('studyArea')}
        
        keyNames = dxind.index.names[0:2]
        for keys, gdf_raw in dxind.groupby(level=keyNames):
            #setup group 
            keys_d = dict(zip(keyNames, keys))
            log.debug(keys_d)
            gdf = gdf_raw.droplevel(keyNames) #drop grouping keys
            assert gdf.index.name==gcn
            
            #get the layer for this
            finv_agg_vlay = finv_agg_d[keys_d['studyArea']]
            df_raw = vlay_get_fdf(finv_agg_vlay).sort_values(gcn).loc[:, [gcn]]
            
            #check key match
            d = set_info(df_raw[gcn], gdf.index)
            assert len(d['diff_left'])==0, 'some results keys not on the layer \n    %s'%d
            
            #join results
            res_df = df_raw.join(gdf, on=gcn).dropna(subset=['tl'], axis=0)
            assert len(res_df)==len(gdf)
            
            #create the layer
            finv_agg_vlay.removeSelection() 
            finv_agg_vlay.selectByIds(res_df.index.tolist()) #select just those with data
            geo_d = vlay_get_geo(finv_agg_vlay, selected=True, logger=log)
            res_vlay= self.vlay_new_df(res_df, geo_d=geo_d, logger=log, 
                                layname='%s_%s_res'%(finv_agg_vlay.name().replace('_', ''), keys_d['event']))
            
            #===================================================================
            # wrap
            #===================================================================
            res_lib[keys[0]][keys[1]]=res_vlay
            cnt+=1
            
        #=======================================================================
        # wrap
        #=======================================================================
        log.info('finished on %i layers'%cnt)
        
        return res_lib
            
 
    def load_layer_d(self,  # generic retrival for finv type intermediaries
                  fp=None, dkey=None,
                  **kwargs):
        #=======================================================================
        # defaults
        #=======================================================================
        log = self.logger.getChild('load_layer_d.%s' % dkey)
        assert dkey in [  'finv_agg_d', 'finv_sg_d', 'drlay_d'], dkey
        
        vlay_fp_lib = self.load_pick(fp=fp, dkey=dkey)  # {study area: aggLevel : vlay filepath}}
        
        # load layers
        lay_d = dict()
        
        for studyArea, fp in vlay_fp_lib.items():
 
            log.info('loading %s from %s' % (studyArea, fp))
            
            """will throw crs warning"""
            ext = os.path.splitext(os.path.basename(fp))[1]
            #===================================================================
            # vectors
            #===================================================================
            if ext in ['.gpkg', '.geojson']:
            
                lay_d[studyArea] = self.vlay_load(fp, logger=log, 
                                               #set_proj_crs=False, #these usually have different crs's
                                                       **kwargs)
            elif ext in ['.tif']:
                lay_d[studyArea] = self.rlay_load(fp, logger=log, 
                                               #set_proj_crs=False, #these usually have different crs's
                                                       **kwargs)
            else:
                raise Error('unrecognized filetype: %s'%ext)
        
        return lay_d
    
    
    

    def store_layer_d(self,  # consistent storage of finv containers 
                       layer_d,
                       dkey,
                       out_dir=None,
                       logger=None,
                       write_pick=True,
                       overwrite=None,
                       add_subfolders=True,
                       compression=None,
                       **kwargs):
        
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger = self.logger
        log = logger.getChild('store_layer_d')
        if out_dir is None: out_dir = os.path.join(self.wrk_dir, dkey)
        if overwrite is None: overwrite=self.overwrite
        
        log.info('writing \'%s\' layers to %s' % (dkey, out_dir))
        
        #=======================================================================
        # write layers
        #=======================================================================
        ofp_d = dict()
        cnt = 0
        for studyArea, layer in layer_d.items():
            #===================================================================
            # # setup directory
            #===================================================================
            if add_subfolders:
                od = os.path.join(out_dir, studyArea)

            else:
                od = out_dir
            
            if not os.path.exists(od):
                os.makedirs(od)
                
            #===================================================================
            # filepaths
            #===================================================================
 
            out_fp = os.path.join(od, layer.name())
            
 
            #===================================================================
            # write vectors
            #===================================================================
            if isinstance(layer, QgsVectorLayer):
                # write each sstudy area
                ofp_d[studyArea] = self.vlay_write(layer,out_fp,logger=log, **kwargs)
            elif isinstance(layer, QgsRasterLayer):
                ofp_d[studyArea] = self.rlay_write(layer,ofp=out_fp,logger=log, compression=compression,
                                                   **kwargs)
            else:
                raise Error('bad type: %s'%type(layer))
            
            cnt += 1
        
        log.debug('wrote %i layers' % cnt)
        #=======================================================================
        # filepahts container
        #=======================================================================
        
        # save the pickle
        if write_pick:
            """cant pickle vlays... so pickling the filepath"""
            self.ofp_d[dkey] = self.write_pick(ofp_d,
                os.path.join(self.wrk_dir, '%s_%s.pickle' % (dkey, self.longname)), logger=log)
        # save to data
        self.data_d[dkey] = layer_d
        return ofp_d
    
    def rsamp_trueMean(self,
                       dkey,
 
                       mindex=None,
                       
                       
                       #true controls
                       samp_methodTrue = 'points',
                       finv_sg_true_d=None,
 
                       logger=None,
                       **kwargs):
        
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger = self.logger
        log = logger.getChild('get_rsamp_trueMean')
        
        #=======================================================================
        # retrival
        #=======================================================================
        
 
        if mindex is None: mindex = self.retrieve('finv_agg_mindex')
        #===================================================================
        # get trues-------
        #===================================================================
        """neeed to use all the same functions to derive the trues"""

        
        #=======================================================================
        # sampling geo
        #=======================================================================
        if finv_sg_true_d is None: 
            #load raw inventories
            finv_agg_true_d = self.sa_get(meth='get_finv_clean', write=False, dkey=dkey, get_lookup=False)
            
            #load the sampling geomtry (default is centroids)
            finv_sg_true_d = self.build_sampGeo(finv_agg_d = finv_agg_true_d,write=False)
        

        #=======================================================================
        # sample
        #=======================================================================
 
        true_serx = self.build_rsamps(dkey='rsamps', finv_sg_d = finv_sg_true_d, 
                                      write=False, samp_method=samp_methodTrue, 
                                      idfn='id', **kwargs)        
        
        true_serx.index.set_names('id', level=2, inplace=True)
        #=======================================================================
        # group-----
        #=======================================================================
        #=======================================================================
        # join the gid keys
        #=======================================================================
        jdf = pd.DataFrame(index=mindex).reset_index(drop=False, level=1)
        
        true_serx1 = true_serx.to_frame().join(jdf, how='left').swaplevel(
            ).sort_index().set_index('gid', append=True).swaplevel()
 
 
        #group a series by two levels
        agg_serx = true_serx1.groupby(level=true_serx1.index.names[0:3]).mean().iloc[:,0]
        
        #=======================================================================
        # checks
        #=======================================================================
        #extremes (aggreegated should always be less extreme than raws)
        assert true_serx1.max()[0]>=agg_serx.max()
        assert true_serx1.min()[0]<=agg_serx.min()
        
        #dimensions
        assert len(true_serx1)>=len(agg_serx)
        
        
        #=======================================================================
        # wrap
        #=======================================================================
        log.info('finished on %i'%len(agg_serx))
        
 
        return agg_serx

    def sa_get(self,  # spatial tasks on each study area
                       proj_lib=None,
                       meth='get_rsamps',  # method to run
                       dkey=None,
                       write=True,
                       logger=None,
                       fargs_d=None, #args by studyArea
                       fkwargs=None, #kwargs to split by studyarea and put in function
                       **kwargs):
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger = self.logger
        log = logger.getChild('sa_get')
        
        if proj_lib is None: 
            proj_lib = self.proj_lib
        if not len(proj_lib)>0:
            raise Error('no proj_lib')
        
        log.info('on %i \n    %s' % (len(proj_lib), list(proj_lib.keys())))
        
        # assert dkey in ['rsamps', 'finv_agg_d'], 'bad dkey %s'%dkey
        
        if fkwargs is None:
            fkwargs = {sa:dict() for sa in proj_lib.keys()}
        
        if fargs_d is None:
            fargs_d = {sa:None for sa in proj_lib.keys()}
        #=======================================================================
        # loop and load
        #=======================================================================
        """pass Q handles?
            no.. these are automatically scraped from teh session already"""
        init_kwargs = {k:getattr(self,k) for k in [
            'prec', 'trim', 'out_dir', 'overwrite', 'tag', 'temp_dir',  
            #'logger', automatically scraped from the session
             ]}
        res_d = dict()
        for i, (name, pars_d) in enumerate(proj_lib.items()):
            log.info('%i/%i on %s' % (i + 1, len(proj_lib), name))
            
            with StudyArea(session=self, name=name, 
                           **pars_d, **init_kwargs) as wrkr:
                """
                wrkr.trim
                self.trim
                """
                # raw raster samples
                f = getattr(wrkr, meth)
                
                if fargs_d[name] is None:
                    res_d[name] = f(**{**fkwargs[name], **kwargs})
                else:
                    res_d[name] = f(*fargs_d[name], **{**fkwargs[name], **kwargs})
                
        #=======================================================================
        # write
        #=======================================================================
        if write:
            """frames between study areas dont have any relation to each other... keeping as dict"""
            raise Error('do we use this?')
            self.ofp_d[dkey] = self.write_pick(res_d, os.path.join(self.wrk_dir, '%s_%s.pickle' % (dkey, self.longname)), logger=log)
        
        return res_d
    
    def _get_sa(self, #retrieve a specific studyArea
                pars_d=None,
                       logger=None,
                       **kwargs):
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger = self.logger
        log = logger.getChild('_get_sa')
        
 
 
        #=======================================================================
        # loop and load
        #=======================================================================
        """pass Q handles?
            no.. these are automatically scraped from teh session already"""
        init_kwargs = {k:getattr(self,k) for k in [
            'prec', 'trim', 'out_dir', 'overwrite', 'tag', 'temp_dir',  
            #'logger', automatically scraped from the session
             ]}
        return StudyArea(session=self,**pars_d, **init_kwargs, **kwargs)  
    

    
    def _get_meta(self, #get a dictoinary of metadat for this model
                 ):
        
        d = super()._get_meta()
        
        attns = ['gcn', 'scale_cn']
        
        d = {**d, **{k:getattr(self, k) for k in attns}}
        
        #add project info
        for studyArea, lib in self.proj_lib.items():
            d[studyArea] = lib
            
        #add r un info
        d['date'] = self.start.strftime('%Y-%m-%d %H.%M.%S')
        d['runtime_mins'] = round((datetime.datetime.now() - self.start).total_seconds()/60.0, 3)
 
        
        return d

        
class StudyArea(Model, Qproj):  # spatial work on study areas
    
    finv_fnl = []  # allowable fieldnames for the finv

    def __init__(self,
                 
                 # pars_lib kwargs
                 EPSG=None,
                 finv_fp=None,
 
                 
                 #depth rasters
                 #wd_dir=None,
                 #wd_fp = None,
                 wse_fp_d = None, #{raster tag:fp}
                 dem_fp_d=None, 
                 #drlay_d = None,
                 
                 
                 aoi=None,
                 
                 # control
                 trim=None,
                 idfn='id',  # unique key for assets
                 ** kwargs):
        
        super().__init__(exit_summary=False, **kwargs)
        
        #=======================================================================
        # #set crs
        #=======================================================================
        crs = QgsCoordinateReferenceSystem('EPSG:%i' % EPSG)
        assert crs.isValid()
            
        self.qproj.setCrs(crs)
        
        #=======================================================================
        # simple attach
        #=======================================================================
        self.idfn = idfn
        self.trim=trim
        self.finv_fnl.append(idfn)
        
        #=======================================================================
        # directories
        #=======================================================================
        """todo: decide on a more consistent handling of directories for children"""
        self.temp_dir = os.path.join(self.temp_dir, self.name)
        if not os.path.exists(self.temp_dir): os.makedirs(self.temp_dir)
        
        
        #=======================================================================
        # load aoi
        #=======================================================================
        if not aoi is None and self.trim:
            self.load_aoi(aoi)
            
        #=======================================================================
        # load finv
        #=======================================================================
        if not finv_fp is None:
            finv_raw = self.vlay_load(finv_fp, dropZ=True, reproj=True)
            
            # field slice
            fnl = [f.name() for f in finv_raw.fields()]
            drop_l = list(set(fnl).difference(self.finv_fnl))
            if len(drop_l) > 0:
                finv1 = self.deletecolumn(finv_raw, drop_l)
                self.mstore.addMapLayer(finv_raw)
            else:
                finv1 = finv_raw
            
            # spatial slice
            if not self.aoi_vlay is None:
                finv2 = self.slice_aoi(finv1)
                self.mstore.addMapLayer(finv1)
            
            else:
                finv2 = finv1
            
            finv2.setName(finv_raw.name())
                
            # check
            miss_l = set(self.finv_fnl).symmetric_difference([f.name() for f in finv2.fields()])
            assert len(miss_l) == 0, 'unexpected fieldnames on \'%s\' :\n %s' % (miss_l, finv2.name())
            
            self.finv_vlay = finv2
      
        #=======================================================================
        # attachments
        #=======================================================================
        
        
        self.wse_fp_d=wse_fp_d
        self.dem_fp_d=dem_fp_d
        #self.drlay_d=drlay_d
        self.logger=self.session.logger.getChild(self.name)


        self.logger.info('StudyArea \'%s\' init' % (self.name))
 
    def get_clean_rasterName(self, raster_fn,
                             conv_lib={
                                'LMFRA':{
                                    'AG4_Fr_0050_dep_0116_cmp.tif':'LMFRA_0050yr',
                                    'AG4_Fr_0100_dep_0116_cmp.tif':'LMFRA_0100yr',
                                    'AG4_Fr_0500_dep_0116_cmp.tif':'LMFRA_0500yr',
                                    'AG4_Fr_1000_dep_0116_cmp.tif':'LMFRA_1000yr',
                                    
                                    'AG4_Fr_0050_dep_0116_cmpfnd.tif':'LMFRA_0050yr',
                                    'AG4_Fr_0100_dep_0116_cmpfnd.tif':'LMFRA_0100yr',
                                    'AG4_Fr_0500_dep_0116_cmpfnd.tif':'LMFRA_0500yr',
                                    'AG4_Fr_1000_dep_0116_cmpfnd.tif':'LMFRA_1000yr',
                                    
                                    'AG4_Fr_0500_WL_simu_0415_aoi09_0304.tif':'LM_0500',
                                    'AG4_Fr_0200_WL_simu_0415_aoi09_0304.tif':'LM_0200',
                                    'AG4_Fr_0100_WL_simu_0415_aoi09_0304.tif':'LM_0100'                                    
                                        },
                                
                                'obwb':{
                                    'depth_sB_0500_1218.tif':'obwb_0500yr',
                                    'depth_sB_0100_1218.tif':'obwb_0100yr',
                                    
                                    'depth_sB_0500_1218fnd.tif':'obwb_0500yr',
                                    'depth_sB_0100_1218fnd.tif':'obwb_0100yr',
                                    
                                    'wse_sB_0100_1218_10.tif':'obwb_0100',
                                    'wse_sB_0200_1218_10.tif':'obwb_0200',
                                    'wse_sB_0500_1218_10.tif':'obwb_0500'
                                    
                                    
                                    },
                                'Calgary':{
                                    'IBI_2017CoC_s0_0500_170729_dep_0116.tif':'Calgary_0500yr',
                                    'IBI_2017CoC_s0_0100_170729_dep_0116.tif':'Calgary_0100yr',
                                    
                                    'IBI_2017CoC_s0_0500_170729_dep_0116fnd.tif':'Calgary_0500yr',
                                    'IBI_2017CoC_s0_0100_170729_dep_0116fnd.tif':'Calgary_0100yr',
                                    
                                    'IBI_2017CoC_s0_0500_170729_aoi01_0304.tif':'Calgary_0500',
                                    'IBI_2017CoC_s0_0200_170729_aoi01_0304.tif':'Calgary_0200',
                                    'IBI_2017CoC_s0_0100_170729_aoi01_0304.tif':'Calgary_0100'
                                    },
    
                                        }   
                             ):
        
        rname = raster_fn.replace('.tif', '')
        if self.name in conv_lib:
            if raster_fn in conv_lib[self.name]:
                rname = conv_lib[self.name][raster_fn]
                
        return rname
    
    def get_finv_agg_d(self, #wrapper for executing multiple finv constructors
                       #data
                       finv_vlay_raw=None,
                       
                       #aggregation pars
                       aggType=None, aggLevel_l=None,
                       
                       #pars
                       get_lookup=False,
                       
                       #gen
                        idfn=None, logger=None,
                      **kwargs):
        
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger=self.logger
        log = logger.getChild('get_finv_agg_d')
        
        if finv_vlay_raw is None: finv_vlay_raw = self.finv_vlay
        if idfn is None: idfn = self.idfn
 
        
        kwargsi = dict(get_lookup=get_lookup, finv_vlay=finv_vlay_raw, idfn=idfn)
        #=======================================================================
        # loop and build each
        #=======================================================================
        fam_d, finv_d=dict(), dict()
        for i, aggLevel in enumerate(aggLevel_l):
            assert isinstance(aggLevel, int)
            log.info('building %i/%i w/ \'%s\'\n\n'%(i+1, len(aggLevel_l), aggType))
            temp_dir = os.path.join(self.temp_dir, 'get_finv_agg_d', str(i))
            if not os.path.exists(temp_dir):os.makedirs(temp_dir)
            #===================================================================
            # retrieve the function
            #===================================================================
            if aggType == 'none':  # see Test_p1_finv_none
                f = self.get_finv_clean
 
            elif aggType == 'gridded':  # see Test_p1_finv_gridded
                f = self.get_finv_gridPoly 
  
            elif aggType=='convexHulls':
                f = self.get_finv_convexHull 
 
            else:
                raise Error(aggType)
            
            fam_d[aggLevel], finv_d[aggLevel] = f(aggLevel=aggLevel, logger=logger.getChild(str(i)),
                                                  temp_dir=temp_dir, **kwargsi, **kwargs)
            
        #=======================================================================
        # wrap
        #=======================================================================
        faMap_dx = pd.concat(fam_d, axis=1)
        faMap_dx.columns = aggLevel_l
        
        log.info('finished on %i'%len(aggLevel_l))
        
        return {'faMap_dx':faMap_dx, 'finv_d':finv_d}
                     
    
    def get_finv_clean(self,
                       #data
                       finv_vlay=None,
                       
                       #pars
                       get_lookup=False,
                       
                       #gen
                       gcn=None, idfn=None, logger=None,
                      ):
        
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger=self.logger
        log = logger.getChild('get_finv_clean')
        
        if finv_vlay is None: finv_vlay = self.finv_vlay
        finv_vlay_raw = finv_vlay
        if idfn is None: idfn = self.idfn
        mstore = QgsMapLayerStore() 
        #=======================================================================
        # clean finv
        #=======================================================================
        """general pre-cleaning of the finv happens in __init__"""
        
        drop_fnl = set([f.name() for f in finv_vlay.fields()]).difference([idfn])
 
        if len(drop_fnl) > 0:
            vlay1 = self.deletecolumn(finv_vlay, list(drop_fnl), logger=log)
            mstore.addMapLayer(finv_vlay)  
        else:
            vlay1 = finv_vlay
            
        
            
 
        
        #=======================================================================
        # check
        #=======================================================================
        finv_vlay = vlay1
        
        fnl = [f.name() for f in finv_vlay.fields()]
        
        """relaxing this
        assert finv_vlay.wkbType()==3, 'got bad type on %s'%finv_vlay_raw.name()"""
        assert len(fnl) == 1
        assert idfn in fnl
        
        #check keys
        chk_ser = pd.Series(vlay_get_fdata(finv_vlay, idfn))
        
        bx = chk_ser.duplicated()
        if bx.any():
            raise Error('%s got %i/%i duplicated id vals:\n    %s'%(
                finv_vlay_raw.name(), bx.sum(), len(bx), chk_ser.loc[bx]))
        
        #=======================================================================
        # wrap
        #=======================================================================
        finv_vlay.setName('%s_clean' % finv_vlay_raw.name())
        log.debug('finished on %s' % finv_vlay.name())
        mstore.removeAllMapLayers()
        if not get_lookup:
            return finv_vlay
 
        #=======================================================================
        # #build a dummy lookup for consistency w/ get_finv_gridPoly
        #=======================================================================
        if gcn is None: gcn = self.gcn
        
        # rename indexer to match
        finv_vlay1 = self.renameField(finv_vlay, idfn, gcn, logger=log)
        self.mstore.addMapLayer(finv_vlay)
        finv_vlay1.setName(finv_vlay.name())
        
        df = vlay_get_fdf(finv_vlay)
        df[gcn] = df[idfn]
        
        
        finv_gkey_df = df.set_index(idfn)
        
        bx = finv_gkey_df.index.duplicated()
        if bx.any():
            raise Error('%s got %i/%i duplicated id vals:\n    %s'%(
                self.name, bx.sum(), len(bx), finv_gkey_df.loc[bx, :]))
        
        assert finv_gkey_df.index.is_unique
        assert finv_gkey_df[gcn].is_unique
        
        
        return finv_gkey_df, finv_vlay1
        
    def get_finv_gridPoly(self,  # get a set of polygon grid finvs (for each grid_size)
                  
                  aggLevel=10,
                  idfn=None,
 
                  overwrite=None,
                  finv_vlay=None,
                  temp_dir=None,
                  **kwargs):
        """
        
        how do we store an intermitten here?
            study area generates a single layer on local EPSG
            
            Session writes these to a directory

        need a meta table
        """
        #=======================================================================
        # defaults
        #=======================================================================
        log = self.logger.getChild('get_finvs_gridPoly')
        gcn = self.gcn
        if overwrite is None: overwrite = self.overwrite
        if temp_dir is None: temp_dir=self.temp_dir
        
        if idfn is None: idfn = self.idfn
        grid_size = aggLevel
        
        if finv_vlay is None: finv_vlay = self.get_finv_clean(idfn=idfn, **kwargs)
        
        #=======================================================================
        # get points on finv_vlay
        #=======================================================================
        """for clean grid membership.. just using centroids"""
        if not 'Point' in QgsWkbTypes().displayString(finv_vlay.wkbType()):
            finv_pts = self.centroids(finv_vlay, logger=log)
            self.mstore.addMapLayer(finv_pts)
        else:
            finv_pts = finv_vlay
        
        #===================================================================
        # raw grid
        #===================================================================
        log.info('on \'%s\' w/ grid_size=%i' % (finv_vlay.name(), grid_size))
        log.info('    creategrid w/ spacing=%i on  %i' %(grid_size, finv_vlay.dataProvider().featureCount()))
        gvlay1 = self.creategrid(finv_vlay, spacing=grid_size, logger=log)
        self.mstore.addMapLayer(gvlay1)
        
 
        """causing some inconsitent behavior
        #===================================================================
        # active grid cells only
        #===================================================================
 
        #select those w/ some assets
        gvlay1.removeSelection()
        self.createspatialindex(gvlay1)
        log.info('    selecting from grid based on intersect w/ \'%s\''%(finv_vlay.name()))
        self.selectbylocation(gvlay1, finv_vlay, logger=log)
        
        #save these
        gvlay2 = self.saveselectedfeatures(gvlay1, logger=log, output='TEMPORARY_OUTPUT')"""
        
        gvlay2 = gvlay1
        #===================================================================
        # populate/clean fields            
        #===================================================================
        """TODO: migrate to get_finv_links()"""
        # rename id field
        log.info('   renameField \'id\':\'%s\'' % gcn)
        gvlay3 = self.renameField(gvlay2, 'id', gcn, logger=log)
        self.mstore.addMapLayer(gvlay3)
        
        
        # delete grid dimension fields
        fnl = set([f.name() for f in gvlay3.fields()]).difference([gcn])
        gvlay3b = self.deletecolumn(gvlay3, list(fnl), logger=log)
        # self.mstore.addMapLayer(gvlay3b)
        
        # add the grid_size
        #=======================================================================
        # gvlay4 = self.fieldcalculator(gvlay3b, grid_size, fieldName='grid_size', 
        #                                fieldType='Integer', logger=log)
        #=======================================================================
        gvlay4 = gvlay3b
        
        #===================================================================
        # build refecrence dictionary to true assets
        #===================================================================
        #prep
        self.createspatialindex(finv_pts)
        self.createspatialindex(gvlay4)
        gvlay4.setName('%s_gridded'%finv_vlay.name())
        
        log.info('   joinattributesbylocation \'%s\' (%i) to \'%s\' (%i)'%(
            finv_pts.name(), finv_pts.dataProvider().featureCount(),
            gvlay4.name(), gvlay4.dataProvider().featureCount()))
        
        """very slow for big layers"""
        jd = self.joinattributesbylocation(finv_pts, gvlay4, jvlay_fnl=gcn,
                                           method=1, logger=log,
                                           # predicate='touches',
                 output_nom=os.path.join(temp_dir, 'finv_grid_noMatch_%i_%s.gpkg' % (
                                             grid_size, self.longname)))
        
        # check match
        #=======================================================================
        # noMatch_cnt = finv_vlay.dataProvider().featureCount() - jd['JOINED_COUNT']
        # if not noMatch_cnt == 0:
        #     """gid lookup wont work"""
        #     raise Error('for \'%s\' grid_size=%i failed to join  %i/%i assets... wrote non matcherse to \n    %s' % (
        #         self.name, grid_size, noMatch_cnt, finv_vlay.dataProvider().featureCount(), jd['NON_MATCHING']))
        #=======================================================================
                
        jvlay = jd['OUTPUT']
        self.mstore.addMapLayer(jvlay)
        
        df = vlay_get_fdf(jvlay, logger=log).set_index(idfn)
        
        #=======================================================================
        # clear non-matchers
        #=======================================================================
        """3rd time around with this one... I think this approach is cleanest though"""
        log.info('    cleaning \'%s\' w/ %i'%(gvlay4.name(), gvlay4.dataProvider().featureCount()))
        
        grid_df = vlay_get_fdf(gvlay4)
        
        bx = grid_df[gcn].isin(df[gcn].unique())  # just those matching the raws
        
        assert bx.any()
 
        gvlay4.removeSelection()
        gvlay4.selectByIds(grid_df[bx].index.tolist())
        assert gvlay4.selectedFeatureCount() == bx.sum()
        
        gvlay5 = self.saveselectedfeatures(gvlay4, logger=log)
        self.mstore.addMapLayer(gvlay4)

        assert len(df)>0
        #===================================================================
        # write
        #===================================================================
        
        gvlay5.setName('finv_gPoly_%i_%s' % (grid_size, self.longname.replace('_', '')))
        """moved onto the session
        if write_grids:
            self.vlay_write(gvlay4, os.path.join(od, gvlay4.name() + '.gpkg'),
                            logger=log)"""
 
        #===================================================================
        # #meta
        #===================================================================
        mcnt_ser = df[gcn].groupby(df[gcn]).count()
        meta_d = {
            'total_cells':gvlay1.dataProvider().featureCount(),
            'active_cells':gvlay5.dataProvider().featureCount(),
            'max_member_cnt':mcnt_ser.max()
            }
 
        #=======================================================================
        # wrap
        #=======================================================================
        
        log.info('finished on %i' % len(df))
 
        return df, gvlay5
    
    
    def get_finv_convexHull(self,
                  aggLevel=5, #number of members per group
                  idfn=None,
 
                  overwrite=None,
                  finv_vlay=None,
                  temp_dir=None,
                  **kwargs):
        """
 

        need a meta table
        
        called by StudyArea.get_finv_agg_d()
        """
        #=======================================================================
        # defaults
        #=======================================================================
        log = self.logger.getChild('get_finv_convexHull')
        gcn = self.gcn
        if overwrite is None: overwrite = self.overwrite
        if temp_dir is None: temp_dir=self.temp_dir
        
        if idfn is None: idfn = self.idfn
 
        
        if finv_vlay is None: finv_vlay = self.get_finv_clean(idfn=idfn, **kwargs)
        
        fcnt = finv_vlay.dataProvider().featureCount() 
 
        assert fcnt >aggLevel*2
        
        mstore = QgsMapLayerStore()
        
        #=======================================================================
        # raw aggregated geo-----------
        #=======================================================================
        if aggLevel==1:
            raise Error('dome')
        #=======================================================================
        # calc clusters
        #=======================================================================
        clusters = math.ceil(fcnt/aggLevel)
        
        log.info('on \'%s\' w/ %i feats and aggLevel=%i clusters=%i' % (
            finv_vlay.name(),fcnt, aggLevel, clusters))
        #=======================================================================
        # assign to clusters
        #=======================================================================
 
        """autmatically takes centroids"""
        vlay1 = self.kmeansclustering(finv_vlay, clusters, logger=log, fieldName=gcn)
        
        mstore.addMapLayer(vlay1)
        #=======================================================================
        # build convex hulls
        #=======================================================================
        ofp = self.minimumboundinggeometry(vlay1, logger=log, fieldName=gcn,
                                             output=os.path.join(self.temp_dir, '%s_convexHull%i_raw.gpkg'%(finv_vlay.name(), aggLevel)))
        
        vlay2 = self.fixgeo(ofp, logger=log)
        
        #vlay2 = self.get_layer(ofp, mstore=mstore)
                                             
                                             
        mstore.addMapLayer(vlay2)
        vlay2.setName('finv_cvh_raw_%s_%s' % (aggLevel, self.longname.replace('_', '')))
        #===================================================================
        # build refecrence dictionary to true assets----------
        #===================================================================
        finv_gkey_df, vlay3 = self.get_finv_links(vlay2,finv_raw_vlay = finv_vlay,
                            logger=log, idfn=idfn,
                            allow_miss=False, #should have no orphaned hulls... occasionally were still getting some
                            temp_dir=temp_dir)
        
        vlay4 = self.multiparttosingleparts(vlay3, logger=log)
        #=======================================================================
        # wrap
        #=======================================================================
        vlay4.setName('finv_cvh_%i_%s' % (aggLevel, self.longname.replace('_', '')))
        
        mstore.removeAllMapLayers()
        
        
        return finv_gkey_df, vlay4
            
        
        
        
    def get_finv_links(self, #get linking information between raw and aggregated finvs
                       finv_agg_vlay,
                       finv_raw_vlay=None,
                       idfn=None,
                       allow_miss=True, 
                       write=None,
                       temp_dir=None,
                  **kwargs):
        """
 

        need a meta table
        """
        #=======================================================================
        # defaults
        #=======================================================================
        log = self.logger.getChild('get_finvs_gridPoly')
        gcn = self.gcn
 
        if write is None: write=self.write
        if idfn is None: idfn = self.idfn
        if temp_dir is None: temp_dir=self.temp_dir
 
        
        if finv_raw_vlay is None: finv_raw_vlay = self.get_finv_clean(idfn=idfn, **kwargs)
        
        mstore = QgsMapLayerStore()
        #=======================================================================
        # prep layers--------
        #=======================================================================
        #=======================================================================
        # clean up fields on agg
        #=======================================================================
        #drop all those except the identifer
        raw_fnl = [f.name() for f in finv_agg_vlay.fields()]
        assert gcn in raw_fnl
        fnl = set(raw_fnl).difference([gcn])
        agg_vlay1 = self.deletecolumn(finv_agg_vlay, list(fnl), logger=log)
        
 
        
        agg_vlay1.setName(finv_agg_vlay.name())
        
        self.createspatialindex(agg_vlay1)
        

        log.info('building key references for  %i on %s'%(
            agg_vlay1.dataProvider().featureCount(), agg_vlay1.name()))
        """todo: combine this with finv_gridPoly"""
        
        
        #=======================================================================
        # get points on finv_vlay
        #=======================================================================
        if not 'Point' in QgsWkbTypes().displayString(finv_raw_vlay.wkbType()):
            cent_fp = self.centroids(finv_raw_vlay, logger=log,
                         output=os.path.join(temp_dir, '%s_pts.gpkg'%finv_raw_vlay.name()))
            f_vlay1 = self.get_layer(cent_fp, mstore=mstore)
 
            f_vlay1.setName('%s_pts'%finv_raw_vlay.name())
        else:
            f_vlay1 = finv_raw_vlay
            cent_fp = finv_raw_vlay.source()
        
        self.createspatialindex(f_vlay1)
        
        
        #=======================================================================
        # spatial join---------
        #=======================================================================
        afcnt = agg_vlay1.dataProvider().featureCount()
        log.info('   joinattributesbylocation \'%s\' (%i) to \'%s\' (%i)'%(
            f_vlay1.name(), f_vlay1.dataProvider().featureCount(),
            agg_vlay1.name(), afcnt))
        
        """very slow for big layers"""
        jd = self.joinattributesbylocation(f_vlay1, agg_vlay1, jvlay_fnl=gcn,
                                           method=1, logger=log,
                                           # predicate='touches',
                 output_nom=os.path.join(temp_dir, 'finv_noMatch_%i_%s.gpkg' % (
                                             afcnt, self.longname)))
        
        
        #retrieve
        jvlay = jd['OUTPUT']
        mstore.addMapLayer(jvlay)
        
        
        #=======================================================================
        # match result
        #=======================================================================
        #joined ids
        df = vlay_get_fdf(jvlay, logger=log).set_index(idfn).drop('fid', axis=1, errors='ignore')
        
        #aggregated ids
        agg_df = vlay_get_fdf(agg_vlay1)
        #agg_gids = pd.Series(vlay_get_fdata(agg_vlay1, gcn), name=gcn).sort_values().reset_index(drop=True)
        assert agg_df[gcn].is_unique
        
        set_d = set_info(df[gcn].values, agg_df[gcn].values, result='counts')
 
        
        #=======================================================================
        # handle misses
        #=======================================================================
        if not set_d['symmetric_difference']==0:
            """here we reduce the aggregated finv to the innner set of the spatial join"""

            #set_d1 = set_info(df[gcn].values, agg_gids.values)
            #=======================================================================
            # clear non-matchers
            #=======================================================================
            """5th time around with this one... I think this approach is cleanest though
            WARNING: dont mix up fid (used for selection) and gid (used for set comparison)
            """
            log.info('    cleaning \'%s\' w/ %i'%(agg_vlay1.name(), agg_vlay1.dataProvider().featureCount()))

            #id those with succesful matches 
            bx = agg_df[gcn].isin(df[gcn])
            assert bx.sum()==set_d['intersection']
 
     
            #select these
            agg_vlay1.removeSelection()
            agg_vlay1.selectByIds(agg_df.index[bx].tolist())
            assert agg_vlay1.selectedFeatureCount() == set_d['intersection']
            
            #save just those with some intersect
            agg_vlay2 = self.saveselectedfeatures(agg_vlay1, logger=log)
            mstore.addMapLayer(agg_vlay1)
            
            
            #save aggregated feats that failed to match
            if write:
                """ignores true feats missed by the aggregation"""
                agg_vlay1.invertSelection()
                assert agg_vlay1.selectedFeatureCount() == set_d['diff_right']
 
                aggMiss_fp = self.saveselectedfeatures(agg_vlay1, logger=log,
                            output=os.path.join(self.temp_dir, '%s_miss.gpkg'%finv_agg_vlay.name()))
                log.info('    wrote %i misses to %s'%(agg_vlay1.selectedFeatureCount(), aggMiss_fp))
                
                if not allow_miss:
                    aggRaw_fp = self.vlay_write(agg_vlay1, os.path.join(self.temp_dir, '%s.gpkg'%finv_agg_vlay.name()))
                    join_fp = self.vlay_write(jvlay, os.path.join(self.temp_dir, 'get_finv_links_joinattributesbylocation.gpkg'))
            
            assert allow_miss, 'got %i misses on %s w/ allow_miss=False'%(set_d['symmetric_difference'], agg_vlay1.name()) +\
            ' \n    %s\n    join_vlay:%s\n    NON_MATCHING: %s\n    Raw Points:%s\n    AggRaw:%s\n    AggMisses:%s'%(
                 set_d, join_fp, jd['NON_MATCHING'], cent_fp, aggRaw_fp, aggMiss_fp)
            
            
                        
        else:
            
            agg_vlay2 = agg_vlay1
            
 
        #=======================================================================
        # check
        #=======================================================================
        chk_ser = pd.Series(vlay_get_fdata(agg_vlay2, gcn), name=gcn).sort_values().reset_index(drop=True)
        assert chk_ser.is_unique
        
        set_d = set_info(df[gcn].values, chk_ser.values, result='counts')
        if not set_d['symmetric_difference']==0:
            """
            set_d1 = set_info(df[gcn].values, chk_ser.values)
            set_d1['diff_right']
            
            
            agg_vlay2.source()
            """

            raise Error('keys failed to match \n    %s'%set_d)
        
        assert df.index.name=='id'
        assert np.array_equal(df.columns, np.array([gcn]))
        #=======================================================================
        # wrap
        #=======================================================================
        
 
        mcnt_ser = df[gcn].groupby(df[gcn]).count()
        meta_d = {
            'total_agg_fcnt':agg_vlay1.dataProvider().featureCount(),
            'active_agg_fcnt':agg_vlay2.dataProvider().featureCount(),
            'max_member_cnt':mcnt_ser.max()
            }
        
        
        mstore.removeAllMapLayers()
        
        log.info('finished on %s w/ \n    %s'%(str(df.shape), meta_d))
        
        return df, agg_vlay2
    
    def get_tvalsR_area(self, #get raw tvals from polygon areas
                  #data                  

                  finv_vlay=None, 

                  
                  #gen
                  overwrite=None, idfn=None,                   
                  write=False, #just for debugging (will crash tests)
                  logger=None,
                  ):
        """
        weights the true (raw) tvals coming from raw footprints
            by the area of each footprint within an aggregated cell
            
        for grids:
            often part of a footprint falls somewhere without a grid cell
                (we clean by centroid intersect to maintain the 1:m relation of true:agg)
                more relevant for small grids
        """
 
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger=self.logger
        log = logger.getChild('get_tvals_aSplit')
 
        if overwrite is None: overwrite = self.overwrite
        
        if idfn is None: idfn = self.idfn
 
 
        rcoln= self.scale_cn + '_raw'
        
        mstore = QgsMapLayerStore()
        #=======================================================================
        # #retrieve
        #=======================================================================
        if finv_vlay is None: 
            finv_vlay = self.get_finv_clean(idfn=idfn, logger=log)
            
        log.info('on \' %s\' w/ %i feats'%(finv_vlay.name(), finv_vlay.dataProvider().featureCount()))
        
        #=======================================================================
        # get area from features
        #=======================================================================
        vlay1 = self.addgeometry(finv_vlay, logger=log)
        mstore.addMapLayer(vlay1)
        
        #retrieve
        df1 = vlay_get_fdf(vlay1)
 
        rser = df1.set_index(idfn)['area'].rename(rcoln)
        
        
        #=======================================================================
        # check
        #=======================================================================
        assert rser.min()>10
        assert rser.max()<1e6
        assert rser.notna().all()
        
        #=======================================================================
        # wrap4
        #=======================================================================
        mstore.removeAllMapLayers()
        
        log.debug('finished on %i'%len(rser))
        
        return rser.sort_index()
        
        
        
        
            
            
        
        
    def get_tvals_aSplit(self,
                  #data                  
                  tvals_raw_serx=None,
                  finv_vlay=None, 
                  finv_agg_d=None,
                  
                  #gen
                  overwrite=None, idfn=None,                   
                  write=False, #just for debugging (will crash tests)
                  logger=None,
                  ):
        """
        weights the true (raw) tvals coming from raw footprints
            by the area of each footprint within an aggregated cell
            
        for grids:
            often part of a footprint falls somewhere without a grid cell
                (we clean by centroid intersect to maintain the 1:m relation of true:agg)
                more relevant for small grids
        """
 
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger=self.logger
        log = logger.getChild('get_tvals_aSplit')
        gcn = self.gcn
        if overwrite is None: overwrite = self.overwrite
        
        if idfn is None: idfn = self.idfn
        gcn = self.gcn
 
        rcoln= self.scale_cn
        #=======================================================================
        # #retrieve
        #=======================================================================
        if finv_vlay is None: 
            finv_vlay = self.get_finv_clean(idfn=idfn)
        
        #aggregated finv for this studyArea
        finv_agg_vlay = finv_agg_d[self.name]
        
        #tvals generated for this study area (double index.. no area weighting)
        """todo: fix sa_get to handle passthrough kwargs to each child"""
        tvals_raw_serx = tvals_raw_serx.loc[idx[self.name, :]]
        
        if len(tvals_raw_serx.index.names)==3:
            tvals_raw_serx = tvals_raw_serx.droplevel(0)
        
        fcnt = finv_vlay.dataProvider().featureCount() 
 
        mstore = QgsMapLayerStore()
        
        log.info('on \n    %s (%i) and  %s (%i)'%(
            finv_vlay.name(), finv_vlay.dataProvider().featureCount(),
            finv_agg_vlay.name(), finv_agg_vlay.dataProvider().featureCount()))
        
        #=======================================================================
        # check keys
        #=======================================================================
        #true keys
        finv_df = vlay_get_fdf(finv_vlay)
        assert idfn in finv_df.columns
        
        
        set_d = set_info(finv_df[idfn], tvals_raw_serx.index.unique(idfn), result='counts')
        assert set_d['symmetric_difference']==0
        
        #aggregated keys
        agg_df = vlay_get_fdf(finv_agg_vlay)
        assert gcn in agg_df.columns
        
        assert gcn in tvals_raw_serx.index.names
        
        set_d = set_info(agg_df[gcn], tvals_raw_serx.index.unique(gcn), result='counts')
        assert set_d['symmetric_difference']==0, 'mismatch between finv_agg_vlay and tvals_raw_serx \n    %s'%set_d
        
        #=======================================================================
        # get raw areas
        #=======================================================================
        fvlay1 = self.addgeometry(finv_vlay, logger=log)
        mstore.addMapLayer(fvlay1)
        
        #rename the area field
        fvlay2 = self.renameField(fvlay1, 'area', 'area_raw', logger=log)
        mstore.addMapLayer(fvlay2)
        
        #=======================================================================
        # split on aggregated boundarires
        #=======================================================================
        #convert agggretaged polys to lines
        agg_lines = self.polygonstolines(finv_agg_vlay, logger=log)
        
        #split
        fvlay3 = self.splitwithlines(fvlay2, agg_lines, logger=log)
        
        #=======================================================================
        # calc area ratios
        #=======================================================================
        #add the new geometries
        fvlay4 = self.addgeometry(fvlay3, logger=log)
        
        #extract
        df1 = vlay_get_fdf(fvlay4, logger=log).drop(['perimeter', 'perimeter_2'], axis=1)
        
        df1['areaRatio'] = df1['area']/df1['area_raw']
        
        bx = df1['areaRatio']<=1.0
        if not bx.all():
            """some rounding error?"""
            log.debug(df1.loc[~bx, :])
            log.warning('got %i/%i areaRatios exceeding 1.0...forcing these to 1.0'%(np.invert(bx).sum(), len(bx)))
            df1.loc[~bx, 'areaRatio'] = 0.999
        
        #=======================================================================
        # weight tvals
        #=======================================================================
        """need to add keys to link to aggregated"""
        #join the raw tvals by id
        df2 = df1.join(tvals_raw_serx.reset_index(drop=False).set_index(idfn), on=idfn)
        df2['asset_count'] = 1
        #scale
        df2['tvals_weighted'] = df2[tvals_raw_serx.name]*df2['areaRatio']
        
        #total
        df3 = df2.groupby(gcn).sum().loc[:, ['area_raw', 'area', 'areaRatio', 'asset_count', 'tvals_weighted']]
        
        
        #=======================================================================
        # write layer
        #=======================================================================
        if write:
            """only for debugging... if we are iterating this will write 1 layer per iteration"""
            geo_d = vlay_get_geo(finv_agg_vlay, logger=log)
            
 
            res_vlay = self.vlay_new_df(agg_df.join(df3, on=gcn), geo_d=geo_d, logger=log,
                                        layname='tvals_aSplit')
            out_dir = os.path.join(self.temp_dir, 'get_tvals_aSplit')
            if not os.path.exists(out_dir): os.makedirs(out_dir)
            ofp = os.path.join(out_dir, '%s_tvals_aSplit_%s.gpkg'%(finv_agg_vlay.name(), self.tag))
            self.vlay_write(res_vlay,ofp, logger=log)
        
        #=======================================================================
        # wrap
        #=======================================================================
        tval_ser = df3['tvals_weighted'].rename(rcoln)
        assert np.array_equal(tval_ser.index, tvals_raw_serx.index.unique(0))
        
        stats_d = {stat:'%.3f'%getattr(tval_ser, stat)() for stat in ['min',  'mean', 'max']}
        log.info('finished on %i w/ \n    %s'%(len(tval_ser), stats_d))
        
        
        return tval_ser


    def get_drlay(self, #build a depth layer intelligently

                   #raster selection
                   wse_rlay=None, dem_rlay=None,
                   wse_fp_d=None,dem_fp_d=None,
                   severity='hi',  #which wse rastser to select
                   #dem_res=5,
                   
                   #raster downsampling
                   dsampStage='none', #which stage of the depth raster calculation to apply the downsampling
                        #none: no downSampling happening
                        #pre: resample both rasters before subtraction  
                        #preGW:same as 'pre', but with a groundwater filter
                        #post: subtract rasters first, then resample the result
                        #postFN: same as post, but null values are filled with zeros
                   downSampling='none',

                  resolution=None, #0=raw (nicer for variable consistency)
                  base_resolution=None, #resolution of raw data
                   
                   #gen 
                  logger=None, layerName=None,
                  trim=None, #generally just trimming this by default
                   ):
        
        """separate function for 'severity' and 'resolution' (gdalwarp)
 
        """
           
        #=======================================================================
        # defaults
        #=======================================================================
        if wse_fp_d is None: wse_fp_d = self.wse_fp_d
        if dem_fp_d is None: dem_fp_d=self.dem_fp_d
        if logger is None: logger = self.logger
        if trim is None: trim=self.trim
        temp_dir = os.path.join(self.temp_dir, 'get_drlay')
        if not os.path.exists(temp_dir): os.makedirs(temp_dir)
        start = datetime.datetime.now()
        log = logger.getChild('get_raster')
        
        #resolutions
        if base_resolution is None:
            from definitions import base_resolution
        if resolution is None: resolution = base_resolution
        resolution = int(resolution)
        
        """TODO: get this to return somewhere"""
        meta_d = {'resolution':resolution, 'dsampStage':dsampStage, 'downSampling':downSampling}
        mstore=QgsMapLayerStore()
        #=======================================================================
        # parameter checks
        #=======================================================================
 
        assert dsampStage in ['none', 'post', 'pre', 'preGW', 'postFN'], dsampStage
        #parameter logic
        if dsampStage =='none':
            assert resolution==base_resolution, resolution
            assert downSampling=='none'
            
        if downSampling =='none':
            assert dsampStage == 'none', 'for downSampling=none expects dsampStage=none'
            
            
        assert resolution>=base_resolution
        assert isinstance(resolution, int)
        #=======================================================================
        # #select raster filepaths
        #=======================================================================
        def glay(fp):
            return self.get_layer(fp, mstore=mstore, logger=log)
        #WSE
        if wse_rlay is None:
            assert severity in wse_fp_d
            wse_raw_fp = wse_fp_d[severity]
            assert os.path.exists(wse_raw_fp)
            wse_rlay=glay(wse_raw_fp)
        
        #DEM
        """ starting from the same base for consistency
        if resolution in dem_fp_d:
            dem_raw_fp = dem_fp_d[resolution]
        else:"""
        if dem_rlay is None:
            dem_raw_fp = dem_fp_d[base_resolution] #just take the highest resolution
            assert os.path.exists(dem_raw_fp), dem_raw_fp
            dem_rlay=glay(dem_raw_fp)
 
        #get names
        baseName = wse_rlay.name()
        if layerName is None: 
            layerName = baseName + '_%03i_%s_dep' %(resolution, dsampStage)
 
        log.info('on %s w/ %s'%(baseName ,meta_d))
        #=======================================================================
        # trim
        #=======================================================================
        if trim:
            extents_layer = self.aoi_vlay
 
        else:
            extents_layer = wse_rlay
            
        extents = self.layerextent(extents_layer, 
                                   precision=0.0, #adding this buffer causes some problems with the tests
                                   ).extent()
 
        #=======================================================================
        # check raw resolutions
        #=======================================================================
        """pretty slow"""
        assert self.rlay_get_resolution(dem_rlay)==float(base_resolution)
        #assert self.rlay_get_resolution(dem_raw_fp)==float(base_resolution)
        """I guess this is not true?
        doesnt really matter as we are warping
        assert_rlay_equal(dem_rlay, wse_rlay, msg='get_drlay')"""
        #=======================================================================
        # helper funcs
        #=======================================================================
        def get_resamp(fp):
            return self.get_resamp(fp, resolution, downSampling,  extents=extents, logger=log)
        
        def get_warp(fp):
            if isinstance(fp, QgsRasterLayer):
                fname='%s_preWarp.tif'%fp.name()
            elif isinstance(fp, str):
                fname='%s_preWarp.tif'%os.path.basename(fp).replace('.tif', '')
            return self.warpreproject(fp, compression='none', extents=extents, logger=log,
                                        resolution=base_resolution,
                                        output=os.path.join(temp_dir, fname))
 
        #=======================================================================
        # preCalc -----------
        #=======================================================================
        if dsampStage in ['none', 'post', 'postFN']:
            """easier and cleaner to always start with the same warp"""
            log.info('warpreproject w/ resolution=%i to %s'%(base_resolution, extents))
            wse_fp = get_warp(wse_rlay)
            dem_fp = get_warp(dem_rlay)
 
        elif dsampStage == 'pre':
            log.info('downSampling w/ dsampStage=%s'%dsampStage)
            wse_fp = get_resamp(wse_rlay)
            dem_fp = get_resamp(dem_rlay)
            
        elif dsampStage == 'preGW':
            log.info('downSampling w/ dsampStage=%s'%dsampStage)
            wse1_fp = get_resamp(wse_rlay)
            dem_fp = get_resamp(dem_rlay)
            
            #===================================================================
            # remove groundwater from wse
            #===================================================================
            with HQproj(dem_fp=dem_fp, out_dir=temp_dir, crs=None,base_resolution=resolution,
                overwrite=True, session=self, logger=log) as wrkr:
                wse1_rlay = wrkr.get_layer(wse1_fp, mstore=wrkr.mstore)
                wse2_rlay = wrkr.wse_remove_gw(wse1_rlay)
                wse_fp = wse2_rlay.source()
        else:
            raise Error('badd dsampStage: %s'%dsampStage)
        
        wse1_rlay = glay(wse_fp)
        dem1_rlay = glay(dem_fp)
        assert_rlay_equal(wse1_rlay,dem1_rlay , msg='dem and wse dont match')
 
        #=======================================================================
        # subtraction--------
        #=======================================================================
        log.debug('building RasterCalc')
        with RasterCalc(wse1_rlay, name='dep', session=self, 
                        logger=log,out_dir=self.temp_dir,) as wrkr:
 
            #===================================================================
            # setup
            #===================================================================
            entries_d = {k:wrkr._rCalcEntry(v) for k,v in {'top':wse1_rlay, 'bottom':dem1_rlay}.items()}
            formula = '%s - %s'%(entries_d['top'].ref, entries_d['bottom'].ref)
            
            #===================================================================
            # execute subtraction
            #===================================================================
            log.debug('executing %s'%formula)
            dep_fp1 = wrkr.rcalc(formula, layname=layerName)

 


        #=======================================================================
        # fill nulls
        #=======================================================================
        """wse layer may have some nulls (just from rounded extents)
        these propagate as nulls into the delta calc... here we just set the depth to zero
        in these cells

        """
        #nodcnt = hp.gdal.getNoDataCount(dep_fp1)
        
        """NO! need to preserve no datas to get the accurate averaging calc
        if nodcnt>0:
        
            log.info('filling %i/%i noDataCells w/ 0.0'%(nodcnt, 
                                                         self.rlay_get_cellCnt(dep_fp1, exclude_nulls=False)))
        
            dep_fp2 = self.fillnodata(dep_fp1, fval=0, logger=log, 
                            output = os.path.join(self.temp_dir, os.path.basename(dep_fp1).replace('.tif', '_fillna.tif')))
            
        else:
            dep_fp2 = dep_fp1"""
        dep_fp2 = dep_fp1
        #=======================================================================
        # post-downsample---------
        #=======================================================================
        if dsampStage =='post':
            """these have nulls  which affect the resampling (see postFN)
            note the resultant (low-res) depth raster will be zero-filled"""
            log.info('downSampling w/ dsampStage=%s'%dsampStage)
            dep_fp3 = get_resamp(dep_fp2)
        
        elif dsampStage =='postFN':
            #fill nulls
            """here we are filling the intermittent (hi-res) depth layer prior to subtraction
            this makes the subsequent post zero-filling (done for all methods) somewhat redundant
            but substantially affects the resampling
            """ 
            dep_fp3a = self.fillnodata(dep_fp2, fval=0, logger=log, 
                            output = os.path.join(self.temp_dir, os.path.basename(dep_fp2).replace('.tif', '_fillna.tif')))
            
            dep_fp3 = get_resamp(dep_fp3a)

        else:
            dep_fp3 = dep_fp2
            
        #=======================================================================
        # post null filling
        #=======================================================================
        null_cnt = hp.gdal.getNoDataCount(dep_fp3)
        """having nulls on the depth values breaks some of the raster stat calculators (esp. mean)
        dont' confuse our treatment of nulls on WSE (preserving) with depths (forcing to zero)
            except for dsampStage=pre... where we do preserve the nulls"""
        dep_fp4 = self.fillnodata(dep_fp3, fval=0.0, logger=log)
            
        #=======================================================================
        # check
        #=======================================================================
        rlay = self.rlay_load(dep_fp4,logger=log)
 
        #stats_d = self.get_rasterstats(rlay)
        #assert stats_d['resolution'] == resolution
        assert rlay.crs()==self.qproj.crs()
        assert int(self.rlay_get_resolution(rlay))==resolution

        #=======================================================================
        # wrap
        #=======================================================================
        rlay.setName(layerName)
        mstore.removeAllMapLayers()
        tdelta = datetime.datetime.now() - start
        
        log.info('finished in %s on \'%s\' (%i x %i = %i)'%(tdelta,
            rlay.name(), rlay.width(), rlay.height(), rlay.width()*rlay.height()))

        return {'rlay':rlay, 'noData_cnt':null_cnt, 'wse_fp':wse_fp, 'dem_fp':dem_fp}
    
    def get_resamp(self, #wrapper for  warpreproject
                   fp_raw, resolution, downSampling,  
                extents=None,
                
                #nodata filling
                fval=None,
                
                #output defaults
                out_dir=None,output=None,logger=None,):
        
    
        
        assert not resolution == 0
        
        if isinstance(fp_raw, str):
            basename = os.path.basename(fp_raw).replace('.tif', '') + '_warp%i.tif' % resolution
        elif isinstance(fp_raw, QgsRasterLayer):
            basename = fp_raw.name() + '_warp%03i.tif' % resolution
        else:
            raise IOError(fp_raw)
        
        if out_dir is None: out_dir=self.temp_dir
        if output is None:
            output=os.path.join(out_dir, basename)
        #===================================================================
        # defaults
        #===================================================================
        if logger is None: logger=self.logger
        log = logger.getChild('get_resamp')
        
        log.info('downsampling \'%s\' w/ resolution=%i and downSampling=%s' % (
            basename, resolution, downSampling))

        #===================================================================
        # execute
        #===================================================================
 
        wd_fp1 = self.warpreproject(
            fp_raw, output=output, 
            resolution=resolution, resampling=downSampling, 
            compression='none', crsOut=self.qproj.crs(), extents=extents, 
            logger=log)
        
        #=======================================================================
        # fill no data
        #=======================================================================
        if not fval is None:
            ndcnt = hp.gdal.getNoDataCount(wd_fp1) 
            if ndcnt>0:
                log.warning("got %i noData cells.. filling w/ %.2f"%(ndcnt, fval))
                            
                wd_fp2 = self.fillnodata(wd_fp1, fval=fval, logger=log,
                                          output=os.path.join(out_dir, os.path.basename(wd_fp1).replace('.tif', '') + '_fnd.tif'))
            else:
                wd_fp2=wd_fp1
        else:
            wd_fp2=wd_fp1
            
            
        return wd_fp2 
    
    
    def get_rsamps_d(self, #wrapper for executing multiple rsamps
                       #data
                       finv_agg_lib=None, drlay_lib=None,
                       
                       #defaults
                          logger=None,out_dir=None,
                      **kwargs):
        
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger=self.logger
        log = logger.getChild('get_rsamps_d')
        if out_dir is None:out_dir = self.temp_dir
        
        #=======================================================================
        # extract
        #=======================================================================
        finv_agg_d = self._from_lib(finv_agg_lib)
        drlay_d = self._from_lib(drlay_lib)
        
 
        #=======================================================================
        # loop and build each
        #=======================================================================
        log.info('\n\n%s sampling %i x %i = %i\n\n'%(self.name, 
            len(finv_agg_d), len(drlay_d), len(finv_agg_d)*len(drlay_d)))
            
        res_lib = {k:dict() for k in finv_agg_d.keys()}
        for aggLevel, finv_vlay in finv_agg_d.items():
 
            for resolution, drlay in drlay_d.items():
                k = '%s.%s'%(aggLevel, resolution)
                log.debug('%s for %s X %s'%(k, finv_vlay.name(), drlay.name()))
                
                res_lib[aggLevel][resolution] = self.get_rsamps(
                    finv_sg_d={self.name:finv_vlay},
                    drlay_d={self.name:drlay},
                    idfn=self.gcn, #not sure why
                    out_dir =os.path.join( out_dir, k),logger=log.getChild(k),
                    **kwargs).iloc[:, 0]
            
 
 
        #=======================================================================
        # wrap
        #=======================================================================
        d = {k:pd.concat(d, axis=1) for k,d in res_lib.items()}
        rdx = pd.concat(d, axis=0, names=['aggLevel', self.gcn])
        
        """
        view(rdx)
        """
        
        log.info('finished on %s'%str(rdx.shape))
        
        return rdx
                   

    def get_rsamps(self,  # sample a raster with a finv
                   
                   finv_sg_d=None, drlay_d=None, #{name:layer}
                   idfn=None,
                   logger=None,
                   
                   samp_method='points',
                   zonal_stat='Mean',  # stats to use for zonal. 2=mean
                   prec=None, out_dir=None,
                   #**kwargs,
                   ):
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger = self.logger
        log = logger.getChild('get_rsamps')
        
        # if finv_vlay_raw is None: finv_vlay_raw=self.finv_vlay
        if idfn is None: idfn = self.idfn
        if prec is None: prec = self.prec
        if out_dir is None: out_dir=os.path.join(self.temp_dir, 'get_rsamps')
        if not os.path.exists(out_dir): os.makedirs(out_dir) 
        #=======================================================================
        # precheck finv
        #=======================================================================
        
        finv_vlay_raw = finv_sg_d[self.name]
        
        
        if samp_method == 'points':
            assert 'Point' in QgsWkbTypes().displayString(finv_vlay_raw.wkbType())
        elif samp_method == 'zonal':
            assert 'Polygon' in QgsWkbTypes().displayString(finv_vlay_raw.wkbType())

        else:raise Error('bad key')
            
        assert idfn in [f.name() for f in finv_vlay_raw.fields()], 'missing \'%s\' in %s'%(idfn, finv_vlay_raw.name())
        #=======================================================================
        # clean finv
        #=======================================================================
        """general pre-cleaning of the finv happens in __init__"""
        
        drop_fnl = set([f.name() for f in finv_vlay_raw.fields()]).difference([idfn])
 
        if len(drop_fnl) > 0:
            finv_vlay = self.deletecolumn(finv_vlay_raw, list(drop_fnl), logger=log)
            self.mstore.addMapLayer(finv_vlay)  # keep the raw alive
            finv_vlay.setName(finv_vlay_raw.name()+'_dc')
        else:
            finv_vlay = finv_vlay_raw
            
        assert [f.name() for f in finv_vlay.fields()] == [idfn]
        
        #=======================================================================
        # prechekc depth raster
        #=======================================================================
        rlay = drlay_d[self.name]
        assert isinstance(rlay, QgsRasterLayer)
        
        rname = rlay.name()
        
        #=======================================================================
        #sample--------
        #=======================================================================
 
        #===================================================================
        # sample
        #===================================================================
        """note.. these are saved as temporary layres for debugging
        they dont correspond exactly tot he outputs"""
        ofp = os.path.join(out_dir, finv_vlay_raw.name() +'_rsamps.gpkg')
        if samp_method == 'points':
            vlay_samps = self.rastersampling(finv_vlay, rlay, logger=log, pfx='samp_',
                                             output=ofp)
        
        elif samp_method == 'zonal':
            vlay_samps = self.zonalstatistics(finv_vlay, rlay, logger=log, pfx='samp_', 
                                              stat=zonal_stat, output=ofp)
        else:
            raise Error('not impleented')
        
        #===================================================================
        # post           
        #===================================================================
        #retrieve data
        vlay_samps = self.get_layer(vlay_samps)
        self.mstore.addMapLayer(vlay_samps)
        
        df = vlay_get_fdf(vlay_samps, logger=log).drop('fid', axis=1)
        
        # change resulting sample column name        
        colbx = df.columns.str.startswith('samp_')
        assert colbx.sum()==1, 'non-unique match on sample prefixed column'      
        df = df.rename(columns={df.columns[colbx][0]:rname})
        
        # force type
        assert idfn in df.columns, 'missing key \'%s\' on %s' % (idfn, finv_vlay.name())
        df.loc[:, idfn] = df[idfn].astype(np.int64)
        
        assert df[idfn].is_unique, finv_vlay.name()
 
        df = df.set_index(idfn).sort_index()
        
        assert len(df.columns)==1
        #=======================================================================
        # fill zeros
        #=======================================================================
        res_df = df.fillna(0.0).round(prec)
        #=======================================================================
        # wrap
        #=======================================================================
        
        assert res_df.index.is_unique
        
        log.info('finished on %s and %i rasters w/ %i/%i dry' % (
            finv_vlay.name(), 1, res_df.isna().sum().sum(), res_df.size))
        
        return res_df #always 1 column
    
    def _from_lib(self, lib_raw): #extract yoru data froma lib
        return {k:lay for k,d in lib_raw.items() for sa,lay in d.items() if sa==self.name }
 
    
class ModelStoch(Model):
    def __init__(self,
                 modelID = 0, #unique model ID for the catalog
                 iters=10, #number of stochastic iterations
                 **kwargs):
        
        super().__init__(**kwargs)
        
        self.modelID=modelID
        self.iters=iters
        
 
        
    def write_lib(self, #writing pickle w/ metadata
                  lib_dir = None, #library directory
                  mindex = None,
                  overwrite=None,
                  modelID=None,
                  cat_d = {}, #passthrough atts to write in the index
                  dkey='tloss',
                  ):
        #=======================================================================
        # defaults
        #=======================================================================
        log = self.logger.getChild('write_lib')
        if overwrite is None: overwrite=self.overwrite
        if lib_dir is None:
            lib_dir = os.path.join(self.work_dir, 'lib', self.name)
 
        assert os.path.exists(lib_dir), lib_dir
        if modelID is None: modelID=self.modelID
        
        modTag = '%03i_%s'%(modelID, self.longname)
        #=======================================================================
        # setup filepaths4
        #=======================================================================
        catalog_fp = os.path.join(lib_dir, 'model_run_index.csv')
        vlay_dir = os.path.join(lib_dir, 'vlays', modTag)
        rlay_dir = os.path.join(lib_dir, 'rlays', modTag)
        
        #clear any existing library methods
        """needs to be before all the writes"""
        if os.path.exists(catalog_fp):
            with Catalog(catalog_fp) as wrkr:
                wrkr.remove_model(modelID)
                
        #=======================================================================
        # retrieve
        #=======================================================================
        tl_dx = self.retrieve(dkey) #best to call this before finv_agg_mindex
        if mindex is None: mindex = self.retrieve('finv_agg_mindex')
        
        
        
        
        """no! this is disaggrigation which is ambigious
        best to levave this for the analysis phase (and just pass the index)
        self.reindex_to_raw(tl_dx)"""
        #=======================================================================
        # write to csv
        #=======================================================================
        """not part of the library
        out_fp = os.path.join(self.out_dir, '%s_tloss.csv'%self.longname)
        tl_dx.to_csv(out_fp)"""

        
        #=======================================================================
        # write vectors
        #=======================================================================
        """here we copy each aggregated vector layer into a special directory in the libary
        these filepaths are then stored in teh model pickle"""
        #setup
        if not os.path.exists(vlay_dir):os.makedirs(vlay_dir)
        
        #retrieve
        finv_agg_d = self.retrieve('finv_agg_d')
        
        #write each layer into the directory
        ofp_d = self.store_layer_d(finv_agg_d, 'finv_agg_d', out_dir=vlay_dir, logger=log, write_pick=False)
        
        #=======================================================================
        # write depth rasters
        #=======================================================================
        drlay_d = self.retrieve('drlay_d')
        if not os.path.exists(rlay_dir):os.makedirs(rlay_dir)
        drlay_ofp_d = self.store_layer_d(drlay_d, 'drlay_d', out_dir=rlay_dir, logger=log, write_pick=False)
        
        #=======================================================================
        # build meta
        #=======================================================================
        meta_d = self._get_meta()
        res_meta_d = serx_smry(tl_dx.loc[:, idx[dkey, :]].mean(axis=1).rename(dkey))
        
        meta_d = {**meta_d, **res_meta_d}
        #=======================================================================
        # add to library
        #=======================================================================
        out_fp = os.path.join(lib_dir, '%s.pickle'%modTag)
        
        meta_d = {**meta_d, **{'pick_fp':out_fp, 'vlay_dir':vlay_dir, 'rlay_dir':rlay_dir}}
        
        d = {'name':self.name, 'tag':self.tag,  
             'meta_d':meta_d, 'tloss':tl_dx, 'finv_agg_mindex':mindex, 
             'finv_agg_d':ofp_d, 'vlay_dir':vlay_dir,
             'drlay_d':drlay_ofp_d, 'rlay_dir':rlay_dir}
        
        self.write_pick(d, out_fp, overwrite=overwrite, logger=log)
        
        
        
        #=======================================================================
        # update catalog
        #=======================================================================
        #kwargs from meta
        cat_d.update({k:meta_d[k] for k in [
            'modelID', 'name', 'tag', 'date', 'pick_fp', 'vlay_dir','rlay_dir', 'runtime_mins', 'out_dir', 'iters',]})
        
 
        cat_d = {**cat_d, **res_meta_d}
        cat_d['pick_keys'] = str(list(d.keys()))
        
        with Catalog(catalog_fp) as wrkr:
            wrkr.add_entry(cat_d)
        
            
        
        #=======================================================================
        # write 
        #=======================================================================
 
        
        log.info('updated catalog %s'%catalog_fp)
        
        return catalog_fp
    
    def build_tvals_raw(self, #stochastic calculation of tvals
                    dkey='tvals',
                    mindex=None, 
                    #finv_agg_d=None,
                    tval_type='rand',  # type for total values
                    iters=None, write=None,
                    logger=None,
                    **kwargs): 
        
        #=======================================================================
        # defaults
        #=======================================================================
        if write is None: write=self.write
        if logger is None: logger=self.logger
        log=logger.getChild('build_tvals_rawS')
        

        if iters is None: iters=self.iters
        
        assert dkey=='tvals_raw'
        
        #=======================================================================
        # retrieve
        #=======================================================================
        if mindex is None: 
            mindex = self.retrieve('finv_agg_mindex')  # studyArea, id : corresponding gid
        
        #if finv_agg_d is None: finv_agg_d = self.retrieve('finv_agg_d')
        
        #=======================================================================
        # setup pars
        #=======================================================================
        if tval_type == 'uniform':
            iters = 1
 
        #=======================================================================
        # execute children
        #=======================================================================
        log.info('on %i w/ iters=%i'%(len(mindex), iters))
        res_d = self.model_retrieve(dkey, tval_type=tval_type,
                               mindex=mindex,iters=iters,
                               #finv_agg_d=finv_agg_d, 
                               logger=log, **kwargs)
        
        #=======================================================================
        # assemble and collapse
        #=======================================================================

        dxind = pd.concat(res_d, axis=1)
        
        dx = pd.concat({dkey:dxind},axis=1, names=['dkey', 'iter'])
        
        #=======================================================================
        # write lyaer 
        #=======================================================================
        """todo?"""
 

        
        #=======================================================================
        # write pick
        #=======================================================================
        log.info('finished w/ %s'%str(dx.shape))
        if write:
            self.ofp_d[dkey] = self.write_pick(dx,
                                   os.path.join(self.wrk_dir, '%s_%s.pickle' % (dkey, self.longname)),
                                   logger=log)
            
        return dx
        
    def build_tvals(self, #stochastic calculation of tvals
                    dkey='tvals',
                    
                    #data
                    #mindex=None, 
                    finv_agg_d=None,
                    tvals_raw_serx = None,
                    
                    #parameters
                    dscale_meth=None,
                    
                    #pars (default)
                    iters=None, write=None,logger=None,
                    **kwargs): 
        
        #=======================================================================
        # defaults
        #=======================================================================
        if write is None: write=self.write
        if logger is None: logger=self.logger
        log=logger.getChild('build_tvalsS')

        assert dkey=='tvals'
        if iters is None: iters=self.iters
        #=======================================================================
        # retrieve
        #=======================================================================
        if finv_agg_d is None: 
            finv_agg_d = self.retrieve('finv_agg_d')
            
            
        #=======================================================================
        # if mindex is None: 
        #     mindex = self.retrieve('finv_agg_mindex')  # studyArea, id : corresponding gid
        #     
        #=======================================================================
        if tvals_raw_serx is None:
            tvals_raw_serx = self.retrieve('tvals_raw')
            
        assert len(tvals_raw_serx.columns)==iters
        assert np.array_equal(tvals_raw_serx.columns.unique('iter'), np.array(range(iters)))
        
        #split per iter
        """beacuse each call of Model.build_tvals needs its respective tvals_raw"""
        d = dict()
        for i, gdx in tvals_raw_serx.groupby(level='iter', axis=1):
            assert len(gdx.columns)==1
            serx = gdx.droplevel('iter', axis=1).iloc[:,0]
 
            assert isinstance(serx, pd.Series)
            assert serx.name== 'tvals_raw'
            d[i] = {'tvals_raw_serx':serx}
        #=======================================================================
        # setup pars
        #=======================================================================
 
 
        #=======================================================================
        # execute children
        #=======================================================================
        log.info('on %i w/ iters=%i and dscale_meth=%s'%(len(tvals_raw_serx), iters, dscale_meth))
        res_d = self.model_retrieve(dkey, dscale_meth=dscale_meth,
                               #mindex=mindex,
                               iters=iters,
                               finv_agg_d=finv_agg_d,
                               iter_kwargs =d,

                               logger=log, **kwargs)
        
        #=======================================================================
        # assemble and collapse
        #=======================================================================
        """
        res_d.keys()
        res_d[0].keys()"""
        dxind = pd.concat(res_d, axis=1)
        
        dx = pd.concat({dkey:dxind},axis=1, names=['dkey', 'iter'])
        
        #=======================================================================
        # write lyaer 
        #=======================================================================
        """only writing the area splits for now"""
 

        
        #=======================================================================
        # write pick
        #=======================================================================
        log.info('finished w/ %s'%str(dx.shape))
        if write:
            self.ofp_d[dkey] = self.write_pick(dx,
                                   os.path.join(self.wrk_dir, '%s_%s.pickle' % (dkey, self.longname)),
                                   logger=log)
            
        return dx
    
 
 
        
    
    def model_retrieve(self,
                       dkey=None,  # method to run
 
                       iters=None,
                       logger=None,
                       iter_kwargs=None,
                       **kwargs):
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger = self.logger
        log = logger.getChild('retM')
        if iters is None: iters=self.iters
 
        if iter_kwargs is None: iter_kwargs = {i:dict() for i in range(iters)}
        
        log.info('\n\ncalling \'%s\' on %i\n\n' %(dkey, iters))
        
        """
        self.data_d.keys()
        """
        #=======================================================================
        # precheck
        #=======================================================================
        
        assert len(iter_kwargs)==iters
        assert np.array_equal(np.array(list(iter_kwargs.keys())), np.array(range(iters)))
 
        #=======================================================================
        # loop and load
        #=======================================================================

        init_kwargs = {k:getattr(self,k) for k in [
            'name', 'prec', 'trim', 'out_dir', 'overwrite', 'temp_dir', 'bk_lib', 'write', 'proj_lib']}
        res_d = dict()
        for i in range(iters):
            log.info('%i/%i' % (i + 1, iters))
            
            with Model(session=self, tag='%s_%i'%(self.tag, i),  **init_kwargs) as wrkr:
 
                res_d[i] = wrkr.retrieve(dkey, write=False, logger=logger.getChild('%i'%i), 
                                         **{**kwargs,**iter_kwargs[i]}, #combine function level with iteration kwargs
                                          )
                
                
        log.info('finished \'%s\' w/ %i'%(dkey, len(res_d)))
        
        return res_d
    
    def _get_meta(self, #get a dictoinary of metadat for this model
                 ):
        
        d = super()._get_meta()
 
        
        attns = ['iters', 'modelID']
        
 
        return {**d, **{k:getattr(self, k) for k in attns}}
    
 

