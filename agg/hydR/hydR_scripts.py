'''
Created on Apr. 26, 2022

@author: cefect

analysis to focus on rasters
'''

#===============================================================================
# imports-----------
#===============================================================================
import os, datetime, math, pickle, copy, sys
import qgis.core
from qgis.core import QgsRasterLayer, QgsMapLayerStore
import pandas as pd
import numpy as np



idx = pd.IndexSlice
from hp.exceptions import Error
from hp.pd import get_bx_multiVal
import hp.gdal

from hp.Q import assert_rlay_equal, QgsMapLayer, view
from hp.basic import set_info
from agg.hyd.hscripts import  RasterCalc

from agg.hydR.hydR_coms import RRcoms, Catalog, assert_lay_lib, assert_lay_d
    


class RastRun(RRcoms):
    
    phase_l=['depth', 'diff']
    index_col = list(range(6))
    
    def __init__(self,
                 name='rast',
                 phase_l=['depth'],
                 data_retrieve_hndls={},
                 rcol_l=None, 
                 pick_index_map={},
                 phase_d={},
                 **kwargs):
        
        
        if rcol_l is None:
            rcol_l=[self.saCn, self.resCn]
        self.rcol_l=rcol_l
        
        data_retrieve_hndls = {**data_retrieve_hndls, **{
            #depth rasters
            'drlay_lib':{ #overwrites Model's method
                'compiled':lambda **kwargs:self.load_layer_lib(**kwargs),
                'build':lambda **kwargs: self.build_drlays2(**kwargs),
                },
            'rstats':{  
                'compiled':lambda **kwargs:self.load_pick(**kwargs),
                'build':lambda **kwargs: self.build_stats(**kwargs),
                },
            'wetStats':{  
                'compiled':lambda **kwargs:self.load_pick(**kwargs),
                'build':lambda **kwargs: self.build_wetStats(**kwargs),
                },
 
            'gwArea':{  
                'compiled':lambda **kwargs:self.load_pick(**kwargs),
                'build':lambda **kwargs: self.build_gwArea(**kwargs),
                },
            
            'noData_cnt':{  #note, rstats will also have this
                'compiled':lambda **kwargs:self.load_pick(**kwargs),
                #'build':lambda **kwargs: self.build_stats(**kwargs),
                },
 
            'noData_pct':{  
                'compiled':lambda **kwargs:self.load_pick(**kwargs),
                'build':lambda **kwargs: self.build_noDataPct(**kwargs),
                },

            
            #difference rasters
            'difrlay_lib':{  
                'compiled':lambda **kwargs:self.load_layer_lib(**kwargs),
                'build':lambda **kwargs: self.build_difrlays(**kwargs),
                },
            
            'rstatsD':{  
                'compiled':lambda **kwargs:self.load_pick(**kwargs),
                'build':lambda **kwargs: self.build_stats(**kwargs),
                },
            'rmseD':{  
                'compiled':lambda **kwargs:self.load_pick(**kwargs),
                'build':lambda **kwargs: self.build_rmseD(**kwargs),
                },
 
           
            }}
        
        #=======================================================================
        # pickle indexers
        #=======================================================================
        pick_index_map.update({
            'drlay_lib':(self.resCn, self.saCn),
            })
        pick_index_map['difrlay_lib'] = copy.copy(pick_index_map['drlay_lib'])
        self.pick_index_map=pick_index_map
        
        #=======================================================================
        # phase
        #=======================================================================
        phase_d.update({
            'depth':('rstats', 'wetStats', 'gwArea','noData_cnt', 'noData_pct'),
            'diff':('rstatsD','rmseD'),
 
            
            })
        
        
        
        super().__init__(phase_d=phase_d,
                         data_retrieve_hndls=data_retrieve_hndls, name=name,
                         **kwargs)
        
        self.phase_l=phase_l
        

                
            
        
        

    #===========================================================================
    # RASTER DATA construction-----------
    #===========================================================================
    def runDownsample(self):
        
        #=======================================================================
        # #rasters and stats
        #=======================================================================
        self.retrieve('drlay_lib')
        
        self.retrieve('rstats')
        
        self.retrieve('wetStats')
        
 
        
        self.retrieve('gwArea')
        
 
        
        self.retrieve('noData_pct')
        
        
 
    
    def build_drlays2(self,
                      
                      #data
                      proj_lib=None,
                     
                     #parameters [calc loop]
                     iters=3, #number of downsamples to perform
                     resolution_scale = 2, 
                     base_resolution=None, #resolution of raw data
                     resolution_iters=None,
                     
                     
                     #parameters [get_drlay]. for non base_resolution
                     dsampStage='pre',downSampling='Average',severity='hi',
                     
                     sequenceType='none', #how to construct from previous iteration... or from raw
                        #none: always start from base/raw data
                        #inputs: use inputs (dem, wse) from previous iter (only for dsampStage=pre)
                        #outputs: use outputs (depths) from previous iter. negates dsampStage for most iters
                     
                     #outputs
                     out_dir=None,
                     
                     dkey='drlay_lib',logger=None,write=None,**kwargs):
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger=self.logger
        log = logger.getChild('build_drlays')
        if write is None: write=self.write
        assert dkey=='drlay_lib'
        
        #resolutions
        if base_resolution is None:
            from definitions import base_resolution
        
        assert not dsampStage=='none'
        assert not downSampling=='none'
        
        temp_dir = self.temp_dir #collect
        mstore = QgsMapLayerStore()
        
        def glay(fp):
            return self.get_layer(fp, mstore=mstore, logger=log)
        #=======================================================================
        # clean proj_lib
        #=======================================================================
        if proj_lib is None:
            proj_lib = copy.deepcopy(self.proj_lib)
        for sa, d in proj_lib.items():
            if 'finv_fp' in d:
                del d['finv_fp']
        #=======================================================================
        # build iter loop
        #=======================================================================
        #[10, 30, 90]
        if resolution_iters is None:
            resolution_iters = [base_resolution*(resolution_scale)**i for i in range(iters)]
        
        assert max(resolution_iters)<1e5
        
        #=======================================================================
        # setup sequential
        #=======================================================================
        """this is only needed by sequenceType=inputs... 
        but cleaner to just use this for all (and not reset)"""
        ins_lay_lib={k:dict() for k in proj_lib.keys()}
        #if sequenceType=='inputs':
        #setup inputs for first iter
        
        for sa, d in proj_lib.items():
            ins_lay_lib[sa]['wse_rlay'] = glay(d['wse_fp_d'][severity])
            ins_lay_lib[sa]['dem_rlay'] =glay(d['dem_fp_d'][base_resolution])
        
        
                
        #=======================================================================
        # retrive rasters per StudyArea
        #=======================================================================
        """leaving everything on the StudyArea to speed things up"""
        
        #execute
        log.info('constructing %i: %s'%(len(resolution_iters), resolution_iters))
        res_lib = dict()
        meta_lib=dict()
        cnt=0
        first = True
        lay_d=None
        for i, resolution in enumerate(resolution_iters):
            log.info('\n\n%i/%i at %i\n'%(i+1, len(resolution_iters), resolution))
            assert_lay_lib(ins_lay_lib, msg='sequence setup')
            #===================================================================
            # #handle parameters
            #===================================================================
            if i==0:
                """because get_drlay has expectations for the base
                could also skip the downsampling... but this is mroe consistent"""
                dStage, dSamp='none', 'none'
            else:
                dStage, dSamp=dsampStage, downSampling
                
            #reset temp_dir
            self.temp_dir = os.path.join(temp_dir, 'r%i'%resolution)
            if not os.path.exists(self.temp_dir):os.makedirs(self.temp_dir)

            #===================================================================
            # #build the depth layer
            #===================================================================
            d1 = {k:dict() for k in ['rlay','noData_cnt', 'wse_fp', 'dem_fp']}
            
            #common kwargs
            saKwargs = dict(logger=log.getChild(str(i)), write=False,proj_lib=proj_lib)
            
            #normal calcs
            if (first) or (not sequenceType=='outputs'):
                try:
                    """
                    StudyArea.get_drlay()
                    """
                    d = self.sa_get(meth='get_drlay', fkwargs=ins_lay_lib, dkey=dkey,
                              base_resolution=base_resolution,dsampStage=dStage, 
                               resolution=resolution, downSampling=dSamp,
                                severity=severity,**{**kwargs, **saKwargs})
                    
    
                    
                except Exception as e:
                    raise IOError('failed get_drlay on reso=%i w/ \n    %s'%(resolution, e))
                
 
                #rekey results
                for k in d1.keys():
                    d1[k] = {sa:d0.pop(k) for sa, d0 in d.items()}
                
                #handle just the layers
                lay_d = d1.pop('rlay')
                
 
            #===================================================================
            # just downsample
            #===================================================================
            elif sequenceType=='outputs':
                #buidl args for StudyArea.get_resamp()
                fargs_d = {sa:[rlay, resolution, downSampling] for sa,rlay in lay_d.items()} 
                
                fkwargs = {sa:{'extents':rlay.extent()} for sa,rlay in lay_d.items()} 
 
                #downsample
                fp_d = self.sa_get(meth='get_resamp',fargs_d=fargs_d,fkwargs=fkwargs,
                                   fval=0.0, #seems to be happening on the noise
                                   **{**kwargs, **saKwargs})
                
                #add noData
                d1['noData_cnt'] = {sa:hp.gdal.getNoDataCount(r) for sa,r in fp_d.items()}
                
                
                
                #load layers (not adding to mstore)
                lay_d = {sa:self.get_layer(fp,logger=log) for sa,fp in fp_d.items()}
                
 
            
            else:
                raise IOError(sequenceType)
            
            #===================================================================
            # handle results
            #===================================================================
            assert_lay_d(lay_d, msg='res=%i'%resolution)
            
            for sa, drlay in lay_d.items():
                assert hp.gdal.getNoDataCount(drlay.source())==0, 'got nulls on %s'%sa
            
            #add back names
            d1['rlay'] = {k:r.name() for k,r in lay_d.items()}
            
            #add temp filepaths (we re-write below)
            d1['drlay_fp'] = {k:r.source() for k,r in lay_d.items()}

            cnt+=len(lay_d)
            res_lib[resolution]=lay_d
 
            meta_lib[resolution] = pd.DataFrame.from_dict(d1)
            
            #===================================================================
            # wrap
            #===================================================================
            if sequenceType=='inputs':
                mstore.removeAllMapLayers() #kill previous iter
                
                for sa in proj_lib.keys():
                    ins_lay_lib[sa]['wse_rlay'] = glay(d1['wse_fp'][sa])
                    ins_lay_lib[sa]['dem_rlay'] =glay(d1['dem_fp'][sa])
                    
            log.debug('finished %i'%i)
            first=False
                     
 
 
        self.temp_dir = temp_dir #revert
        log.info('finished building %i'%cnt)
        
 
        #=======================================================================
        # handle layers----
        #=======================================================================
        if write:
            self.store_lay_lib(  res_lib, dkey,out_dir=out_dir, logger=log)
            
        #=======================================================================
        # handle meta
        #=======================================================================
        if write:
            rdx = pd.concat(meta_lib, names=[self.resCn, self.saCn]) #.T.stack(level=0).swaplevel().sort_index(sort_remaining=True)
            #rdx.index.set_names([self.resCn, self.saCn], inplace=True)
            """
            view(rdx)
            """
            
            #rename to not conflict
            #rdx = rdx.rename(columns={'noData_cnt':'noData_cnt2'})
            
            rserx = rdx['noData_cnt'].rename('noData_cnt2')
            
            assert np.array_equal(rserx.index.names, np.array([self.resCn, self.saCn]))
            
            dkey1 = 'noData_cnt'
            self.data_d[dkey1] = rserx.copy()
            self.ofp_d[dkey1] = self.write_pick(rserx,
                                   os.path.join(self.wrk_dir, '%s_%s.pickle' % (dkey1, self.longname)),
                                   logger=log)
 
        #=======================================================================
        # wrap-----
        #=======================================================================
 
        assert_lay_lib(res_lib, msg='%s post'%dkey)
        mstore.removeAllMapLayers()
   
        return res_lib
    
 
    def build_stats(self, #calc the layer stats 
                    dkey='rstats',
                    logger=None, 
                    lay_lib=None,
                     **kwargs):
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger=self.logger
        log = logger.getChild('build_stats')
        #assert dkey=='rstats'
        
        #=======================================================================
        # retrieve approriate lib
        #=======================================================================
        if lay_lib is None:
            if dkey =='rstats':
                lay_lib = self.retrieve('drlay_lib')
            elif dkey=='rstatsD':
                lay_lib = self.retrieve('difrlay_lib')
            else:
                raise IOError(dkey)
        
        #=======================================================================
        # execut ethe function on the stack
        #=======================================================================
        return self.calc_on_layers(
            func=lambda rlay, meta_d={}, **kwargs:self.rlay_getstats(rlay, **kwargs), 
            logger=log, dkey=dkey, lay_lib=lay_lib, **kwargs)
        
    def build_wetStats(self, #calc wetted stats
                    dkey='wetStats',
                    logger=None, **kwargs):
 
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger=self.logger
        log = logger.getChild('build_wetStats')
        assert dkey=='wetStats'
        
        dx = self.retrieve('rstats')
        """
        dx.index
        """
        
        #=======================================================================
        # define the function
        #=======================================================================
        def func(rlay, logger=None, meta_d={}):
            
            
            assert hp.gdal.getNoDataCount(rlay.source())==0
            
            #build a mask layer
            mask_rlay = self.mask_build(rlay, logger=logger, layname='%s_mask'%rlay.name(),
                                        thresh_type='lower_neq', thresh=0.00)
            
            #tally all the 1s
            wet_cnt = self.rasterlayerstatistics(mask_rlay)['SUM']
            
            #retrieve stats for this iter
            stats_ser = dx.loc[idx[meta_d['resolution'], meta_d['studyArea']], :]
            
            res_d = {'wetArea':wet_cnt * stats_ser['rasterUnitsPerPixelY']*stats_ser['rasterUnitsPerPixelX'],
                    'wetCnt':wet_cnt}
            
                        
            #===================================================================
            # wet volume     
            #===================================================================
            #apply the mask
            rlay_maskd = self.mask_apply(rlay, mask_rlay, logger=log, layname='%s_noGW'%rlay.name())
            
            mask_stats = self.rasterlayerstatistics(rlay_maskd)
            tval = mask_stats['SUM']
            
            res_d.update( {'wetVolume':tval * stats_ser['rasterUnitsPerPixelY']*stats_ser['rasterUnitsPerPixelX'],
                    'tval':tval})
            
            #===================================================================
            # wet mean
            #===================================================================
            assert round(tval/wet_cnt, 3)==round(mask_stats['MEAN'], 3)
            res_d['wetMean'] = mask_stats['MEAN']
            
            
            return res_d
            
 
            
        #=======================================================================
        # execute on stack
        #=======================================================================
        rdx = self.calc_on_layers(func=func, logger=log, dkey=dkey, **kwargs)
        return rdx
        
    def build_gwArea(self,#negative cell count
                    dkey='gwArea',
                    logger=None, **kwargs):
        """TODO: write a test for this"""
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger=self.logger
        log = logger.getChild('build_gwArea')
        assert dkey=='gwArea'
 
        dx = self.retrieve('rstats')
        #=======================================================================
        # define the function
        #=======================================================================
        def func(rlay, logger=None, meta_d={}):

            if self.rlay_getstats(rlay)['MIN']>=0:
                wet_cnt=0
            else:
            
                #build a mask layer
                mask_rlay = self.mask_build(rlay, logger=logger, layname='%s_neg_mask'%rlay.name(),
                                            thresh_type='upper_neq', thresh=0.0)
                
                #tally all the 1s
                wet_cnt = self.rasterlayerstatistics(mask_rlay)['SUM']
            
            #retrieve stats for this iter
            stats_ser = dx.loc[idx[meta_d['resolution'], meta_d['studyArea']], :]

            return {dkey:wet_cnt * stats_ser['rasterUnitsPerPixelY']*stats_ser['rasterUnitsPerPixelX']}
 
 
        #=======================================================================
        # execute on stack
        #=======================================================================
        return self.calc_on_layers(func=func, logger=log, dkey=dkey, **kwargs)
 
    
    def build_noDataPct(self,
                    dkey='noData_pct',
                    logger=None, **kwargs):
        """TODO: write a test for this (need to run get_drlay)"""
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger=self.logger
        log = logger.getChild('build_noDataPct')
        assert dkey=='noData_pct'
        
        """need to execute drlays"""
        ndc_dx = self.retrieve('noData_cnt')
        
        dx = self.retrieve('rstats').drop('noData_cnt', axis=1).join(ndc_dx)
 
        serx = dx['noData_cnt2']/dx['cell_cnt']
 
        return serx.rename(dkey).to_frame()
    
    #===========================================================================
    # RASTER DIFFs--------
    #===========================================================================
    def runDiffs(self):#run sequence for difference layer calcs
        
        
        self.retrieve('difrlay_lib')
        
        dx = self.retrieve('rstatsD')
        
        dx = self.retrieve('rmseD')
        
        
 



    def build_difrlays(self, #generate a set of delta rasters and write to the catalog
                      dkey='difrlay_lib',
                      lay_lib=None,

                   logger=None,
                   out_dir=None,write=None,
 
                      **kwargs):
        """
        revised to match behavior of build_drlays2
        
        NULLS: treating these as zeros for difference calculations
        """
        
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger=self.logger
        log = logger.getChild('build_difrlays')
        assert dkey=='difrlay_lib'
        saCn=self.saCn
        if write is None: write=self.write
        mstore= QgsMapLayerStore()
        if out_dir is None: out_dir= self.temp_dir
            
        if lay_lib is None: 
            lay_lib = self.retrieve('drlay_lib')
            
        """
        lay_lib.keys()
        """
        
        #=======================================================================
        # loop and execute on each layer
        #=======================================================================
        log.info('on %i'%len(lay_lib))
        
        res_lib=dict()
        base_d = dict()
        first=True
        cnt=0
        for resolution, d0 in lay_lib.items():
            d = dict()
            assert isinstance(d0, dict), 'got bad type on reso=%s: %s'%(resolution, type(d0))
            log.info('\n\nfor resolution=%i building %i delta rasters \n\n'%(resolution, len(d0)))
            for studyArea, rlay in d0.items():
                #setup and precheck
                tagi = '%i.%s'%(resolution, studyArea)
                assert isinstance(rlay, QgsRasterLayer), tagi
                
                #match the layers crs
                self.qproj.setCrs(rlay.crs())
                
                temp_dir = os.path.join(self.temp_dir, studyArea, str(resolution))
                if not os.path.exists(temp_dir): os.makedirs(temp_dir) 
                #===============================================================
                # #handle baselines
                #===============================================================
                if first:
                    """not sure why this is needed?"""
                    rlay = self.get_layer(self.warpreproject(rlay, 
                                  output=os.path.join(temp_dir, rlay.name()+'.tif'), 
                                    resolution=resolution, compression='none', crsOut=self.qproj.crs(),   
                                    logger=log), mstore=mstore)
                    
                    base_d[studyArea] = self.get_layer(
                        self.fillnodata(rlay,fval=0, logger=log,
                                        output=os.path.join(self.temp_dir, 
                                                '%s_fnd.tif'%rlay.name())), mstore=mstore)
 
                #===============================================================
                # #execute
                #===============================================================
                d[studyArea]=self.get_diff_rlay(
                    rlay, base_d[studyArea], #agg - true
                    logger=log.getChild('%i.%s'%(resolution, studyArea)),
                     out_dir = os.path.join(out_dir, studyArea)
                    )
                
                assert isinstance(d[studyArea], QgsRasterLayer)
                cnt+=1
 
            #wrap
            res_lib[resolution] = d
            first=False
            
        #=======================================================================
        # wrap
        #=======================================================================
        #rdx = pd.concat(res_lib, names=['resolution', 'studyArea'])
        
        log.info('finished building %i'%cnt)
        #=======================================================================
        # handle layers----
        #=======================================================================
 
        if write:
            self.store_lay_lib(  res_lib, dkey,logger=log)
            
        #=======================================================================
        # wrap
        #=======================================================================
        mstore.removeAllMapLayers()
        assert_lay_lib(res_lib, msg='%s post'%dkey)
        return res_lib
    
    
    def build_rmseD(self, #dry RMSE
                    dkey='rmseD',
                    lay_lib=None,
                    logger=None, **kwargs):
        """
        computes starting from the difference layers
            these have no nulls (as parent depth layers were zero-filled)
            
        """
 
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger=self.logger
        log = logger.getChild('build_rmseD')
        assert dkey=='rmseD'
 
 
        #=======================================================================
        # retrieve approriate lib
        #=======================================================================
        if lay_lib is None:
            lay_lib = self.retrieve('difrlay_lib')
            
        #=======================================================================
        # calculator
        #=======================================================================
        def func(rlay, logger=None, meta_d={}):
            
            #setup
            logger.info('on %s'%rlay.name())
            out_dir = os.path.join(self.temp_dir, meta_d[self.saCn], dkey)
            if not os.path.exists(out_dir):os.makedirs(out_dir)
 
            #square the differences
            #res_fp = self.rcalc2(rlay, '{}@1^2'.format(rlay.name()), layname=rlay.name()+'_sq', logger=log)
            #res_fp = self.rastercalculator(rlay, '{}@1^2'.format(rlay.name()), logger=logger)
            
            with RasterCalc(rlay,  session=self,logger=log, name='rmse',
                            out_dir=out_dir, temp_dir=self.temp_dir) as wrkr:
                
                rce = wrkr._rCalcEntry(wrkr.ref_lay)
                formula = '{}^2'.format(rce.ref)
                res_fp = wrkr.rcalc(formula, layname=rlay.name()+'_sq')
            
            #get the stats 
            sum_sq = self.rasterlayerstatistics(res_fp, logger=logger)['SUM']

            cnt = float(self.rlay_get_cellCnt(res_fp, exclude_nulls=False)) #shouldnt be any nulls
            
            rmse =  math.sqrt(sum_sq/cnt)
            
            """this would be faster... but not sure what 'sumOfSquares' is supposed to be
            stats_d = self.rasterlayerstatistics(rlay, logger=logger) 
 
            dp = rlay.dataProvider()
            
            bstats = dp.bandStatistics(1,
                               qgis.core.QgsRasterBandStats.All,
                               qgis.core.QgsRectangle(), 0)
            
            bstats.sum
            assert stats_d['SUM_OF_SQUARES'] == bstats.sumOfSquares
 
            rmse =  math.sqrt(stats_d['SUM_OF_SQUARES']/cnt)"""
                        
                        
 
            return {'rmse':rmse}
        
        #=======================================================================
        # execut ethe function on the stack
        #=======================================================================
        serx= self.calc_on_layers(
            func=func, logger=log, dkey=dkey, lay_lib=lay_lib, **kwargs)
        
        return serx
 

    


    #===========================================================================
    # HELPERS-------
    #===========================================================================
    

        
    def calc_on_layers(self,
                       #data
                       lay_lib=None,
                       
                       #parameters
                       func=lambda rlay, **kwargs:{},
                       subIndexer='resolution',
                       format='dataFrame',
                       
                       #writing
                       write=None, dkey=None,
                       
                       logger=None, **kwargs):
        
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger=self.logger
        log = logger.getChild('calcLayers')
        
        if lay_lib is None: lay_lib = self.retrieve('drlay_lib')
        
        if write is None: write=self.write
        
        #=======================================================================
        # loop and execute on each layer
        #=======================================================================
        log.info('on %i'%len(lay_lib))
        
        res_d=dict()
        for resolution, d0 in lay_lib.items():
            d = dict()
            for studyArea, rlay in d0.items():
                #setup and precheck
                tagi = '%i.%s'%(resolution, studyArea)
                if not isinstance(rlay, QgsMapLayer):
                    raise Error('got bad type on %s: %s'%(
                        tagi, type(rlay)))
                
                #match the layers crs
                self.qproj.setCrs(rlay.crs())
                
                #execute
                res = func(rlay, logger=log.getChild(tagi),meta_d={'studyArea':studyArea, subIndexer:resolution}, **kwargs)
                
                #post
                
                #assert isinstance(res, dict)                
                d[studyArea]=res
                
            #wrap
            if format=='dataFrame':
                res_d[resolution] = pd.DataFrame(d).T
            else:
                res_d[resolution]=d
            
        #=======================================================================
        # wrap
        #=======================================================================
        if format=='dataFrame':
            rdx = pd.concat(res_d, names=[subIndexer, 'studyArea'])
            
            assert isinstance(rdx, pd.DataFrame)
            
        elif format=='dict':
            rdx = res_d
        else:
            raise IOError(format)
        
        
        log.info('finished on %i'%len(rdx))
        
        if write:
            self.ofp_d[dkey] = self.write_pick(rdx,
                                   os.path.join(self.wrk_dir, '%s_%s.pickle' % (dkey, self.longname)),
                                   logger=log)
        
        return rdx
    
    def get_diff_rlay(self, #top minus bottom (agg - base)
                      top_rlay, bot_rlay, 
                      
                      base_resolution=None,
                      out_dir=None,
                      logger=None,
                      ):
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger=self.logger
        log = logger.getChild('get_diff_rlay')
        
        temp_dir = os.path.join(self.temp_dir, 'get_diff_rlay')
        if not os.path.exists(temp_dir): os.makedirs(temp_dir)
        if out_dir is None: out_dir = os.path.join(self.temp_dir, 'difrlay_lib')
        start = datetime.datetime.now()
        mstore = QgsMapLayerStore()
        
        if base_resolution is None:
            from definitions import base_resolution
        
        
        extents = self.layerextent(bot_rlay,precision=0.0, ).extent()
 
        """pretty slow"""
        assert self.rlay_get_resolution(bot_rlay)==float(base_resolution)
        
        #=======================================================================
        # fill nulls
        #=======================================================================
        assert hp.gdal.getNoDataCount(bot_rlay.source())==0
        
        topr1_fp = self.fillnodata(top_rlay, fval=0, logger=log,
                           output=os.path.join(temp_dir, '%s_fnd.tif'%top_rlay.name()))
 
        #=======================================================================
        # warop top to match
        #=======================================================================
        tres = self.rlay_get_resolution(top_rlay)
        if tres > float(base_resolution):
            log.info('warpreproject w/ resolution=%i to %s'%(base_resolution, extents))
            topr2_fp = self.warpreproject(topr1_fp, compression='none', extents=extents, logger=log,
                                            resolution=base_resolution,
                                            output=os.path.join(
                                                temp_dir, 'preWarp_%000i_%s'%(int(base_resolution), os.path.basename(top_rlay.source()))
                                                ))
        elif tres < float(base_resolution):
            raise IOError(tres)
 
        else:
            topr2_fp = topr1_fp
 
        #=======================================================================
        # subtract
        #=======================================================================
        
        log.debug('building RasterCalc')
        with RasterCalc(topr2_fp, name='diff', session=self, logger=log,out_dir=out_dir,
                        ) as wrkr:
 
            entries_d = {k:wrkr._rCalcEntry(v) for k,v in {
                'top':wrkr.ref_lay, 'bottom':bot_rlay}.items()}
            
            assert_rlay_equal(entries_d['top'].raster, entries_d['bottom'].raster)
            
            formula = '%s - %s'%(entries_d['top'].ref, entries_d['bottom'].ref)
            
            #===================================================================
            # execute subtraction
            #===================================================================
            log.info('executing %s'%formula)
            diff_fp1 = wrkr.rcalc(formula, layname='diff_%s'%os.path.basename(topr1_fp).replace('.tif', ''))
            
        #=======================================================================
        # null check
        #=======================================================================
        null_cnt = hp.gdal.getNoDataCount(diff_fp1)
        if not null_cnt==0:
            basename, ext = os.path.splitext(diff_fp1)
            """not sure why this is happenning for some layers"""
            log.warning('got %i nulls on diff for %s...filling'%(null_cnt, top_rlay.name()))
            diff_fp2 = self.fillnodata(diff_fp1, fval=0.0, logger=log,
                                       output=basename+'_fnd.tif')
        else:
            diff_fp2 = diff_fp1
        
        
 
        mstore.removeAllMapLayers()
        
        rlay = self.rlay_load(diff_fp2, logger=log)
        rlay.setName('%s_diff'%top_rlay.name())
        return rlay
 



 