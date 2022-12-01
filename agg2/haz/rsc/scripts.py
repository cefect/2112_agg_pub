'''
Created on Aug. 19, 2022

@author: cefect

classifying downsample type

let's try w/ mostly rasterio?
simple session (no retrieve)
'''
 

import numpy as np
import numpy.ma as ma
import os, copy, datetime
import rasterio as rio
from definitions import wrk_dir 
from hp.np import apply_block_reduce, downsample 
from hp.oop import Session
from hp.rio import RioWrkr, assert_extent_equal, is_divisible, assert_rlay_simple, load_array
import scipy.ndimage
from agg2.haz.coms import assert_dem_ar, assert_wse_ar, cm_int_d


from hp.plot import plot_rast
import matplotlib.pyplot as plt

def now():
    return datetime.datetime.now()
 
class ResampClassifier(RioWrkr): 
    """shareable tools for build downsample classification masks"""
    
    cm_int_d=cm_int_d
    
    def __init__(self, 
                 aggscale=2,
                 obj_name=None,
 
                 **kwargs):
        """
        
        Parameters
        ----------
        aggscale: int, default 2
            multipler for new pixel resolution
            oldDimension*(1/aggscale) = newDimension
 
        """
        if obj_name is None: obj_name='dsc%03i'%aggscale
        
        super().__init__(obj_name=obj_name, **kwargs)
        
        #=======================================================================
        # attach
        #=======================================================================
        self.aggscale=aggscale
 

        
    
    #===========================================================================
    # UNIT BUILDRES-------
    #===========================================================================
    def build_crop(self, raw_fp, new_shape=None, divisor=None, **kwargs):
        """build a cropped raster which achieves even division
        
        Parameters
        ----------
        bounds : optional
        
        divisor: int, optional
            for computing the nearest whole division
        """
        #=======================================================================
        # defaults
        #=======================================================================
        rawName = os.path.basename(raw_fp).replace('.tif', '')[:6]
        log, tmp_dir, out_dir, ofp, resname, write = self._func_setup('crops%s'%rawName,  **kwargs)
        
        assert isinstance(divisor, int)
        
        with RioWrkr(rlay_ref_fp=raw_fp, session=self) as wrkr:
            
            raw_ds = wrkr._base()
            
            """
            raw_ds.read(1)
            """
            #=======================================================================
            # precheck
            #=======================================================================
            assert not is_divisible(raw_ds, divisor), 'no need to crop'            
 
            #===================================================================
            # compute new_shape
            #===================================================================
            if new_shape is None: 
                new_shape = tuple([(d//divisor)*divisor for d in raw_ds.shape])
                
            log.info('cropping %s to %s for even divison by %i'%(
                raw_ds.shape, new_shape, divisor))
                
 
            
            self.crop(rio.windows.Window(0,0, new_shape[1], new_shape[0]), dataset=raw_ds,
                      ofp=ofp, logger=log)
            
        return ofp
            
 
                

 
    
    def build_coarse(self,
                        raw_fp,
                        aggscale=None,
                        resampleAlg='average',
                        **kwargs):
        
        """
        construct a coarse raster from a raw raster
        
        Parameters
        ----------
        raw_fp: str
            filepath to fine raster
 
        """
        #=======================================================================
        # defaults
        #=======================================================================
        rawName = os.path.basename(raw_fp).replace('.tif', '')[:6]
        
        log, tmp_dir, out_dir, ofp, resname, write = self._func_setup('coarse%s'%rawName,  **kwargs)
        
        if aggscale is None: aggscale=self.aggscale
        #=======================================================================
        # precheck
        #=======================================================================
        assert isinstance(aggscale, int)
        assert aggscale>1
        
        if __debug__:
            #===================================================================
            # check we have a divisible shape
            #===================================================================
            with rio.open(raw_fp, mode='r') as dem_ds:
 
                dem_shape = dem_ds.shape
                
                assert_rlay_simple(dem_ds, msg='DEM')
 
                
            #check shape divisibility
            for dim in dem_shape:
                assert dim%aggscale==0, 'unequal dimension (%i/%i -> %.2f)'%(dim, aggscale, dim%aggscale)
                
 
            
        #=======================================================================
        # downsample
        #=======================================================================
        resampling = getattr(rio.enums.Resampling, resampleAlg)
        with RioWrkr(rlay_ref_fp=raw_fp, session=self) as wrkr:
            
            assert_dem_ar(wrkr._base().read(1)) 
            
            res_ds = wrkr.resample(resampling=resampling, scale=1/aggscale)
            wrkr.write_memDataset(res_ds, ofp=ofp)
            
            
        #=======================================================================
        # wrap
        #=======================================================================
        #mstore.removeAllMapLayers()
        assert os.path.exists(ofp)
        return ofp
    
    def build_delta(self, 
                    dem_fp, wse_fp,
                    **kwargs):
        
        """build DEM WSE delta
        
        this is a bit too simple...
        """
        
        log, tmp_dir, out_dir, ofp, resname, write = self._func_setup('delta',  **kwargs)
        
        
        with RioWrkr(rlay_ref_fp=dem_fp, session=self) as wrkr:
            
            #===================================================================
            # load layers----
            #===================================================================
            #dem
            dem_ar = wrkr._base().read(1)
            
            #load the wse
            #===================================================================
            # wse_ds = wrkr.open_dataset(wse_fp)
            # wse_ar = wse_ds.read(1)
            #===================================================================
            wse_ar = load_array(wse_fp)
            
            assert dem_ar.shape==wse_ar.shape
 
            
            #===================================================================
            # compute
            #===================================================================
            delta_ar = np.nan_to_num(wse_ar-dem_ar, nan=0.0)
            
            assert np.all(delta_ar>=0)
            
            wrkr.write_array(delta_ar, ofp=ofp, logger=log)
            
        return ofp
    
    
    




    
    
    def get_catMasks(self,

                     dem_ds=None,
                     wse_ds=None,
                     dem_ar=None, wse_ar=None,
                     aggscale=None,
                     **kwargs):
        """compute the a mask for each resample category
        
        Returns
        -------
        cm_d, dict
            four masks from build_cat_masks {category label: np.ndarray}
        
        """
        
        #=======================================================================
        # defautls
        #=======================================================================
        log, tmp_dir, out_dir, ofp, resname, write = self._func_setup('cm',  **kwargs)
        if aggscale is None: aggscale=self.aggscale
        
        assert isinstance(aggscale, int)
        start = now()
        #===================================================================
        # load layers----
        #===================================================================
        #fine dem
        if dem_ar is None:
            dem_ar = load_array(dem_ds)
            
            assert_dem_ar(dem_ar)
 
        #assert_extent_equal(dem_ds, wse_ds)        
        
        if wse_ar is None:
            wse_ar = load_array(wse_ds)
            
            assert_wse_ar(wse_ar)
 
        
        #=======================================================================
        # globals
        #=======================================================================
        def apply_reducer(ar, func):
            #apply aggregation
            arC = apply_block_reduce(ar, func, aggscale=aggscale) #max of each coarse block
            
            #rescale back to original
            """would have been nicer to just keep the reduce dscale"""
            fine_ar = scipy.ndimage.zoom(arC, aggscale, order=0, mode='reflect',   grid_mode=True)
            #return np.kron(arC, np.ones((aggscale,aggscale))) #rescale back to original res
 
            assert fine_ar.shape==ar.shape
            return fine_ar
 
        
        def log_status(k):
            log.info('    calcd %i/%i \'%s\''%(cm_d[k].sum(), dem_ar.size, k))
            
        cm_d = dict()
        #===================================================================
        # compute delta
        #===================================================================
        log.info('    computing deltas')
        delta_ar = np.nan_to_num(wse_ar-dem_ar, nan=0.0)
        assert np.all(delta_ar>=0)
        
        #=======================================================================
        # #dry-dry: max(delta) <=0
        #=======================================================================
        log.info('    computing DD')
        delta_max_ar = apply_reducer(delta_ar, np.max)
        
        cm_d['DD'] = delta_max_ar<=0
        log_status('DD')
        #===================================================================
        # #wet-wet: min(delta) >0
        #===================================================================
        delta_min_ar = apply_reducer(delta_ar, np.min)
        
        cm_d['WW'] = delta_min_ar>0
        log_status('WW')
        #===================================================================
        # #partials: max(delta)>0 AND min(delta)==0
        #===================================================================
        partial_bool_ar = np.logical_and(
            delta_max_ar>0,delta_min_ar==0)
        
        #check this is all remainers
        assert partial_bool_ar.sum() + cm_d['WW'].sum() + cm_d['DD'].sum() == partial_bool_ar.size
        
        if not np.any(partial_bool_ar):
            log.warning('no partials!')
        else:
            log.info('    flagged %i/%i partials'%(partial_bool_ar.sum(), partial_bool_ar.size))
    

        #===============================================================
        # compute means
        #===============================================================
        dem_mean_ar = apply_reducer(dem_ar, np.mean)
        wse_mean_ar = apply_reducer(wse_ar, np.nanmean) #ignore nulls in denomenator
        #===============================================================
        # #wet-partials: mean(DEM)<mean(WSE)
        #===============================================================
        cm_d['WP'] = np.logical_and(partial_bool_ar,
                                     dem_mean_ar<wse_mean_ar)
        log_status('WP')
        
        #dry-partials: mean(DEM)>mean(WSE)
        cm_d['DP'] = np.logical_and(partial_bool_ar,
                                     dem_mean_ar>=wse_mean_ar)
        
        log_status('DP')
        #===================================================================
        # compute stats
        #===================================================================
        stats_d = {k:ar.sum()/ar.size for k, ar in cm_d.items()}
        
        
        log.info('computed in %.2f secs w/ \n    %s'%((now()-start).total_seconds(), stats_d))
        
        #===================================================================
        # check
        #===================================================================
        chk_ar = np.add.reduce(list(cm_d.values()))==1
        assert np.all(chk_ar), '%i/%i failed logic'%((~chk_ar).sum(), chk_ar.size)
        
        for k, ar in cm_d.items():
            #print(ar.shape)
            assert dem_ar.shape==ar.shape, k
        #===================================================================
        # output rasteres
        #===================================================================
        
        return cm_d


    def get_catMasks2(self,
 
                     dem_ar=None, wse_ar=None,
                     aggscale=None,
                     **kwargs):
        """compute the a mask for each resample category
        
        converted to masked arrays
        
        Returns
        -------
        cm_d, dict
            four masks from build_cat_masks {category label: np.ndarray}
            
        Notes
        --------
        could speed this up with xarray's coarsen
        
        """
        
        #=======================================================================
        # defautls
        #=======================================================================
        log, tmp_dir, out_dir, ofp, resname, write = self._func_setup('cm',  **kwargs)
        if aggscale is None: aggscale=self.aggscale
        
        assert isinstance(aggscale, int)
        start = now()
        #===================================================================
        # load layers----
        #===================================================================
 
            
        assert_dem_ar(dem_ar, masked=False)            
        assert_wse_ar(wse_ar, masked=True)
        
        #=======================================================================
        # globals
        #=======================================================================
 
 
        
        def log_status(k):
            log.info('    calcd %i/%i \'%s\''%(cm_d[k].sum(), dem_ar.size, k))
            
        cm_d = dict()
        #===================================================================
        # compute delta
        #===================================================================
        log.info('    computing deltas')
        delta_ar = (wse_ar-dem_ar).filled(0.0)
        assert np.all(delta_ar>=0)
        
        #=======================================================================
        # #dry-dry: max(delta) <=0
        #=======================================================================
        log.info('    computing DD')        
        delta_max_ar = apply_block_reduce(delta_ar, np.max, aggscale=aggscale)
        
        cm_d['DD'] = delta_max_ar<=0
        log_status('DD')
        #===================================================================
        # #wet-wet: min(delta) >0
        #===================================================================        
        delta_min_ar = apply_block_reduce(delta_ar, np.min, aggscale=aggscale)
        
        cm_d['WW'] = delta_min_ar>0
        log_status('WW')
        #===================================================================
        # #partials: max(delta)>0 AND min(delta)==0
        #===================================================================
        partial_bool_ar = np.logical_and(
            delta_max_ar>0,delta_min_ar==0)
        
        #check this is all remainers
        assert partial_bool_ar.sum() + cm_d['WW'].sum() + cm_d['DD'].sum() == partial_bool_ar.size
        
        if not np.any(partial_bool_ar):
            log.warning('no partials!')
        else:
            log.info('    flagged %i/%i partials'%(partial_bool_ar.sum(), partial_bool_ar.size))
    

        #===============================================================
        # compute means
        #===============================================================
        
        
        dem_mean_ar =apply_block_reduce(dem_ar, np.mean, aggscale=aggscale).data
        
        """same thing
        apply_block_reduce(wse_ar.filled(np.nan), np.nanmean, aggscale=aggscale)"""
        wse_mean_ar = apply_block_reduce(wse_ar, np.mean, aggscale=aggscale).filled(np.nan) #ignore nulls in denomenator
        
 
        #===============================================================
        # #wet-partials: mean(DEM)<mean(WSE)
        #===============================================================
        cm_d['WP'] = np.logical_and(partial_bool_ar,
                                     dem_mean_ar<wse_mean_ar)
        log_status('WP')
        
        #dry-partials: mean(DEM)>mean(WSE)
        cm_d['DP'] = np.logical_and(partial_bool_ar,
                                     dem_mean_ar>=wse_mean_ar)
        
        log_status('DP')
        #===================================================================
        # compute stats
        #===================================================================
        stats_d = {k:ar.sum()/ar.size for k, ar in cm_d.items()}
        
        
        log.info('computed in %.2f secs w/ \n    %s'%((now()-start).total_seconds(), stats_d))
        
        #===================================================================
        # check
        #===================================================================
        chk_ar = np.add.reduce(list(cm_d.values()))==1
        assert np.all(chk_ar), '%i/%i failed logic'%((~chk_ar).sum(), chk_ar.size)
        
        for k, ar in cm_d.items(): 
            assert delta_max_ar.shape==ar.shape, k
            assert not isinstance(ar, ma.MaskedArray)
        #===================================================================
        # output rasteres
        #===================================================================
        
        return cm_d

    def get_catMosaic(self, cm_d, cm_int_d=None, logger=None):
        """moasic together the four dsc cat masks
        
        Parameters
        ------------
        cm_d: dict
            four masks from build_cat_masks {category label: np.ndarray}
        
        cm_int_d: dict
            integer mappings for each category
            
        Returns
        ---------
        cm_ar: np.ndarray
            unique int values for each dsc category
        """
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger=self.logger
        log=logger.getChild('get_cMos')
        
        if cm_int_d is None:
            cm_int_d = self.cm_int_d.copy()
        start = now()
        #=======================================================================
        # precheck
        #=======================================================================
        miss_l = set(cm_int_d.keys()).symmetric_difference(cm_d.keys())
        assert len(miss_l) == 0, miss_l
        assert np.all(np.add.reduce(list(cm_d.values())) == 1), 'discontinuous masks'
        #=======================================================================
        # loop and build
        #=======================================================================
        log.info('building dsc cat Mosaic on %s'%str(cm_d['DD'].shape))
        res_ar = np.full(cm_d['DD'].shape, np.nan, dtype=np.int32) #build an empty dumm
        for k, mar in cm_d.items():
            log.info('    for %s setting %i/%i' % (k, mar.sum(), mar.size))
            np.place(res_ar, mar, cm_int_d[k])
        
        #=======================================================================
        # check
        #=======================================================================
        assert_cm_ar(res_ar)
        
        log.info('finished building mosaic in %.2f secs'%((now()-start).total_seconds()))
        return res_ar
    
 
        
    def get_catMasksStats(self, cm_d):
        """get stats from container of category masks"""
        
        d = dict()
        for name, mask_ar in cm_d.items():
            assert isinstance(mask_ar, np.ndarray)
            d[name] = {
                'key':self.cm_int_d[name],
                'sum':mask_ar.sum(),
                'frac':mask_ar.sum()/mask_ar.size
                }
            
        return d
 
            
    def mosaic_to_masks(self,cm_ar):
        """convert mosaic back boolean masks"""
        assert_cm_ar(cm_ar)
        
        res_d = dict()
        for name, val in self.cm_int_d.items():
            res_d[name]=cm_ar==val
            
        return res_d
            
    #===========================================================================
    # PRIVATE HELPERS---------
    #===========================================================================
 
        

        
 
    
    
class ResampClassifierSession(ResampClassifier, Session):
    """standalone session for downsample classification"""
 
    
    def __init__(self, 
 
                 obj_name='dsc',
 
                 **kwargs):
 
 
        
        super().__init__(obj_name=obj_name, wrk_dir=wrk_dir,  **kwargs)
        
    #===========================================================================
    # MAIN RUNNERS-----
    #===========================================================================
    def run_all(self,demR_fp, wseR_fp,
                demC_fp=None,
                 aggscale=None, **kwargs):
        """prep layers and build downsample classification 
        
        
        Parameters
        ----------
        demC_fp: str optional
            filepath to the coarse DEM. Otherwise, this is built from teh raw/fine DEM
 
        """
        #=======================================================================
        # defaults
        #=======================================================================
        if aggscale is None: aggscale=self.aggscale
        start = datetime.datetime.now()
        log, tmp_dir, out_dir, ofp, resname, write = self._func_setup('dsc_%03i'%aggscale,  **kwargs)
        skwargs = dict(logger=log, tmp_dir=tmp_dir, out_dir=tmp_dir, write=write)
        
        
        #=======================================================================
        # precheck
        #=======================================================================
        assert_extent_equal(demR_fp, wseR_fp)
        #=======================================================================
        # check divisibility
        #=======================================================================
        if not is_divisible(demR_fp, aggscale):
            log.warning('uneven division w/ %i... clipping'%aggscale)
            
            dem_fp = self.build_crop(demR_fp, divisor=aggscale, **skwargs)
            wse_fp = self.build_crop(wseR_fp, divisor=aggscale, **skwargs)
            
        else:
            dem_fp, wse_fp = demR_fp, wseR_fp
 
        
        
        #=======================================================================
        # algo
        #=======================================================================
        #build coarse dem
        #=======================================================================
        # if demC_fp is None:
        #     demC_fp = self.build_coarse(dem_fp, aggscale=aggscale, **skwargs)
        #=======================================================================
            
        #each mask
        cm_d, _ = self.build_cat_masks(dem_fp, wse_fp, **skwargs)
        
        #moasic together
        cm_fp = self.build_cat_mosaic(cm_d,ofp=ofp, **skwargs)
        
        #=======================================================================
        # wrap
        #=======================================================================
        log.info('finished in %s'%(datetime.datetime.now()-start))
        return ofp
    
    def build_cat_masks(self,
                        dem_fp, wse_fp, 
                        aggscale=None,
 
 
                        **kwargs):
        
        """build masks for each category
        
        Parameters
        -----------
        dem_fp: str
            filepath to fine/raw DEM
 
        wse_fp: str
            filepath  to fine WSE dem
        """
        #=======================================================================
        # defaults
        #=======================================================================
        log, tmp_dir, out_dir, ofp, resname, write = self._func_setup('cmask',  **kwargs)
        if aggscale is None: aggscale=self.aggscale
 
        #=======================================================================
        # exec
        #=======================================================================
        ofp_d=dict()
        with ResampClassifier(rlay_ref_fp=dem_fp, session=self, aggscale=aggscale) as wrkr:
            #load the layers
            wse_ds = wrkr.open_dataset(wse_fp)
            dem_ds = wrkr._base()
            
            #build the masks
            res_d = wrkr.get_catMasks(dem_ds=dem_ds, wse_ds=wse_ds)
            
            
            #write masks
            if write:
                for k, mar in res_d.items():
                    ofp_d[k] = wrkr.write_array(mar.astype(int), ofp=os.path.join(out_dir, '%s_mask.tif'%k))
            
            
            
        log.info('finished writing %i'%len(ofp_d))
        return res_d, ofp_d
    
    def build_cat_mosaic(self, cm_d,
                         cm_int_d=None,
                         output_kwargs={},
                         **kwargs):
        """
        construct a mosaic from the 4 category masks
        

        """
        #=======================================================================
        # defaults
        #=======================================================================
        log, tmp_dir, out_dir, ofp, resname, write = self._func_setup('cmMosaic',  **kwargs)
        res_ar = self.get_catMosaic(cm_d, cm_int_d=cm_int_d)
        
 
        #=======================================================================
        # write
        #=======================================================================
        if write:
            self.write_array(res_ar, ofp=ofp, logger=log, **output_kwargs)
        
        return ofp
    
    
#===============================================================================
# HELPER FUNCS-------
#===============================================================================
def assert_cm_ar(ar, msg=''):
    """check dsc mosaic array satisfies assumptions"""
    if not __debug__: # true if Python was not started with an -O option
        return
    
    __tracebackhide__ = True
    
    assert isinstance(ar, np.ndarray) 
    assert 'int' in ar.dtype.name
    
    if not np.all(ar % 2 == 1):
        raise AssertionError('failed to get all odd values\n'+msg)
    
    #check we only have valid values
    vali_ar = np.array(list(ResampClassifier.cm_int_d.values()))
    
    if not np.all(np.isin(np.unique(ar), vali_ar)):
        raise AssertionError('got some unexpected category values\n'+msg)
    

def runr(
        dem_fp=None, wse_fp=None,
        **kwargs):
    with ResampClassifier(rlay_ref_fp=dem_fp, **kwargs) as ses:
        ofp = ses.run_all(dem_fp, wse_fp)
        
    return ofp
 
 
    
