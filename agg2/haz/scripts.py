'''
Created on Aug. 27, 2022

@author: cefect

aggregation hazard 
'''
import numpy as np
import numpy.ma as ma
import pandas as pd
import os, copy, datetime, gc
import rasterio as rio
from rasterio.enums import Resampling
#import scipy.ndimage

from osgeo import gdal
from sklearn.metrics import confusion_matrix
import scipy.stats

import rioxarray
import xarray as xr
from dask.diagnostics import ProgressBar
 
import dask
 

from hp.rio import RioWrkr, assert_extent_equal, is_divisible, assert_rlay_simple, load_array, \
    assert_ds_attribute_match, get_xy_coords
from hp.basic import get_dict_str
from hp.pd import view, append_levels
from hp.sklearn import get_confusion

from agg2.coms import Agg2Session, AggBase
from agg2.haz.rsc.scripts import ResampClassifier
from agg2.haz.coms import assert_dem_ar, assert_wse_ar, assert_dx_names, index_names, coldx_d, assert_xda, assert_xds
idx= pd.IndexSlice

#from skimage.transform import downscale_local_mean
#debugging rasters
from hp.plot import plot_rast
import matplotlib.pyplot as plt
 

#import dask.array as da

def now():
    return datetime.datetime.now()


class UpsampleChild(ResampClassifier, AggBase):
    """child for performing a single downsample set
    
    NOTES
    -------
    I thought it cleaner, and more generalizeable, to keep this on a separate worker"""
    
    def __init__(self, 
 
                 subdir=True,
                 **kwargs):
 
        #=======================================================================
        # build defaults
        #=======================================================================
        
        super().__init__(subdir=subdir,**kwargs)
        

        
    def agg_direct(self,
                         ds_d,
                         resampleAlg='average',
                         aggscale=None,
                         **kwargs):
        """direct aggregation of DEM and WD. WSE is recomputed
        
        NOTE
        ----------
         from DEM and WD (not a direct average of WSEs1)
        """
        
        #=======================================================================
        # defaults
        #=======================================================================
        log, tmp_dir, out_dir, _, resname, write = self._func_setup('direct',  **kwargs)
        if aggscale is None: aggscale=self.aggscale
        start = now()
        #=======================================================================
        # downscale DEM an WD each
        #=======================================================================
        log.info('aggscale=%i on %s'%(aggscale, list(ds_d.keys())))
        res_d, ar_d = dict(), dict()
        for k, raw_ds in ds_d.items():
            if k=='wse':continue
            ds1 = self.resample(dataset=raw_ds, resampling=getattr(rio.enums.Resampling, resampleAlg), scale=1/aggscale)
            
            #load array (for wse calc)
            ar_d[k] = ds1.read(1, masked=False)
            
            #write it
            res_d[k] = self.write_memDataset(ds1, dtype=np.float32,
                       ofp=os.path.join(out_dir, '%s_%s.tif'%(k, self.obj_name)),masked=True,
                       logger=log)
            
        #=======================================================================
        # compute WSE
        #=======================================================================
        k='wse'
        #wse_ar = ma.array(ar_d['dem'] + ar_d['wd'], mask=ar_d['wd']<=0, fill_value=ds1.nodata)
        wse_ar = np.where(ar_d['wd']<=0, ds1.nodata, ar_d['dem'] + ar_d['wd']).astype(np.float32)
        
        del ar_d
        
        res_d[k] = self.write_array(wse_ar,  masked=False, ofp=os.path.join(out_dir, '%s_%s.tif'%(k, self.obj_name)),
                               logger=log, nodata=ds1.nodata,
                               transform=ds1.transform, #use the resampled from above
                               )
            
        #=======================================================================
        # wrap
        #=======================================================================
        log.info('finished downscaling and writing %i in %.2f secs'%(len(ds_d), (now()-start).total_seconds()))
        
        return res_d
            
    def agg_filter(self,
                         ds_d,
                         resampleAlg='average',
                         aggscale=None,
                         **kwargs):
        """fitlered agg of DEM and WSE. WD is recomputed."""
        
        #=======================================================================
        # defaults
        #=======================================================================
        log, tmp_dir, out_dir, _, resname, write = self._func_setup('filter',  **kwargs)
        if aggscale is None: aggscale=self.aggscale
        start = now()
        """
        
        load_array(ds_d['wse'])
        """
 
        #=======================================================================
        # downscale dem and wse
        #=======================================================================
        log.info('aggscale=%i on %s'%(aggscale, list(ds_d.keys())))
 
        wse_ds1 = self.resample(dataset=ds_d['wse'], resampling=getattr(rio.enums.Resampling, resampleAlg), scale=1/aggscale)

        wse_ar1 = load_array(wse_ds1).astype(np.float32)
        
        wse_ds1.close() #dont need this anymore
            
        dem_ds1 = self.resample(dataset=ds_d['dem'], resampling=getattr(rio.enums.Resampling, resampleAlg), scale=1/aggscale)
        dem_ar1 = load_array(dem_ds1).astype(np.float32)
        
        self._base_inherit(ds=dem_ds1) #set class defaults from this
        #=======================================================================
        # filter wse
        #=======================================================================
        wse_ar2 = wse_ar1.copy()
        np.place(wse_ar2, wse_ar1<=dem_ar1, np.nan)
        wd_ds1 = self.load_memDataset(wse_ar2, name='wd')
        
        #=======================================================================
        # subtract to get depths
        #=======================================================================
        wd_ar = np.nan_to_num(wse_ar2-dem_ar1, nan=0.0).astype(np.float32)
        wd_ds = self.load_memDataset(wd_ar, name='wd')
        
        #=======================================================================
        # write all
        #=======================================================================
        res_d=dict()
        for k, ds in {'dem':dem_ds1, 'wse':wd_ds1, 'wd':wd_ds}.items():
            res_d[k] = self.write_memDataset(ds, dtype=np.float32,
                       ofp=os.path.join(out_dir, '%s_%s.tif'%(k, self.obj_name)),
                       logger=log)
            
        #=======================================================================
        # wrap
        #=======================================================================
        log.info('finished downscaling and writing %i in %.2f secs'%(len(ds_d), (now()-start).total_seconds()))
        
        return res_d
        
    def write_dataset_d(self,
                        ar_d,
                        logger=None,
                        out_dir=None,
                         **kwargs):
        """helper for writing three main rasters consistently"""
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger=self.logger
        log=logger.getChild('write_dsd')
        if out_dir is None: out_dir=self.out_dir
        #=======================================================================
        # precheck
        #=======================================================================
        miss_l = set(['dem', 'wse', 'wd']).symmetric_difference(ar_d.keys())
        assert len(miss_l)==0
        
        log.info('writing %i to %s'%(len(ar_d), out_dir))
        
        """
        self.transform
        """
        
        #=======================================================================
        # loop and write
        #=======================================================================
        res_d = dict()
        for k,ar in ar_d.items():
            res_d[k] = self.write_array(ar, 
                       ofp=os.path.join(out_dir, '%s_%s.tif'%(k, self.obj_name)),
                       logger=log, **kwargs)
            
        #=======================================================================
        # wrap
        #=======================================================================
        return res_d
    
class RasterArrayStats(object):

 
 #==============================================================================
 #    def __init__(self,
 #                 engine='np', 
 #                 **kwargs):
 #        """methods for ocmputing raster stats on arrays
 #        
 #        Parameters
 #        ----------
 #        engine: str, default 'np'
 #            whether to use dask or numpy
 #        """
 # 
 #        #=======================================================================
 #        # build defaults
 #        #=======================================================================
 #        
 # 
 #        super().__init__(**kwargs)
 #        
 #        self.engine=engine
 #==============================================================================
        
        
    def _build_statFuncs(self, engine=None):
        """construct the dictinoary of methods"""
        if engine is None: engine=self.engine
        
        if engine=='np':
            d = {'wse':lambda ar, **kwargs:self._get_wse_stats(ar, **kwargs),
                 'wd':lambda ar, **kwargs:self._get_depth_stats(ar, **kwargs),
                 'diff':lambda ar, **kwargs:self._get_diff_stats(ar, **kwargs),
                }
            
        else:
            raise IOError('not implemented')
        
        #check
        miss_l = set(d.keys()).difference(coldx_d['layer'])
        assert len(miss_l)==0, miss_l
        
        self.statFunc_d = d
        
        
    #===========================================================================
    # TP----
    #===========================================================================
    def _get_TP_statsXR(self, xda,
                         agg_kwargs = dict(dim=('band', 'y', 'x'), skipna=True),
                         ):
        
        #check expectations
        assert_xda(xda)        
 
 
        
        return {            
            'mean':xda.mean(**agg_kwargs), #compute mean for each scale 
            'real_count':np.invert(np.isnan(xda)).sum(**agg_kwargs),
            'max':xda.mean(**agg_kwargs), #compute mean for each scale 
                }
            
        
        
    #===========================================================================
    # WSE--------
    #===========================================================================
    def _get_wse_stats(self, mar, **kwargs):
        
        assert_wse_ar(mar, masked=True)
        return {'mean': mar.mean()}
    
    def _get_wse_statsXR(self, xda,
                         agg_kwargs = dict(dim=('band', 'y', 'x'), skipna=True),
                         ):
        
        #check expectations
        assert_xda(xda)        
        assert np.isnan(xda.values).any()
 
        
        return {            
            'mean':xda.mean(**agg_kwargs), #compute mean for each scale 
            'real_count':np.invert(np.isnan(xda)).sum(**agg_kwargs),
            'var':xda.var(**agg_kwargs)
                }
 

    #===========================================================================
    # WD-------
    #===========================================================================
    def _get_depth_stats(self, mar, pixelArea=None):
 
        res_d=dict()
        #=======================================================
        # simple mean
        #=======================================================
        res_d['mean'] = mar.mean()
        #===================================================================
        # inundation area
        #===================================================================
        res_d['posi_area'] = np.sum(mar>0) * (pixelArea) #non-nulls times pixel area
        #===================================================================
        # volume
        #===================================================================
        res_d['vol'] = mar.sum() * pixelArea
        
        return res_d
    
    def _get_wd_statsXR(self, xda,
                         agg_kwargs = dict(dim=('band', 'y', 'x'), skipna=True),
                         ):
        
        #check expectations
        assert_xda(xda)
        
        return {
            
            'mean':xda.mean(**agg_kwargs), #compute mean for each scale 
            'posi_count':(xda>0).sum(**agg_kwargs),
            'sum':xda.sum(**agg_kwargs),
            'var':xda.var(**agg_kwargs)
                }

 
    
    #===========================================================================
    # DIFFERENCE-------
    #===========================================================================
 #==============================================================================
 #    def _get_diff_wd_stats(self, ar):
 #        """compute stats on difference grids.
 #        NOTE: always using reals for denometer"""
 #        assert isinstance(ar, ma.MaskedArray)
 #        assert not np.any(np.isnan(ar))
 #        
 #        
 #        
 #        #fully masked check
 #        if np.all(ar.mask):
 #            return {'meanErr':0.0, 'meanAbsErr':0.0, 'RMSE':0.0}
 #        
 # 
 #        res_d = dict()
 #        rcnt = (~ar.mask).sum()
 #        res_d['sum'] = ar.sum()
 #        res_d['meanErr'] =  res_d['sum']/rcnt #same as np.mean(ar)
 #        res_d['meanAbsErr'] = np.abs(ar).sum() / rcnt
 #        res_d['RMSE'] = np.sqrt(np.mean(np.square(ar)))
 #        return res_d
 #==============================================================================
    
    def _get_diff_wd_statsXR(self, xda,
                         agg_kwargs = dict(dim=('band', 'y', 'x'), skipna=True),
                         ):
        
        #check expectations
        assert_xda(xda) 
        
        return {            
            'mean':xda.mean(**agg_kwargs), #compute mean for each scale 
            'mean_abs':np.abs(xda).mean(**agg_kwargs), #compute mean for each scale 
            #'real_count':np.invert(np.isnan(xda)).sum(**agg_kwargs),
            'RMSE':np.sqrt(
                np.square(xda).sum(**agg_kwargs)
                ),
            'sum':xda.sum(**agg_kwargs),
            'posi_frac':(xda>0).astype(int).mean(**agg_kwargs)
                }
 
    def _get_diff_wse_statsXR(self, xda,
                         agg_kwargs = dict(dim=('band', 'y', 'x'), skipna=True),
                         ):
        
        #check expectations
        assert_xda(xda) 
        
        return {            
            'mean':xda.mean(**agg_kwargs), #compute mean for each scale 
            'mean_abs':np.abs(xda).mean(**agg_kwargs), #compute mean for each scale 
            #'real_count':np.invert(np.isnan(xda)).sum(**agg_kwargs),
            'RMSE':np.sqrt(
                np.square(xda).sum(**agg_kwargs)
                ),
            #'sum':xda.sum(**agg_kwargs),
            'posi_frac':(xda>0).astype(int).mean(**agg_kwargs)
                }
 
class UpsampleSession(Agg2Session, RasterArrayStats, UpsampleChild):
    """tools for experimenting with downsample sets"""
    
    def __init__(self,method='direct', scen_name=None, obj_name='haz',
                 dsc_l=None,
                 **kwargs):
 
        if scen_name is None: scen_name=method
        super().__init__(obj_name=obj_name,scen_name=scen_name, **kwargs)
        self.method=method
        self.dsc_l=dsc_l
        
        print('finished UpsampleSession.__init__')
        
 
        
    #===========================================================================
    # UPSAMPLING (aggregating)-----
    #===========================================================================
    def run_agg(self,demR_fp, wseR_fp,
 
                 dsc_l=None,
 
                 
                 method=None,
                 bbox=None,
 
                 **kwargs):
        """build downsample set
        
        Parameters
        -----------
        dem_fp: str
            filepath to DEM raster
        wse_fp: str
            filepath to WSE raster
        
        dsc_l: list, optional
            set of new resolutions to construct
            default: build from 

 
        """
        #=======================================================================
        # defaults
        #=======================================================================
        if method is None: method = self.method
        if bbox is None: bbox = self.bbox
        if dsc_l is None: dsc_l = self.dsc_l
        start = now()
        # if out_dir is None: out_dir=os.path.join(self.out_dir, method)
        log, tmp_dir, out_dir, ofp, resname, write = self._func_setup('agg', ext='.pkl', subdir=True, **kwargs)
        skwargs = dict(logger=log, tmp_dir=tmp_dir, out_dir=tmp_dir, write=write, bbox=bbox)
        
        log.info('for %i upscales using \'%s\' from \n    DEM:  %s\n    WSE:  %s' % (
            len(dsc_l), method, os.path.basename(demR_fp), os.path.basename(wseR_fp)))
            
        #=======================================================================
        # check layers
        #=======================================================================
        for layName, fp in {'dem':demR_fp, 'wse':wseR_fp}.items():
            assert_ds_attribute_match(fp, crs=self.crs, nodata=self.nodata, msg=layName) 
            
        #=======================================================================
        # check divisibility
        #=======================================================================
        max_aggscale = dsc_l[-1]
        if (not is_divisible(demR_fp, max_aggscale)) or (not bbox is None):
            log.warning('uneven division w/ %i... clipping' % max_aggscale)
            
            dem_fp = self.build_crop(demR_fp, divisor=max_aggscale, **skwargs)
            wse_fp = self.build_crop(wseR_fp, divisor=max_aggscale, **skwargs)
            
        else:
            dem_fp, wse_fp = demR_fp, wseR_fp
            
        #=======================================================================
        # build the set from this
        #=======================================================================
        res_lib = self.build_dset(dem_fp, wse_fp, dsc_l=dsc_l, method=method, out_dir=out_dir)
        
        log.info('finished in %.2f secs' % ((now() - start).total_seconds()))
        
        #=======================================================================
        # assemble meta
        #=======================================================================
 
        meta_df = pd.DataFrame.from_dict(res_lib).T.rename_axis('scale')
 
        # write the meta
        meta_df.to_pickle(ofp)
        log.info('wrote %s meta to \n    %s' % (str(meta_df.shape), ofp))
 
        return ofp
            
    def get_dscList(self,
 
                           reso_iters=3,
 
                           **kwargs):
        """get a fibonaci like sequence for aggscale multipliers
        (NOT resolution)
        
        Parameters
        ---------

        base_resolution: int, optional
            base resolution from which to construct the dsc_l
            default: build from dem_fp
            
        Returns
        -------
        dsc_l: list
            sequence of ints for aggscale. first is 1
            NOTE: every entry must be a divisor of the last entry (for cropping)
        
        """
        
        #=======================================================================
        # defaults
        #=======================================================================
        log, tmp_dir, out_dir, ofp, resname, write = self._func_setup('dscList',  **kwargs)
        
        l = [1]
        for i in range(reso_iters-1):
            l.append(l[i]*2)
        log.info('got %i: %s'%(len(l), l))
        return l
    
    def build_crop(self, raw_fp, new_shape=None, divisor=None, bbox=None, **kwargs):
        """build a cropped raster which achieves even division
        
        anchors to top left
        
        Parameters
        ----------
        bounds : optional
        
        divisor: int, optional
            for computing the nearest whole division
            
            
        Note
        ---------
        writing again to file hrere is not ideal
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
            raw_ds.shape
            """
 
            #=======================================================================
            # precheck
            #=======================================================================
            assert not is_divisible(raw_ds, divisor), 'no need to crop'
            
            #===================================================================
            # get starting window
            #===================================================================
            if not bbox is None:
                window1 = rio.windows.from_bounds(*bbox.bounds, transform=raw_ds.transform)
            else:
                window1 = rio.windows.from_bounds(*raw_ds.bounds, transform=raw_ds.transform)
                            
            #get the equivalent shape
            window1_shape = tuple(map(int,(window1.width, window1.height)))#num_rows, num_cols
            
             
            
            for k in window1_shape:
                assert k/divisor>=2
            #===================================================================
            # compute new_shape
            #===================================================================
            if new_shape is None: 
                new_shape = tuple([(d//divisor)*divisor for d in window1_shape])
                
            #===================================================================
            # build new window
            #===================================================================
            #rio.windows.crop(window1, new_shape[0], new_shape[1])
            #window1.crop(10, 10)
            #window2 = window1.crop(new_shape[1], new_shape[0])
            window2 = rio.windows.Window(window1.col_off, window1.row_off, *new_shape)
                
            #===================================================================
            # write to file
            #===================================================================
            log.info('cropping %s to %s for even divison by %i'%(
                raw_ds.shape, new_shape, divisor))
            
            self.crop(window2, dataset=raw_ds,ofp=ofp, logger=log)
            
        return ofp

    def _load_datasets(self, dem_fp, wse_fp, divisor=None, **kwargs):
        """helper to load WSe and DEM datasets with some checks and base setting"""
        dem_ds = self._base_set(dem_fp, **kwargs)
        _ = self._base_inherit()

        wse_ds = self.open_dataset(wse_fp, **kwargs)
        
        #precheck
        assert_rlay_simple(dem_ds, msg='dem')
        assert_extent_equal(dem_ds, wse_ds, msg='dem vs wse')
        if not divisor is None:
            assert is_divisible(dem_ds, divisor), 'passed DEM shape not evenly divisible (%i)' % divisor
        
        return dem_ds, wse_ds

    def build_dset(self,
            dem_fp, wse_fp,
            dsc_l=None,
            method='direct', resampleAlg='average',
            compress=None,
  
            out_dir=None,  
            **kwargs):
        """build a set of downsampled rasters
        
        Parameters
        ------------
        method: str default 'direct'
            downsample routine method
            
        compress: str, default 'compress'
            compression for outputs
            
        resampleAlg: str, default 'average'
            rasterio resampling method
            
        buildvrt: bool, default True
            construct a vrt of each intermittent raster (for animations)
            
        """
        #=======================================================================
        # defaults
        #=======================================================================
        start = now()
        log, tmp_dir, _, ofp, resname, write = self._func_setup('dsmp',  **kwargs)
        
        skwargs = dict(logger=log, write=write)
        
        #directories
        if out_dir is None: out_dir = os.path.join(self.out_dir, 'dset')
        if not os.path.exists(out_dir):os.makedirs(out_dir)
        
        log.info('building %i downsamples from \n    %s\n    %s'%(
            len(dsc_l)-1, os.path.basename(dem_fp), os.path.basename(wse_fp)))
        
        #=======================================================================
        # open base layers
        #=======================================================================
        dem_ds, wse_ds = self._load_datasets(dem_fp, wse_fp, divisor=dsc_l[-1],**skwargs)
        
        dem_ar = load_array(dem_ds)
        assert_dem_ar(dem_ar)
        
        wse_ar = load_array(wse_ds)
        assert_wse_ar(wse_ar)
        
        base_resolution = int(dem_ds.res[0])
        log.info('base_resolution=%i, shape=%s' % (base_resolution, dem_ds.shape))
        
        #=======================================================================
        # build base depth
        #=======================================================================
        wd_ar = np.nan_to_num(wse_ar-dem_ar, nan=0.0)
        
        #create a datasource in memory        
        wd_ds = self.load_memDataset(wd_ar, **skwargs)
        
        base_ar_d = {'dem':dem_ar, 'wse':wse_ar, 'wd':wd_ar}
        base_ds_d = {'dem':dem_ds, 'wse':wse_ds, 'wd':wd_ds}
        #=======================================================================
        # loop and build downsamples
        #=======================================================================
 
        res_lib=dict()
        for i, aggscale in enumerate(dsc_l):
            log.info('    (%i/%i) reso=%i'%(i, len(dsc_l), aggscale))
            
            with UpsampleChild(session=self,aggscale=aggscale, 
                                 crs=self.crs, nodata=self.nodata,transform=self.transform,
                                 compress=compress, out_dir=out_dir) as wrkr:
 
                #===================================================================
                # base/first
                #===================================================================
                """writing raw for consistency"""
                if i==0:
                    assert aggscale==1
                    res_lib[aggscale] = wrkr.write_dataset_d(base_ar_d, logger=log) 
                    continue
                
                #===============================================================
                # aggscale
                #===============================================================
                if method=='direct':
                    res_lib[aggscale] = wrkr.agg_direct(base_ds_d,resampleAlg=resampleAlg, **skwargs)
                elif method=='filter':
                    res_lib[aggscale] = wrkr.agg_filter(base_ds_d,resampleAlg=resampleAlg, **skwargs)
                else:
                    raise IOError('not implemented')
 
        #=======================================================================
        # wrap
        #=======================================================================
        log.info('finished on %i w/ method=%s in %.2f secs'%(
            len(res_lib), method, (now()-start).total_seconds()))
        
        return res_lib
    
    #===========================================================================
    # COMPILING---------
    #===========================================================================
    def run_vrts(self, pick_fp, 
 
                 #cols = ['dem', 'wse', 'wd', 'catMosaic'],
                 **kwargs):
        """this really isnt working well... should find a different format"""
        log, tmp_dir, out_dir, ofp, resname0, write = self._func_setup('vrt',  subdir=True, **kwargs)
 
        """
        view(dxcol_raw)
        self.out_dir
        meta_df.columns
        """
        
        log.info('compiling \'%s\' vrt from %s'%(resname0, os.path.basename(pick_fp))) 
        res_d = dict()
        
        for layName, col in df.items():  
            if not layName in coldx_d['layer']:
                continue
            #if not layName=='wse': continue 
            fp_d = col.dropna().to_dict()
            
            
            try:
                ofpi = self.build_vrts(fp_d,ofp = os.path.join(out_dir, '%s_%s_%i.vrt'%(resname0, layName,  len(fp_d))))
                
                log.info('    for \'%s\' compiled %i into a vrt: %s'%(layName, len(fp_d), os.path.basename(ofpi)))
                
                res_d['%s'%(layName)] = ofpi
            except Exception as e:
                log.error('failed to build vrt on %s w/ \n    %s'%(layName, e))
        #=======================================================================
        # for layer, gdx in dxcol_raw.groupby(level=0, axis=1):
        #     for coln, col in gdx.droplevel(0, axis=1).items():
        #         if not coln.endswith('fp'):
        #             continue
        #         fp_d = col.dropna().to_dict()
        #         ofpi = self.build_vrts(fp_d,ofp = os.path.join(out_dir, '%s_%s_%i.vrt'%(layer, coln, len(fp_d))))
        #         
        #         log.info('    for \'%s.%s\' compiled %i into a vrt: %s'%(layer, coln, len(fp_d), os.path.basename(ofpi)))
        #         
        #         res_d['%s_%s'%(layer, coln)] = ofpi
        #=======================================================================
        
        log.info('finished writing %i to \n    %s'%(len(res_d), out_dir))
        
        return res_d
    
    def build_vrts(self,fp_d,ofp):
        """build vrts of the results for nice animations"""
 
        #log, tmp_dir, out_dir, ofp, resname, write = self._func_setup('b', subdir=False, **kwargs)
 
        """
        help(gdal.BuildVRT)
        gdal.BuildVRTOptions()
        help(gdal.BuildVRTOptions)
        """
 
        #ofp = os.path.join(out_dir, '%s_%i.vrt'%(sub_dkey.replace('_fp', ''), len(d)))
        
        #pull reals
        fp_l = [k for k in fp_d.values() if isinstance(k, str)]
        for k in fp_l: assert os.path.exists(k), k
        
        gdal.BuildVRT(ofp, fp_l, separate=True, resolution='highest', resampleAlg='nearest')
        
        if not os.path.exists(ofp): 
            raise IOError('failed to build vrt')
            

        
        return ofp
    
    

            
    
    #===========================================================================
    # CASE MASKS---------
    #===========================================================================


    #===========================================================================
    # STATS-------
    #===========================================================================

    def run_stats(self, agg_fp, cm_fp,
 
                 layName_l = ['wse', 'wd'],
 
 
                 **kwargs):
        """
        compute global stats for each aggregated raster using each mask.
        mean(depth), sum(area), sum(volume)
        """
        #=======================================================================
        # defaults
        #=======================================================================
        log, tmp_dir, out_dir, ofp, resname, write = self._func_setup('stats',  subdir=True,ext='.pkl', **kwargs)
 
        df, start = self._rstats_init(agg_fp, cm_fp, layName_l, log)
        
        res_lib, meta_d =dict(), dict()
        #=======================================================================
        # compute for each scale
        #=======================================================================        
        for i, (scale, row) in enumerate(df.iterrows()):
            log.info(f'    {i+1}/{len(df)} on scale={scale}')
 
            #the complete mask
            with rio.open(row['wd'], mode='r') as ds:
                shape = ds.shape                    
                mask_d = {'all':np.full(shape, True)}
 
                pixelArea = ds.res[0]*ds.res[1]
                pixelLength=ds.res[0]
                
 
            fkwargs = dict(pixelArea=pixelArea)
            #build other masks
            if i>0:
                cm_ar = load_array(row['catMosaic']) 
                assert cm_ar.shape==shape
                
                #boolean mask of each category
                mask_d.update(self.mosaic_to_masks(cm_ar))
                
 
            #===================================================================
            # compute on each layer
            #===================================================================
            res_d1 = dict()
            for layName, fp in row.items():
                if layName=='catMosaic': continue
                logi = log.getChild('%i.%s'%(i, layName))
                
                #===============================================================
                # get stats func
                #===============================================================
                func = self.statFunc_d[layName]
                
                #===============================================================
                # load and compute
                #===============================================================
                log.debug('loading from %s'%fp)
                with rio.open(fp, mode='r') as ds:
                    ar_raw = load_array(ds, masked=True)                
 
                    d = self.get_maskd_func(mask_d, ar_raw, func, logi, **fkwargs)
 
                res_d1[layName] = pd.DataFrame.from_dict(d)
            
 
            #===============================================================
            # store
            #===============================================================
            """
            view(res_dx)
            """
            res_lib[scale] = pd.concat(res_d1, axis=1, names=['layer', 'dsc'])
            meta_d[scale] = {'pixelArea':pixelArea, 'pixelLength':pixelLength}                    
 
            
        #=======================================================================
        # wrap
        #=======================================================================
        res_dx = self._rstats_wrap(res_lib, meta_d, ofp)
        log.info('finished in %.2f wrote %s to \n    %s'%((now()-start).total_seconds(), str(res_dx.shape), ofp))
        
        return ofp
    
 
    def run_stats_fine(self, agg_fp, cm_fp, 
 
                 layName_l = ['wse', 'wd'],
 
 
                 **kwargs):
        """
        compute global stats on fine/raw rasters using cat masks
        """
        #=======================================================================
        # defaults
        #=======================================================================
        log, tmp_dir, out_dir, ofp, resname, write = self._func_setup('statsF',  subdir=True,ext='.pkl', **kwargs)
 
        df, start = self._rstats_init(agg_fp, cm_fp, layName_l, log)
        
        meta_d =dict()
        
        """because we loop in a differnet order from normal stats... need to precosntruct to match"""
        res_lib = {scale:{layName:dict() for layName in layName_l} for scale in df.index}
        #=======================================================================
        # compute for each layer
        #=======================================================================
        """runing different masks against the same layer... need a different outer loop"""
        for layName, fp in df.loc[1, layName_l].to_dict().items(): 
            #===================================================================
            # #load baseline layer
            #===================================================================
            with rio.open(fp, mode='r') as ds:
                #get baseline data
                
                ar_raw = load_array(ds, masked=True)                    
                shape = ds.shape
                                
                #scale info
                pixelArea = ds.res[0]*ds.res[1] #needed for scaler below
                
                metaF_d = get_pixel_info(ds)
                

            #===================================================================
            # loop on scale
            #===================================================================
            log.info('for \'%s\' w/ %s computing against %i scales'%(layName, str(ar_raw.shape), len(df)))
            
            
            #stat vars for this layer
            mask_full = np.full(shape, True)
            func = self.statFunc_d[layName]
            fkwargs = dict(pixelArea=pixelArea)
            

            #loop
            for i, (scale, row) in enumerate(df.iterrows()):
                #setup this scale
                logi = log.getChild('%i.%s'%(scale, layName))
                logi.debug(row.to_dict())
                
 
 
                def upd_meta(ds):
                    meta_d[scale] = get_pixel_info(ds)
                    
                #===============================================================
                # build baseline mask
                #===============================================================
                mask_d = {'all':mask_full} 
                if i==0:
                    meta_d[scale] = copy.deepcopy(metaF_d)
     
                #===================================================================
                # add other masks
                #===================================================================
                else:
                    """here we need to do wnscale"""
                    with rio.open(row['catMosaic'], mode='r') as ds:
                        cm_ar = ds.read(1, out_shape=shape, resampling=Resampling.nearest, masked=False)                        
                        upd_meta(ds)     
     
                    assert cm_ar.shape==shape
                    
                    mask_d.update(self.mosaic_to_masks(cm_ar))
                    
                    
                #===================================================================
                # compute stats function on each mask
                #=================================================================== 
                d = self.get_maskd_func(mask_d, ar_raw, func, logi, **fkwargs)
 
                res_lib[scale][layName] = pd.DataFrame.from_dict(d) 
                
            #===============================================================
            # wrap layer
            #===============================================================
            """handling this in the below reorient"""
 
            
        #=======================================================================
        # wrap
        #=======================================================================
        #re-orient container to match stats expectations
        d = dict()
        for scale, d1 in res_lib.items():
            d[scale] = pd.concat({l:df for l,df in d1.items()}, axis=1, names=['layer', 'dsc'])
        """
        view(res_dx)
        """
        res_dx = self._rstats_wrap(d, meta_d, ofp)
        log.info('finished in %.2f wrote %s to \n    %s'%((now()-start).total_seconds(), str(res_dx.shape), ofp))
        
        return ofp
        
        


    def get_maskd_func(self, mask_d, ar_raw, func, log, **kwargs):
        """apply each mask to the array then feed the result to the function
        
        
        Returns
        -------
        pre_count: int
            initial count of unmasked values on each mask
            
        post_count: int
            unmasked values after applying the mask (joining of data mask and filter mask)
            
        """
        
        log.debug('    on %s w/ %i masks'%(str(ar_raw.shape), len(mask_d)))
        res_lib = dict()
        
        for maskName, mask_ar in mask_d.items():
            """
            mask_ar=True: values we are interested in (opposite of numpy's mask convention)
            """
            log.info('     %s (%i/%i) on %s' % (maskName, mask_ar.sum(), mask_ar.size, str(mask_ar.shape)))
            res_d = {'pre_count':mask_ar.sum()}
            assert mask_ar.shape==ar_raw.shape
            #===============================================================
            # construct masked array
            #===============================================================
            mar = ma.array(1, mask=True) #dummy all invalid
            if np.any(mask_ar):
                #===============================================================
                # apply the mask
                #===============================================================
                #update an existing mask
                if isinstance(ar_raw, ma.MaskedArray):
                    #ar_raw.harden_mask() #make sure we don't unmask anything
                    mar = ar_raw.copy()
                    
                    if not np.all(mask_ar):                        
                        mar[~mask_ar] = ma.masked 
 
                    
                #construct mask from scratch 
                else:
                    mar = ma.array(ar_raw, mask=~mask_ar) #valids=True
                    
            #===============================================================
            # #execute the stats function
            #===============================================================
            res_d['post_count'] = (~mar.mask).sum()
            
            if res_d['post_count']>0:
                if  __debug__:
                    log.debug('    %i/%i valids and kwargs: %s'%(
                        res_d['post_count'], mar.size, kwargs))
                    
                res_d.update(func(mar, **kwargs))
            
            #===================================================================
            # everything invalid
            #===================================================================
            else:
                log.warning('%s got no valids' % (maskName))
            
            #===================================================================
            # wrap
            #=================================================================== 
            res_lib[maskName] = res_d #store
        
        log.debug('    finished on %i masks'%len(res_lib))
        return res_lib
    
    def _rstats_wrap(self, res_lib, meta_d, ofp):
        """post for rstat funcs
        Parameters
        -----------
        res_lib, dict
            {scale:dxcol (layer, dsc)}
        """
 
        res_dx = pd.concat(res_lib, axis=0, names=['scale', 'metric']).unstack()
        
        #ammend commons to index
        if not meta_d is None:
            mindex = pd.MultiIndex.from_frame(
                res_dx.index.to_frame().reset_index(drop=True).join(pd.DataFrame.from_dict(meta_d).T.astype(int), on='scale'))
            res_dx.index = mindex
            
        #sort and clean
        res_dx = res_dx.sort_index(axis=1, sort_remaining=True).dropna(axis=1, how='all')
        
 
        #checks
        assert not res_dx.isna().all().all()
        assert_dx_names(res_dx)
        
        #write
        res_dx.to_pickle(ofp)
        return res_dx
 


    def _rstats_init(self, agg_fp, cm_fp, layName_l, log):
        """init for rstat funcs"""
        start = now()
 
        self._build_statFuncs()
        
        #=======================================================================
        # load data
        #=======================================================================
        #run_agg
        df_raw = pd.read_pickle(agg_fp).loc[:, layName_l]
        
        
        #join data from catMasks
        cm_ser = pd.read_pickle(cm_fp)['fp'].rename('catMosaic')
        df = df_raw.join(cm_ser)
 
 
        log.info('computing stats on %s' % str(df.shape))
        return df, start
        
    #===========================================================================
    # ERRORS-------
    #===========================================================================

                
        
            
            
 
            
 
        
        
    def run_diffs(self,
                  pick_fp,
                  confusion=True,
                  **kwargs):
        
        log, tmp_dir, out_dir, ofp, resname, write = self._func_setup('diffs',  subdir=True,ext='.pkl', **kwargs)
        start = now()
        
        layName_l = ['wse', 'wd']
        #=======================================================================
        # load paths
        #=======================================================================
        df_raw = pd.read_pickle(pick_fp).loc[:, layName_l]
 
        #=======================================================================
        # loop and build diffs for each layer
        #=======================================================================
        res_d = dict()
        for layName, col in df_raw.items():
            
            fp_d = col.to_dict()
        
            #===================================================================
            # run
            #===================================================================
            res_d[layName] = self.get_diffs(fp_d, out_dir=os.path.join(out_dir, layName),
                                            resname=layName, logger=log.getChild(layName),
                                            dry_val={'wse':-9999, 'wd':0.0}[layName],
                                            confusion=confusion)
            
        #=======================================================================
        # wrap
        #=======================================================================
        res_dx = pd.concat(res_d, axis=1, names=['layer']).rename_axis(df_raw.index.name)
        res_dx.to_pickle(ofp)
        
        
        log.info('finished on %s in %.2f secs wrote to\n    %s'%(str(res_dx.shape), (now()-start).total_seconds(), ofp))
        
        return ofp
    
    
    def get_diffs(self,fp_d, 
                  write=True,
                  dry_val=-9999,
                  confusion=False,
                   **kwargs):
        """build difference grids for each layer respecting dry masks
        
        
        TODO:
        use xarray and parallelize the delta?
        """
        
        log, tmp_dir, out_dir, _, resname, write = self._func_setup('g', subdir=False, ext='.pkl',write=write, **kwargs)

        #=======================================================================
        # load
        #=======================================================================
        
        log.info('on %i' % len(fp_d))
 
        #===================================================================
        # baseline
        #===================================================================
        base_fp = fp_d[1]
        log.info('from %s' % (os.path.basename(base_fp)))
        
        def read_ds(ds, **kwargs):
            """custom nodata loading"""
            ar_raw = ds.read(1, masked=True, **kwargs) 
            
            #apply the custom mask
            if ds.nodata!=dry_val:
                ar = ma.masked_array(ar_raw.data, mask = ar_raw.data==dry_val, fill_value=-9999)
            else:
                ar = ar_raw
                
            return ar
 
        #===================================================================
        # #load baseline
        #===================================================================
        with rio.open(base_fp, mode='r') as ds: 
            assert ds.res[0] == 1
            base_ar = read_ds(ds) 
            self._base_inherit(ds=ds)
            
        #handle non-native nulls
        """for consistency, treating wd=0 as null for the difference calcs"""

            
        assert base_ar.mask.shape == base_ar.shape        #get the exposure mask
        wets = ~base_ar.mask
        #===================================================================
        # loop on reso
        #===================================================================
        res_d, res_cm_d = dict(), dict()
        for i, (scale, fp) in enumerate(fp_d.items()):
            log.info('    %i/%i scale=%i from %s'%(i+1, len(fp_d), scale, os.path.basename(fp)))
        
            #===============================================================
            # vs. base (no error)
            #===============================================================
            if i==0: 
                res_ar = ma.masked_array(np.full(base_ar.shape, 0), mask=base_ar.mask, fill_value=-9999)
                
                fine_ar = base_ar #for cofusion
 
                
            #===============================================================
            # vs. an  upscale
            #===============================================================
            else:
                #get disagg
                with rio.open(fp, mode='r') as ds:
                    fine_ar = read_ds(ds, out_shape=base_ar.shape, resampling=Resampling.nearest) 
 
                    assert fine_ar.shape==base_ar.shape
                    
                #compute errors
                res_ar = fine_ar - base_ar
 
                
            #===================================================================
            # confusion
            #===================================================================
            if confusion:
                """positive = wet"""
                cm_ar = confusion_matrix(wets.ravel(), ~fine_ar.mask.ravel(),labels=[False, True]).ravel()
     
     
                res_cm_d[scale] =  pd.Series(cm_ar,index = ['TN', 'FP', 'FN', 'TP'])
                
            else:
                res_cm_d[scale] = pd.Series()
            #===============================================================
            # write
            #===============================================================
            assert isinstance(res_ar, ma.MaskedArray)
            assert not np.any(np.isnan(res_ar))
            assert not np.all(res_ar.mask)
        
            if write:
                res_d[scale] = self.write_array(res_ar, 
                                                ofp=os.path.join(out_dir, '%s_diff_%03i.tif'%(resname, scale)), 
                                            logger=log.getChild(f'{scale}'), masked=True)
            else:
                res_d[scale] = np.nan 
            
 
            
        #===================================================================
        # wrap  
        #===================================================================
        
        res_df = pd.concat(res_cm_d, axis=1).T.join(pd.Series(res_d).rename('diff_fp')).rename_axis('subdata', axis=1)
            
 
        #=======================================================================
        # #write
        #=======================================================================
        
        log.debug('finshed on %s'%str(res_df.shape))
        
        
        return res_df
    
 


    def run_diff_stats(self,pick_fp, cm_pick_fp, **kwargs):
        """compute stats from diff rasters filtered by each cat mask
        
        
        Notes
        -----------
        These are all at the base resolution
        
        speedups?
            4 cat masks
            2 layers
            ~6 resolutions
            
            
            distribute to 8 workers: unique cat-mask + layer combinations?
            
        calculating too many stats here. really only care about
            wd vol diff
            wd and wse mean diff
            
        
        """
        #=======================================================================
        # defaults
        #=======================================================================
        log, tmp_dir, out_dir, ofp, resname, write = self._func_setup('diffStats',  subdir=True,ext='.pkl', **kwargs)
        start = now()
        layName_l = ['wse', 'wd']
        
        #=======================================================================
        # load data
        #=======================================================================
        #layers
        df_raw = pd.read_pickle(pick_fp)
        
        df = df_raw.loc[:, idx[layName_l, 'diff_fp']].droplevel('subdata', axis=1)
        
        #catMasks
        cm_df = pd.read_pickle(cm_pick_fp)        
        df = df.join(cm_df['fp'].rename('catMosaic'))
            
        assert df.isna().sum().sum()==1
        
        #===================================================================
        # loop on each scale
        #===================================================================
        res_lib, meta_d=dict(), dict()
        for i, (scale, ser) in enumerate(df.iterrows()):
            log.info('    %i/%i scale=%i'%(i+1, len(df), scale))
            
            
            #===================================================================
            # setup masks
            #===================================================================
            #the complete mask
            if i==0:
                """just using the ds of the raw wse for shape"""
                with rio.open(ser['wse'], mode='r') as ds:
                    shapeF = ds.shape                    
                    meta_d[scale] = get_pixel_info(ds)
 
                mask_d = {'all':np.full(shapeF, True)} #persists in for loop
                
            #build other masks
            else: 
                with rio.open(ser['catMosaic'], mode='r') as ds:
                    cm_ar = ds.read(1, out_shape=shapeF, resampling=Resampling.nearest, masked=False) #downscale
                    meta_d[scale] = get_pixel_info(ds)
                    
                assert cm_ar.shape==shapeF            
                mask_d.update(self.mosaic_to_masks(cm_ar))
            
            #=======================================================================
            # loop on each layer
            #=======================================================================
            res_d = dict()
            for layName, fp in ser.drop('catMosaic').items():
 
                log.debug('on \'%s\''%layName)

                    
                #===============================================================
                # compute metrics
                #===============================================================
                with rio.open(fp, mode='r') as ds:                    
                    if i==0:
                        rd1 = {'all':{'count':ds.width*ds.height, 'meanErr':0.0, 'meanAbsErr':0.0, 'RMSE':0.0}}

                    else:    
                        ar = ds.read(1, masked=True)    
                        
                        func = lambda x:self._get_diff_stats(x)
                        rd1 = self.get_maskd_func(mask_d, ar, func, log.getChild('%i.%s'%(i, layName)))
                    
 
                #===============================================================
                # wrap layer
                #===============================================================
                res_d[layName] = pd.DataFrame.from_dict(rd1)
            #===================================================================
            # wrap layer loop
            #===================================================================
            res_lib[scale] = pd.concat(res_d, axis=1, names=['layer', 'dsc'])   
            #pd.DataFrame.from_dict(res_d).T.astype({k:np.int32 for k in confusion_l})
        #=======================================================================
        # wrap on layer
        #=======================================================================
        """
        view(res_dx)
        """
 
        
        res_dx = pd.concat(res_lib, axis=0, names=['scale', 'metric']).unstack()        
        
        # add confusion
        cm_dx = df_raw.drop('diff_fp', axis=1, level=1)
        cm_dx.columns.set_names('metric', level=1, inplace=True)
        # cm_dx = cm_df.drop('fp', axis=1).rename_axis('metric', axis=1)
 
        res_dx = res_dx.join(
            pd.concat({'all':cm_dx}, names=['dsc'], axis=1).reorder_levels(res_dx.columns.names, axis=1)
            )
 
        # sort and clean
        res_dx = res_dx.sort_index(axis=1, sort_remaining=True).dropna(axis=1, how='all')
        
        #append pixel info to index
        """to conform with other stats objects"""
        mindex = pd.MultiIndex.from_frame(
                res_dx.index.to_frame().reset_index(drop=True).join(pd.DataFrame.from_dict(meta_d).T.astype(int), on='scale'))
        res_dx.index = mindex
 
        # checks
        assert not res_dx.isna().all().all()
        assert_dx_names(res_dx)
        
        # write
        res_dx.to_pickle(ofp)        
        
        log.info('finished on %s in %.2f secs and wrote to\n    %s' % (str(res_dx.shape), (now() - start).total_seconds(), ofp))
        
        return ofp
    
    def concat_stats(self, fp_d, **kwargs):
        """quick combining and writing of stat pickls"""
        log, tmp_dir, out_dir, ofp, resname, write = self._func_setup('_smry',  subdir=True,ext='.pkl', **kwargs)
        
        #=======================================================================
        # concat all the picks
        #=======================================================================
        d = dict()
        for base, fp in fp_d.items():
            if not fp.endswith('.pkl'): continue 
            d[base] = pd.read_pickle(fp)
 
            assert_dx_names(d[base], msg=base)
        
        dx = pd.concat(d, axis=1, names=['base'])
        
        #=======================================================================
        # wrap
        #=======================================================================
        dx.to_pickle(ofp)
        log.info(f'wrote xls {str(dx.shape)} to \n    {ofp}')
        
        if write:
            ofp1 = os.path.join(out_dir, f'{resname}_{len(dx)}_stats.xls')
            with pd.ExcelWriter(ofp1) as writer:       
                dx.to_excel(writer, sheet_name='stats', index=True, header=True)
                
            log.info(f'wrote {str(dx.shape)} to \n    {ofp1}')
                
        return ofp
    
#===============================================================================
#     def run_pTP(self, nc_fp, cm_pick_fp, layName_l=['wse'], **kwargs):
#         #=======================================================================
#         # defaults
#         #=======================================================================
#         log, tmp_dir, out_dir, ofp, resname, write = self._func_setup('pTP',  subdir=True,ext='.pkl', **kwargs)
#         
# 
#         
#         #=======================================================================
#         # execute on the dataset
#         #=======================================================================
#         log.info('from %s'%nc_fp)
#         with xr.open_dataset(nc_fp, engine='netcdf4',chunks='auto' ,decode_coords="all") as ds:
#  
#             log.info(f'loaded dims {list(ds.dims)}'+
#                      f'\n    {list(ds.coords)}'+
#                      f'\n    {list(ds.data_vars)}' +
#                      f'\n    chunks:{ds.chunks}')
#  
#             #===================================================================
#             # loop on each layer
#             #===================================================================
#             for layName in layName_l:
#                 self.get_pTP(ds[layName], logger=log.getChild(layName), resname=layName, out_dir=out_dir, write=write)
#                 
#     def get_pTP(self, xar,crs=None, **kwargs):
#         """get the s1_expo_cnt/s2_expo_cnt fraction for each scale
#         
#         variable shapes... dont want to collapse into DataArray
#         
#         could write a tiff or 1 dim netcdf for each?
#         
#         opted to compute the stat directly 
#         
#         """
#         #=======================================================================
#         # defautls
#         #=======================================================================
#         log, tmp_dir, out_dir, ofp, resname, write = self._func_setup('gpTP', **kwargs)
#         if crs is None: crs=self.crs
#  
#         #convert to boolean (True=exposed)
#         stack_xar1 = np.invert(np.isnan(xar))
#         
#         #=======================================================================
#         # get baseline values
#         #=======================================================================
#         ofp_d = dict()
#         delay_ofp_l = list()
#         d = dict()
#         scale_l = xar['scale'].values.tolist()
#         for i, scale in enumerate(scale_l):
#             log.info(f'    {i+1}/{len(scale_l)}')
#             if i==0:        
#                 #get the baseline
#                 base_xar = stack_xar1.isel(scale=0)
#  
#             #===================================================================
#             # compute the fine exposures (at s2)
#             #===================================================================
#             s1_expo_xar_s2 = base_xar.coarsen(dim={'x':scale, 'y':scale}, boundary='exact').sum()
#             
#             #update the scal
#             def clean(xar): 
#                 xar = xar.reset_coords(names=['scale'], drop=True)
#                 xar.attrs['scale'] = scale
#                 return xar
#             
#             s1_expo_xar_s2 = clean(s1_expo_xar_s2)
#             
#             assert (base_xar.shape[1]//scale, base_xar.shape[2]//scale) ==s1_expo_xar_s2.shape[1:]
#             
#             #===================================================================
#             #retrieve coarse exposures
#             #===================================================================
#             #decimate and scale
#             s2_expo_xar_s2 = clean(stack_xar1.sel(scale=scale).coarsen(dim={'x':scale, 'y':scale}, boundary='exact').max()*(scale**2))
#             
#  
#             #===================================================================
#             # compute ratio
#             #===================================================================
#             s12_expo_xar_s2 = s1_expo_xar_s2/s2_expo_xar_s2
#             
#             #===================================================================
#             # output
#             #===================================================================
#             if write:
#                 ofpi = os.path.join(out_dir, f's12expoFrac_{resname}_{scale:03d}.tif')
#                 
#                 s12_expo_xar_s2.rio.write_crs(crs, inplace=True)
#                 
#                 #append to the que
#                 delay_ofp_l.append(
#                     s12_expo_xar_s2.rio.to_raster(ofpi, compute=False, lock=None, crs=crs, nodata=-9999)
#                     )
#                 
#                 ofp_d[scale] = ofpi
#                 
#             #===================================================================
#             # zonal stat
#             #===================================================================
#             s12_expo_xar_s2
#             
#             """
#             s12_expo_xar_s2.plot()
#             """
#  
#             
#  
#             
#         #concat
#         """"""
#         log.info(f'writing {len(delay_ofp_l)} files to disk')
#         with ProgressBar():
#             #res_xar = xr.merge(d.values()).compute()
#             dask.compute(*delay_ofp_l)
#         
#         return 
#===============================================================================
 

class UpsampleSessionXR(UpsampleSession):
    def __init__(self,xr_dir=None,**kwargs):
        """session for xarray baseed hazard agg
        
        Parameters
        -----------
        xr_dir: str
            directory to place nc files from xarray
        """
 
 
        super().__init__(**kwargs)
        
        if xr_dir is None:
 
            
            xr_dir = os.path.join(self.out_dir, '_xr')
        if not os.path.exists(xr_dir):
            os.makedirs(xr_dir)
                
        self.xr_dir=xr_dir
        
        print('finished UpsampleSessionXR.__init__')
        


    def _prep_xds_write(self, xds_res):
        #append spatial info
        xds_res.rio.write_crs(self.crs.to_string(), inplace=True).rio.write_coordinate_system(inplace=True)
    #add meta
        for attn in ['today_str', 'run_name', 'proj_name']:
            xds_res.attrs[attn] = getattr(self, attn)
        
    #do some checks
        assert_xds(xds_res)

    def _save_mfdataset(self, xds_res,  resname, log, xr_dir=None):
        """write xds in scale and data_var chunks"""
        #=======================================================================
        # defaults
        #=======================================================================
        if xr_dir is None: xr_dir=self.xr_dir
        
        self._prep_xds_write(xds_res)
        
        #=======================================================================
        # prepare groups by data_var
        #=======================================================================
        ds_l, ofp_l = list(), list()
        for dataName, xda in xds_res.data_vars.items():
 
            #get the subdir
            odi = os.path.join(xr_dir, dataName)
            if not os.path.exists(odi):os.makedirs(odi)
 
            #split along scales
            si_l, dsi_l = zip(*xda.to_dataset().groupby('scale'))
            ofpi_l = [os.path.join(odi, f'{resname}_s{k:03d}.nc') for k in si_l]
            
 
            
            ds_l = ds_l + list(dsi_l)
            ofp_l = ofp_l + ofpi_l
            
        #=======================================================================
        # write the set
        #=======================================================================
        log.info(f'save_mfdataset on {len(ds_l)} to {xr_dir}')
        o = xr.save_mfdataset(ds_l, ofp_l, mode='w', format='NETCDF4', engine='netcdf4', compute=False)
        #o = xds_res.to_netcdf(path=ofp, mode='w', format ='NETCDF4', engine='netcdf4', compute=False)
        
        
            
        with ProgressBar():
            _ = o.compute()
        return ofp_l

    def build_downscaled_aggXR(self, fp_df, 
                                    layName_l=[
                                        'wse', 
                                        'wd'], **kwargs):
        """compile an aggregated stack (downsampled) into an xarray
        
        here we just combine the aggs... would be better to integrate like we did with catMasks
        """
        
        #=======================================================================
        # defaults
        #=======================================================================
        xr_dir = self.xr_dir
        log, tmp_dir, out_dir, ofp, resname, write = self._func_setup('aggXR',subdir=False, ext='.nc', **kwargs)
        start = now()
 
        
        #=======================================================================
        # loop and load
        #=======================================================================
        log.info('loading %s to xarrayDS'%(str(fp_df.shape)))
        res_lib = dict()
        ofp_l=list()
        cnt=0
        for layName, row in fp_df.items():
            log.info(f'layer {layName}')
            if not layName in layName_l: continue
            odi = os.path.join(xr_dir, layName)
            if not os.path.exists(odi):os.makedirs(odi)
            
            
            
            d = dict()
            for i, (scale, fp) in enumerate(row.to_dict().items()):
                log.info(f'    {i+1}/{len(fp_df)} on {layName} from {fp}')
                #get base values for downsampling
                if i==0:
                    with rio.open(fp, model='r') as ds:
                        base_shape = ds.shape
                        base_res = ds.res[0]
                        crs = ds.crs
                        
                        #=======================================================
                        # ar = ds.read(1)
                        # ar.max()
                        #=======================================================
                        
                    rio_open_kwargs = dict(shape=base_shape, resampling=Resampling.nearest)
     
                #===================================================================
                # #load datasource
                #===================================================================
                xda = rioxarray.open_rasterio(fp,masked=True, chunks='auto')            
                
                assert scale ==xda.rio.resolution()[0]
     
                log.info(f'for scale = {scale} loaded {xda.dtype.name} raster {xda.shape}  w/\n' +
                         f'    crs {xda.rio.crs} nodata {xda.rio.nodata} and bounds {xda.rio.bounds()}\n' +
                         f'    from {fp}')
                
                """
                xda.max().values
                    array(26.19882965)
                    
                    
                
                xda.plot()
                """
                
                #===================================================================
                # resample
                #===================================================================
                if not i==0:
                    xda1 = xda.rio.reproject(xda.rio.crs, **rio_open_kwargs)
                    xda.close() #not sure this makes a difference
                else:
                    xda1 = xda
                    
                #===================================================================
                # wrap
                #===================================================================
                assert xda1.rio.shape == base_shape
                assert xda1.rio.resolution()[0] == base_res
                
                #write this layer+scale
                xds_i= xr.concat([xda1], pd.Index([scale], name='scale', dtype=int)).to_dataset(name=layName) 
                self._prep_xds_write(xds_i)
                ofpi = os.path.join(odi, f'{resname}_{layName}_s{scale:03d}.nc')
                
                log.info(f'         writing {xda1.rio.shape} to {ofpi}')
                xds_i.to_netcdf(ofpi, mode='w', format='NETCDF4', engine='netcdf4', compute=True)
                
                #===============================================================
                # wrap
                #===============================================================
                ofp_l.append(ofpi)
                del xda, xda1, xds_i
                
                
                cnt+=1
            gc.collect()
            #===================================================================
            #     d[scale] = xda1
            #     
            # 
            # log.info(f'concat {layName}')
            # xds_i= xr.concat(d.values(), pd.Index(d.keys(), name='scale', dtype=int)).to_dataset(name=layName)
            # ofp_l+=self._save_mfdataset(xds_i, resname, log, xr_dir=xr_dir)
            #===================================================================
            
        #=======================================================================
        # #merge
        #=======================================================================
        #join datasets together on a new axis
 
        
        """    
        xds_res.plot(col='scale')
        """
        #promote to dataset
        log.info('compiled %i datasets'%cnt)
        #xds_res = xr.Dataset(res_lib)
 
        #=======================================================================
        # write batch
        #=======================================================================
        #ofp_l = self._save_mfdataset(xds_res, resname, log, xr_dir=xr_dir)
 
        log.info(f'finished in {(now()-start).total_seconds():.2f} secs  '+
 
                 f'\n    {len(ofp_l)} files written to {xr_dir}')
        
        return xr_dir
    
    def run_catMasksXR(self, dem_fp, wse_fp,dsc_l=None,
                     write_tif=False,xr_dir=None,
                    **kwargs):
        """build the dsmp cat mask for each reso iter
        
        Parametrs
        ---------
        demR_fp: str
            filepath to pre-clipped DEM raster
            
            
        Notes
        ---------
        no need to window as here we load the pre-processed rasters
        """
        
        #=======================================================================
        # defaults
        #=======================================================================
        start = now()
        idxn = self.idxn
        log, tmp_dir, out_dir, ofp, resname, write = self._func_setup('cMasks',subdir=True, ext='.pkl', **kwargs)
        if dsc_l is None: dsc_l=self.dsc_l
 
        
        
        #=======================================================================
        # load base layers-----
        #=======================================================================
        #=======================================================================
        # dem
        #=======================================================================
        dem_ds, wse_ds = self._load_datasets(dem_fp, wse_fp)
        rtransform = dem_ds.transform
        rshape = dem_ds.shape
        
        log.info(f'')
        dem_ds.close() 
        
        #dem_ar = load_array(dem_ds, masked=True)        
                
        
        wse_ar = load_array(wse_ds, masked=True)
        assert_wse_ar(wse_ar, masked=True)
 
        wse_ds.close()
        
        #load the dem as an xarray
        dem_xda = rioxarray.open_rasterio(dem_fp,  masked=False)
        assert_dem_ar(dem_xda.values, masked=False)
 
        #=======================================================================
        # build for each
        #=======================================================================
        log.info(f'on {len(dsc_l)-1} from a base shape of {rshape} height: {dem_ds.height} width: {dem_ds.width}')
        res_d=dict()
        meta_lib = dict()
        ofp_d = dict()
        for i, scale in enumerate(dsc_l): 
            if i==0: #no catmasks on the baseline                
                continue           
            #===================================================================
            # defaults
            #===================================================================
            iname = '%03d'%scale
            skwargs = dict(out_dir=tmp_dir, logger=log.getChild(iname), tmp_dir=tmp_dir) 
 
            #===================================================================
            # classify
            #===================================================================
            """build the cat mask at s2"""
            log.info('    (%i/%i) aggscale=%i'%(i+1, len(dsc_l), scale)) 
            with ResampClassifier(session=self, aggscale = scale,  **skwargs) as wrkr:
                #build each mask
                cm_d = wrkr.get_catMasks2(wse_ar=wse_ar, dem_ar=dem_xda.values[0])
                
                #build the mosaic
                cm_ar = wrkr.get_catMosaic(cm_d)
                
                #compute some stats
                stats_d = wrkr.get_catMasksStats(cm_d)
                
                #update
                #res_d[scale], meta_lib[scale] = cm_ar, stats_d
                meta_lib[scale] = stats_d
                
            #===================================================================
            # check
            #===================================================================
            assert tuple([v/scale for v in rshape])==cm_ar.shape
                
            #===================================================================
            # write
            #===================================================================
            #new transform
 
            
            transform_i = rtransform * rtransform.scale(scale)
            
            #===================================================================
            # transform_i = dem_ds.transform * dem_ds.transform.scale(
            #                     (dem_ds.width / cm_ar.shape[-1]),
            #                     (dem_ds.height / cm_ar.shape[-2])
            #                 )
            #===================================================================
                
            if write_tif:                        
                ofp_d[scale] = self.write_array(cm_ar, logger=log,ofp=os.path.join(out_dir, 'catMosaic_%03i.tif'%scale), transform=transform_i)
                
            #===================================================================
            # xarrayd
            #===================================================================
            """
            #xarray from geotiff
            xda_test = rioxarray.open_rasterio(ofp_d[scale],masked=True)
            xda_test['x'].values
 
            """
 
            
            #calculate the sptial dimensions
            x_ar, y_ar = get_xy_coords(transform_i, cm_ar.shape)
 
            
            #build a RasterArray
            xda = xr.DataArray(np.array([cm_ar]), 
                               coords={'band':[1],  'y':y_ar, 'x':x_ar} #order is important
                               #coords = [[1],  ys, xs,], dims=["band",  'y', 'x']
                               #).rio.write_transform(transform_i
                               ).rio.write_nodata(dem_xda.rio.nodata, inplace=True
                              ).rio.set_crs(dem_xda.rio.crs, inplace=True)
                              
 
            assert xda.rio.transform() == transform_i
            
            #downscale to match the raw
            res_d[scale] = xda.rio.reproject_match(dem_xda, resampling=Resampling.nearest) 
                        
        log.info('finished building %i dsc mask mosaics. writing to \n    %s'%(len(res_d), ofp))
        
        #=======================================================================
        # concat
        #=======================================================================
        xda_cat = xr.concat(res_d.values(), pd.Index(res_d.keys(), name='scale', dtype=int))
        xds_res = xr.Dataset({'catMosaic':xda_cat})
        
        
        
        """    
        xda_cat.plot(col='scale')
        """
        ofp_l = self._save_mfdataset(xds_res, resname, log, xr_dir=xr_dir)
            
        
        #=======================================================================
        # #assemble meta
        #=======================================================================
        dx = pd.concat({k:pd.DataFrame.from_dict(v) for k,v in meta_lib.items()})
 
        #just the sum
        meta_df = dx.loc[idx[:, 'sum'], :].droplevel(1).astype(int).rename_axis(idxn)

        meta_df = meta_df.join(pd.Series(ofp_d, dtype=str).rename('fp'), on=idxn)
            
        #=======================================================================
        # write meta
        #=======================================================================\
 
        meta_df.to_pickle(ofp)
 
        log.info('finished in %.2f secs and wrote %s to \n    %s'%((now()-start).total_seconds(), str(meta_df.shape), ofp))
        
        return ofp_l, ofp
    
 #==============================================================================
 #    def run_merge_XR(self, fp_l, **kwargs):
 #        """merge the catMosaic and layer datasets
 #        
 #        
 #        better to just write files in parallel"""
 #        log, tmp_dir, out_dir, ofp, _, write = self._func_setup('mXR',subdir=True, ext='.nc', **kwargs)
 #        idxn = self.idxn
 #        start = now()
 #        assert isinstance(fp_l, list)
 #        
 #        log.info(f'opening and combining {len(fp_l)}\n' + '\n    '.join(fp_l))
 #        
 #        """
 #        xr.open_dataset(fp1)
 #        xr.open_dataset(fp2)
 #        """
 #        
 #        with xr.open_mfdataset(fp_l, parallel=True,  engine='netcdf4',
 #                               data_vars='minimal', coords=idxn, combine="by_coords",
 #                               decode_coords="all",
 #                               ) as ds:
 #            
 #            log.info(f'loaded {ds.dims}'+
 #                 f'\n    coors: {list(ds.coords)}'+
 #                 f'\n    data_vars: {list(ds.data_vars)}'+
 #                 f'\n    crs:{ds.rio.crs}'
 #                 )
 #            
 #            for attn in ['today_str', 'run_name', 'proj_name', 'case_name', 'scen_name']:
 #                ds.attrs[attn] = getattr(self, attn)
 #                
 #            ds.attrs['fname'] = 'run_merge_XR'
 # 
 #            o = ds.to_netcdf(path=ofp, mode='w', format ='NETCDF4', engine='netcdf4', compute=False)
 #        
 #            with ProgressBar():
 #                _ = o.compute()
 #                
 #        #=======================================================================
 #        # wrap
 #        #=======================================================================
 #        log.info(f'wrote {len(fp_l)} to file in {(now()-start).total_seconds():.2f}\n    {ofp}')
 #        return ofp
 #==============================================================================
            

    
 #==============================================================================
 # 
 #    def run_diffsXR(self,
 #                  nc_fp,
 #                  xr_dir=None,
 #                  crs=None,
 #                  **kwargs):
 #        
 #        log, tmp_dir, out_dir, ofp, resname, write = self._func_setup('diffsXR',  subdir=True,ext='.nc', **kwargs)
 #        start = now()
 #        idxn = self.idxn
 #        crs=self.crs
 #        #=======================================================================
 #        # open the dataset from disk
 #        #=======================================================================
 #        with xr.open_dataset(nc_fp, engine='netcdf4',chunks='auto', decode_coords="all") as xds:
 #            scale_l = xds['scale'].values.tolist()
 #            log.info(f'loaded w/ \n    data_vars:{list(xds.keys())}\n    scales:{scale_l} \n    {nc_fp}')
 #            
 #            assert xds.rio.crs==crs
 #            #===================================================================
 #            # loop and compute delta for each
 #            #===================================================================
 #            base_xds = xds.isel(scale=0)
 #            res_d = dict()
 #            for i, scale in enumerate(scale_l):
 #                log.info(f'    {i+1}/{len(scale_l)} scale={scale}')
 #                
 #                #compute the difference (with the base mask
 #                xds1 = xds.isel(scale=i).where(np.invert(np.isnan(base_xds))) - base_xds                
 #                
 #                if idxn in xds1.coords:
 #                    xds1 = xds1.reset_coords(names=idxn, drop=True)
 #                
 #                res_d[scale] = xds1
 #                
 #            #===================================================================
 #            # merge
 #            #===================================================================
 # 
 #            """these are datasets"""
 #            res_xds = xr.concat(res_d.values(),pd.Index(res_d.keys(), name=idxn, dtype=int)
 #                                ).rename({k:f'{k}_diff' for k in xds1.data_vars})
 #                                
 #            """
 #            res_xds['wse_diff'].plot(col='scale')
 #            """
 #                                
 #            #===================================================================
 #            # write
 #            #===================================================================
 # 
 #            ofp_l = self._save_mfdataset(res_xds, resname, log, xr_dir=xr_dir)
 #                
 #        log.info(f'finished in {(now()-start).total_seconds()}')
 #        
 #        return ofp_l
 # 
 #==============================================================================

    def _cm_stat_calc(self, cm_ds, lay_ds, func, agg_kwargs={}):
        
        
        d = dict()
        def add_calc(name, ds_i):
            stat_d = func(ds_i, agg_kwargs=agg_kwargs)
            d[name] = xr.concat(stat_d.values(), pd.Index(stat_d.keys(), name='metric', dtype=str))
                
        #===============================================================
        # full
        #===============================================================
        add_calc('full', lay_ds)
        #===============================================================
        # loop on each catmask
        #===============================================================
        for dsc, dsc_int in self.cm_int_d.items():
            """NOTE: this computes against the full stack.. 
    
            but the scale=1 values dont have a CatMask"""
            # get this boolean
            mask_ds = cm_ds == dsc_int
            # mask the data
            lay_mask_ds = lay_ds.where(mask_ds, other=np.nan) # .plot(col='scale')
            # compute the stats
            add_calc(dsc, lay_mask_ds)

 
        return xr.concat(d.values(), pd.Index(d.keys(), name='dsc', dtype=str))
    
 

    def _statXR_wrap(self, res_d, log, start, ofp):
        res_dxr = xr.concat(res_d.values(), pd.Index(res_d.keys(), name='layer', dtype=str))
        with ProgressBar():
            res_dxr.compute()
            #get a frame
        res_dx = res_dxr.to_dataframe()
        """
        view(res_dx1)
        """
        #=======================================================================
        # wrap
        #=======================================================================
        # clean up a bit
        res_dx1 = res_dx.unstack(['layer', 'dsc', 'metric']).droplevel(0, axis=1).dropna(how='all', axis=1)
 
        res_dx1.sort_index().to_pickle(ofp)
        
        log.info(f'finished on {str(res_dx1.shape)} in {(now()-start).total_seconds():.2f} to \n    {ofp}')
        
        return ofp
    """
    view(res_dx1.T)
    """
    
    def run_cmCounts(self, ds, **kwargs):
        """catmask counts"""
        
        #=======================================================================
        # defaults
        #=======================================================================
        start = now()
        idxn=self.idxn
        log, tmp_dir, out_dir, ofp, resname, write = self._func_setup(f'cm_cnt',  subdir=True,ext='.pkl', **kwargs)

        
        scale_l = ds[self.idxn].values.tolist()
        
        #get just the CatMosaic
        xar = ds['catMosaic'].reset_coords(names='spatial_ref', drop=True).squeeze(drop=True).transpose(idxn, ...)
        
        #=======================================================================
        # loop and compute counts on each scale
        #=======================================================================
        log.info('getting catMosaic counts on %i'%len(scale_l))
        
        #setup delated computation
        def get_counts(xar):
            d = dict()
            for scale, xari in xar.groupby(idxn):
                if scale==1: continue
                vals_ar, cnts_ar = np.unique(xari, return_counts=True)
                
                d[scale] = pd.Series(cnts_ar, index=vals_ar.astype(int))
                
            return d
        
        
        o=dask.delayed(get_counts)(xar)
        
        #execute
        with ProgressBar(): 
            cnt_d = o.compute()
            
        #collect results
        df = pd.concat(cnt_d, axis=1).T
        df.columns.name = 'dsc'
        df.index.name=idxn
        
        #=======================================================================
        # promote to match style
        #=======================================================================
        dx = df.rename(columns={v:k for k,v in self.cm_int_d.items()})
        
        dx.columns = append_levels(dx.columns, {'metric':'count', 'layer':'catMosaic'})
        
        log.debug(f'finisheed on \n{dx}')
        
        #=======================================================================
        # wrap
        #=======================================================================
        dx.to_pickle(ofp)
        
        log.info(f'finished on {str(dx.shape)} in {(now()-start).total_seconds():.2f} to \n    {ofp}')
        
        return ofp
 
        
        """
        xar.plot(col='scale')
        """
        
 
    def run_statsXR(self, ds, 
                    base='s2',
                    func_d =  None,
                    **kwargs):
        """compute stats from the xarray stack"""
        #=======================================================================
        # defaults
        #=======================================================================
        start = now()
        idxn=self.idxn
        log, tmp_dir, out_dir, ofp, resname, write = self._func_setup(f'statsXR_{base}',  subdir=True,ext='.pkl', **kwargs)
        agg_kwargs = dict(dim=('band', 'y', 'x'), skipna=True)
        
        if func_d is None: 
            func_d = {
                            'wse':self._get_wse_statsXR,
                            'wd':self._get_wd_statsXR,
                            #'wse_diff':self._get_diff_statsXR,
                            #'wd_diff':self._get_diff_statsXR,
                            } 
            
        scale_l = ds[self.idxn].values.tolist()

        
        # remove spatial         
        ds1 = ds.reset_coords(names='spatial_ref', drop=True) 
        #===================================================================
        # loop on each layer
        #===================================================================
        res_d = dict()
        for layName, func in func_d.items():
                
            #skips
            if base in ('s1', 's12'):
                if not layName in ['wse', 'wd']:continue
            #===============================================================
            # setup
            #===============================================================
            
            
            if base in ('s2', 's12'):
                lay_ds = ds1[layName]
            else:
                #replace the raw onto all scales to match the syntax
                base_dxr = ds1[layName].isel(scale=0).reset_coords(names=idxn, drop=True)
                dx_d = {k:base_dxr for k in scale_l}
                lay_ds = xr.concat(dx_d.values(), pd.Index(dx_d.keys(), name=idxn, dtype=int))
                
            #===================================================================
            # compute the stat on the masks
            #===================================================================
            res_d[layName] = self._cm_stat_calc(ds1['catMosaic'], lay_ds, func, agg_kwargs=agg_kwargs)
            
            log.info(f'finished {layName} w/ {len(res_d[layName])}')
 
            
        #===================================================================
        # wrap
        #===================================================================
 
        
        return self._statXR_wrap(res_d, log, start, ofp)
    
    def get_s12XR(self, xds_raw, **kwargs):
        """get difference stats"""
        #=======================================================================
        # defaults
        #=======================================================================
        start = now()
        idxn=self.idxn
        log, tmp_dir, out_dir, ofp, resname, write = self._func_setup('s12XR',  subdir=True,ext='.pkl', **kwargs)
        #agg_kwargs = dict(dim=('band', 'y', 'x'), skipna=True) 
        
        #=======================================================================
        # prep data
        #=======================================================================
        scale_l = xds_raw['scale'].values.tolist()
        
        log.info(f'on {len(scale_l)} scales')        

        #clean up and reorder dinensions for indexing
        #=======================================================================
        # xds = xds_raw.drop_vars('catMosaic').squeeze(drop=True).reset_coords(
        #     names='spatial_ref', drop=True).transpose(idxn, ...)
        #=======================================================================
 
            
        xds = xds_raw.drop_vars('catMosaic', errors='ignore').transpose(idxn, ...)
        
        #=======================================================================
        # loop on each layer
        #=======================================================================
        res_d = dict()
        for layName, xar in xds.items():
            log.info(f'on {layName}')
        
            #===================================================================
            # loop and compute difference for each scale
            #===================================================================
            base_xar = xar[0]
            d = dict()
            for i, (scale, gxar) in enumerate(xar.groupby(idxn)):
                log.info(f'    {i+1}/{len(scale_l)} scale={scale}')
                
                #get the base mask
                if layName=='wd':
                    base_mask = np.logical_or(
                        base_xar==0,
                        gxar==0)
                    """
                    plt.close('all')
                    base_mask.values
                    gxar.plot()
                    gxar.where(np.invert(base_mask)).plot()
                    """
                elif layName=='wse':
                    base_mask = np.isnan(base_xar)
                
                #compute the difference (with the base mask) #Locations at which to preserve this object’s values.
                gxar_diff = gxar.where(np.invert(base_mask)) - base_xar                
                
                if idxn in gxar_diff.coords:
                    gxar_diff = gxar_diff.reset_coords(names=idxn, drop=True)
                
                d[scale] = gxar_diff
                
            #merge
            res_d[layName] = xr.concat(d.values(), pd.Index(d.keys(), name=idxn, dtype=int))
            
        #=======================================================================
        # #wrap
        #=======================================================================
        #merge the scales back
        xds = xr.merge([xds_raw.drop_vars(res_d.keys())] + list(res_d.values()))
        
        #=======================================================================
        # write
        #=======================================================================
        if write:

            i = 0
            for layer, xar in xds.items():
                od = os.path.join(out_dir, 'xr', layer)
                if not os.path.exists(od):os.makedirs(od)
            
                for scale, xari in xar.groupby(idxn):
                    ofpi = os.path.join(od, f'{resname}_{layer}_s{scale:03d}.nc')
                    xari.to_netcdf(ofpi, mode='w', format='NETCDF4', engine='netcdf4', compute=True)
                    log.info(f'wrote {xari.shape} to {ofpi}')
                    i+=1
            
            log.info(f'finished writing {i} to \n    {od}')
            
 
        return  xds
  
    def run_TP_XR(self, ds, 
 
                    **kwargs):
        """compute TP ratio from xr stack
        
        
        
        here we compute a new stack and get the zonal stats
        cant use the generic function because the new stack has variable resolution
        
        """
        #=======================================================================
        # defaults
        #=======================================================================
        start = now()
        idxn=self.idxn
        log, tmp_dir, out_dir, ofp, resname, write = self._func_setup('s12_TP',  subdir=True,ext='.pkl', **kwargs)
        agg_kwargs = dict(dim=('band', 'y', 'x'), skipna=True) 
        scale_l = ds[idxn].values.tolist()
        
        
        #=======================================================================
        # prep data
        #=======================================================================
        #convert to binary exposure (1=exposed)
        lay_ds = np.invert(np.isnan(ds['wse'].reset_coords(names='spatial_ref', drop=True))).astype(int)
        cm_ds = ds['catMosaic'].reset_coords(names='spatial_ref', drop=True) 
        
        base_xar = lay_ds.isel(scale=0)
        
        def get_stats(xar):
            stat_d= {
                    'mean':xar.mean(**agg_kwargs)                    
                    }
            
            return xr.concat(stat_d.values(), pd.Index(stat_d.keys(), name='metric', dtype=str))
        
        #=======================================================================
        # loop and compute exposures on each
        #=======================================================================
        log.info(f'calculating s1/s2 expo TP w/ :{scale_l}')
        res_d = dict()
        for i, scale in enumerate(scale_l):
            if i==0:continue
            log.info(f'    {i+1}/{len(scale_l)}')
            
            #update the scal
            def clean(xar): 
                xar = xar.reset_coords(names=['scale'], drop=True)
                xar.attrs['scale'] = scale
                return xar
            
            #===================================================================
            # compute the fine exposures (at s2)
            #===================================================================
            """ranges from 0:dry to scale**2:wet. no nulls"""            
            s1_expo_xar_s2 = clean(base_xar.coarsen(dim={'x':scale, 'y':scale}, boundary='exact').sum())
            assert (base_xar.shape[1]//scale, base_xar.shape[2]//scale) ==s1_expo_xar_s2.shape[1:]
            
            #===================================================================
            #retrieve coarse exposures
            #===================================================================
            #decimate and scale
            """either 0=dry or scale**2=wet. no nulls"""
            s2_expo_xar_s2 = clean(lay_ds.sel(scale=scale).coarsen(dim={'x':scale, 'y':scale}, boundary='exact').max()*(scale**2)) 
            assert s2_expo_xar_s2.shape==s1_expo_xar_s2.shape
            #===================================================================
            # compute ratio
            #===================================================================
            """from near zero to 1.0. nulls where  s2_expo_xar_s2==0
            
            for method=filter in DPs, can have some s2 zero where we have non-zero s1 values
                means we get 'inf' for the division
                these are FPs
            """
            s12_expo_xar_s2 = s1_expo_xar_s2/s2_expo_xar_s2
            
            #mask out false positives
            s12_expo_xar_s2 = s12_expo_xar_s2.where(s12_expo_xar_s2!=np.inf, other=np.nan)
            
 
            
            assert np.nanmax(s12_expo_xar_s2.values)==1.0
            assert np.nanmin(s12_expo_xar_s2.values)>0.0
            
            """
            view(pd.DataFrame(s12_expo_xar_s2.values[0]))
            base_xar.values
            np.unique(s1_expo_xar_s2.values)
            np.unique(s2_expo_xar_s2.values)
            
            plt.close('all')
            fig, ax_ar = plt.subplots(nrows=3, figsize=(5,12))
            s1_expo_xar_s2.plot(ax = ax_ar[0])
            s2_expo_xar_s2.plot(ax = ax_ar[1])
            s12_expo_xar_s2.plot(ax = ax_ar[2])
            d = {'s1':s1_expo_xar_s2, 's2':s2_expo_xar_s2, 's12':s12_expo_xar_s2}
            xr.concat(d.values(),  pd.Index(d.keys(), name='base', dtype=str)).plot(col='base')
            """
            
            #=======================================================================
            # get stats on each mask------
            #=======================================================================
            """beacuse of variable resolutions cant use the same function"""
            #s2  mask
            cm_s2 = clean(cm_ds.sel(scale=scale).coarsen(dim={'x':scale, 'y':scale}, boundary='exact').max())
            
            
            #===============================================================
            # loop on each catmask
            #===============================================================
            d=dict()
            
            d['full'] = get_stats(s12_expo_xar_s2)
            
            """
            s12_expo_xar_s2.to_dataframe().mean()
 
            
            d['full'].values
            """
            
            for dsc, dsc_int in self.cm_int_d.items():
                """NOTE: this computes against the full stack.. 
        
                but the scale=1 values dont have a CatMask"""
                # get this boolean
                mask_dar = cm_s2 == dsc_int
                # mask the data
                s12_expo_maskd = s12_expo_xar_s2.where(mask_dar, other=np.nan) # .plot(col='scale')
 
                # compute the stats 
                d[dsc] = get_stats(s12_expo_maskd)
                
            #===================================================================
            # wrap scale
            #===================================================================
            res_d[scale] = xr.concat(d.values(), pd.Index(d.keys(), name='dsc', dtype=str))
            log.info(f'finished scale={scale} w/ {res_d[scale].dims}')
            """
            res_d[scale].values
            """
            
        
        #=======================================================================
        # wrap
        #=======================================================================
        log.info(f'executing on {len(res_d)}')
        res_dxr = xr.concat(res_d.values(), pd.Index(res_d.keys(), name='scale', dtype=int))
        with ProgressBar():
            res_dxr.compute()
            #get a frame
        res_dx = res_dxr.to_dataframe()
        
        """
        view(res_dx)
        """
        #=======================================================================
        # wrap
        #=======================================================================
        # clean up a bit
        res_dx1 = res_dx.unstack(['dsc', 'metric'])
        
        res_dx1.columns.set_names(['layer', 'dsc', 'metric'], inplace=True)
        
        #res_dx1.loc[1, :] = np.nan
 
        
        
        res_dx1.sort_index().to_pickle(ofp)
        
        log.info(f'finished on {str(res_dx1.shape)} in {(now()-start).total_seconds():.2f} to \n    {ofp}')
        
        return ofp
    

    
    

    
    def get_kde_df(self, xar_raw,dim='scale',

                          **kwargs):
        """plot a set of gaussian kdes
        
        Parameters
        -----------
        dim: str
            dimension of DataArray to build kde progression on
            
        Todo
        -----------
        use dask delayed and build in parallel?
        
        just partial zones?
        """
        #=======================================================================
        # defautlts
        #=======================================================================
        log, tmp_dir, out_dir, ofp, resname, write = self._func_setup('kde_df',  ext='.pkl', **kwargs)
        idxn = self.idxn
        scale_l = xar_raw[dim].values.tolist()
        start=now()
        
        log.info(f'building {len(scale_l)} on {xar_raw.shape}')
        
        xar1 = xar_raw.squeeze(drop=True).reset_coords( #drop band
            names=['spatial_ref'], drop=True
            ).transpose(idxn, ...)  #put scale first for iterating
        
        #=======================================================================
        # loop and build values
        #=======================================================================
        
        @dask.delayed
        def get_vals(xari):            
            
            dar = xari.stack(rav=list(xari.coords)).dropna('rav').data
            kde = scipy.stats.gaussian_kde(dar,
                                               bw_method='scott',
                                               weights=None,  # equally weighted
                                               )
            
            xvals = np.linspace(dar.min() + .01, dar.max(), 200)
            
            del dar
            
            return pd.concat({'x':pd.Series(xvals), 'y':pd.Series(kde(xvals))})
            
        def get_all_vals(xar):
            d = dict()
            for i, (scale, xari) in enumerate(xar.groupby(idxn, squeeze=False)):
                xari_s2 = xari.coarsen(dim={'x':scale, 'y':scale}, boundary='exact').max()
                
                """
                xari_s2.plot()
                xari_s2.shape
                xari_s2.values
                """
                d[scale] = get_vals(xari_s2)
 
            return d
 
        log.info(f'executing get_all_vals on  {xar1.shape}')
                 
        def concat(d):
            df = pd.concat(d, axis=1, names=['scale'])
            df.index.set_names(['dim', 'coord'], inplace=True)
            return df.unstack('dim') 
        
        o=dask.delayed(concat)(get_all_vals(xar1))
        #df.visualize(filename=os.path.join(out_dir, 'dask_visualize.svg'))
        #d = dask.compute(get_all_vals(xar1))
        
        """stalling at ~38%"""
        with ProgressBar():
 
            df = o.compute()
        
        #=======================================================================
        # wrap
        #=======================================================================
        if write:
            df.to_pickle(ofp)
            log.info(f'wrote {str(df.shape)} to file\n    {ofp}') 
            
        log.info(f'finished on {df.shape} in %.2f secs'%((now()-start).total_seconds()))       
        
        return df
    
    def write_rasters_XR(self, ds_raw, prefix='', **kwargs):
        """dump xarray into rasters"""
        log, tmp_dir, out_dir, ofp, resname, write = self._func_setup('toRaster', subdir=True, ext='.tif', **kwargs)
        idxn = self.idxn
        
        
        #=======================================================================
        # loop on each layer
        #=======================================================================
        ds = ds_raw.squeeze(drop=True)
        new_band = 'band'
        #ds.assign_coords({new_band:('scale',np.array([1,2,3]))}).reset_index('scale', drop=True).set_index({'band':new_band})
        scale_l = ds_raw['scale'].values.tolist()
        
        
        ds_1 = ds.assign_coords({new_band:('scale',np.arange(len(scale_l)))}).swap_dims({'scale':new_band})
        
        d = dict()
        for layName, xar in ds_1.items():
            
            ofpi = os.path.join(out_dir, f'{resname}_{prefix}_{layName}_{len(scale_l)}.tif')
            #xar.rio.write_crs(crs, inplace=True)
            log.info(f'writing {xar.shape} to \n    {ofpi}')
            with ProgressBar():
                xar.rio.to_raster(ofpi)
            
            
            d[layName] = ofpi
            
        #=======================================================================
        # wrap
        #=======================================================================
        log.info(f'finished on {len(d)}')
        
        return d
        
                
                
 
 
            
            
            
def get_pixel_info(ds):
    return {'pixelArea':ds.res[0]*ds.res[1], 'pixelLength':ds.res[0]}















    
    
