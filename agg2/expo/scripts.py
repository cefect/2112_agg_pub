'''
Created on Sep. 6, 2022

@author: cefect

aggregation exposure/assetse
'''
#===============================================================================
# IMPORTS-------
#===============================================================================

import numpy as np
import pandas as pd
idx= pd.IndexSlice
import os, copy, datetime

import geopandas as gpd
import shapely.geometry as sgeo
import rasterio as rio
#import rasterio.windows

from rasterstats import zonal_stats
#import rasterstats.utils
#import matplotlib.pyplot as plt


#from hp.oop import Session
from hp.gpd import GeoPandasWrkr, get_multi_intersection
from hp.rio import load_array, RioWrkr, get_window, plot_rast, get_ds_attr
from hp.pd import view
from hp.basic import now
#from hp.plot import plot_rast
 
from agg2.haz.rsc.scripts import ResampClassifier
from agg2.coms import Agg2Session




class ExpoWrkr(GeoPandasWrkr, ResampClassifier):
    
    def get_rlays_samp_pts(self, rlay_fp_d, gdf, logger=None):
        """compute  stats for assets on each raster
        
        consider splitting the raster files into pools
        """
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger=self.logger
        log=logger.getChild('get_rlay_samp')
        
        
        assert isinstance(gdf.iloc[1, :]['geometry'], sgeo.point.Point), 'only setup for points here'
        
        #=======================================================================
        # loop and sample
        #=======================================================================
        coord_list = [(x,y) for x,y in zip(gdf['geometry'].x , gdf['geometry'].y)]
        
        res_d = dict()
        for scale, rlay_fp in rlay_fp_d.items():
            log.info('on scale=%i w/ %s' % (scale, os.path.basename(rlay_fp)))
 
            with rio.open(rlay_fp, mode='r') as ds:
                #check consistency
                assert ds.crs.to_epsg() == self.crs.to_epsg() 
 
                #sample the list
                res_d[scale] = [x[0] for x in ds.sample(coord_list, masked=False, indexes=1)]                
                nodata = ds.nodata
                
        #=======================================================================
        # wrap
        #=======================================================================
        rdf = pd.DataFrame.from_dict(res_d).rename_axis(gdf.index.name).rename_axis('scale', axis=1)
        return rdf.replace({nodata:np.nan})

    def get_assetRsc(self, cm_fp_d, gdf, bbox=None, logger=None):
        """compute zonal stats for assets on each resample class mosaic"""
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger=self.logger
        log=logger.getChild('get_assetRsc')
        
        #=======================================================================
        # loop and sample
        #=======================================================================
        res_d = dict()
        for scale, rlay_fp in cm_fp_d.items():
            log.info('on scale=%i w/ %s' % (scale, os.path.basename(rlay_fp)))
            with rio.open(rlay_fp, mode='r') as ds:
                #check consistency
                assert ds.crs.to_epsg() == self.crs.to_epsg()
                
                #check intersection 
                rbnds = sgeo.box(*ds.bounds)
                ebnds = sgeo.box(*gdf.total_bounds)
 
                    
                """relaxing this for now... 
                would be better to determine a single bounds for everything before the zonal though...
                assert ebnds.contains(bbox), 'bounding box exceeds assets extent'                    
                """
                
                
                #build a clean window
                """basically rounding the raster window so everything fits"""
                window, win_transform = get_window(ds, bbox)
                
                
                """
                plt.close('all')
                fig, ax = plt.subplots()
                ax.plot(*rbnds.exterior.xy, color='red', label='raster (raw)')
                ax.plot(*ebnds.exterior.xy, color='blue', label='assets')
                gdf.plot(ax=ax, color='blue')
                ax.plot(*bbox.exterior.xy, color='orange', label='bbox', linestyle='dashed')
                wbnds = sgeo.box(*rio.windows.bounds(window, transform=ds.transform))
                ax.plot(*wbnds.exterior.xy, color='green', label='window', linestyle='dotted')
                #ax.plot(*bbox1.exterior.xy, color='black', label='bbox_buff', linestyle='dashed')
                fig.legend()
                limits = ax.axis()
                """
                    
                    
 
                    
 
                #===============================================================
                # loop each category's mask'
                #=============================================================== 
                mosaic_ar = load_array(ds, window=window) 
                
                cm_d = self.mosaic_to_masks(mosaic_ar)
                #load and compute zonal stats
                zd = dict()
                for catid, ar_raw in cm_d.items():
                    if np.any(ar_raw):
                        stats_d = zonal_stats(gdf, np.where(ar_raw, 1, 0), 
                                                affine=win_transform, 
                                                nodata=0, 
                                                all_touched=True, #when pixels get really large, need more than just the centroid
                                                stats=[ 'max',
                                                       #'nan', #only interested in real interseects
                                                       ])
                        zd[catid] = pd.DataFrame(stats_d)
                        
 
                        """
                        pd.DataFrame(stats_d).dtypes
                        plot_rast(ar_raw, transform=win_transform, ax=ax )
                        """
                #===============================================================
                # wrap
                #===============================================================
                rdx1 = pd.concat(zd, axis=1, names=['dsc']).droplevel(1, axis=1).rename_axis(gdf.index.name)
                
                rdx1 = rdx1.where(~pd.isnull(rdx1), np.nan) #replace None w/ nulls
 
                
                if rdx1.notna().all(axis=1).any():
                    log.warning('got some assets with no hits on scale=%i'%scale)
                
                res_d[scale]=rdx1
        
        #=======================================================================
        # merge
        #=======================================================================
        """dropping spatial data"""
        rdx = pd.concat(res_d, axis=1, names=['scale'])
        
        #=======================================================================
        # checks
        #=======================================================================
        assert not rdx.notna().all(axis=1).any(), 'got some assets with no hits'
        
        log.info('finished w/ %s' % str(rdx.shape))
        return rdx

class ExpoSession(ExpoWrkr, Agg2Session):
    """tools for experimenting with downsample sets"""
    
    def __init__(self, 
                 method='direct',
                 scen_name=None,
                 obj_name='expo',
                 **kwargs):
        """
        
        Parameters
        ----------
 
        """
        if scen_name is None:
            scen_name=method
        super().__init__(obj_name=obj_name, scen_name=scen_name,subdir=False, **kwargs)
        
        #=======================================================================
        # attach
        #=======================================================================
 
        print('finished __init__')
        
        
    def run_expoSubSamp(self):
        """compute resamp class for assets from set of masks"""
        
        #join resampClass to each asset (one column per resolution)
        
        #sample layers on assets
        
        #compute stats
        
    def build_assetRsc(self, cm_fp_d, finv_fp,
                       bbox=None,
                       centroids=True,
                        **kwargs):
        """join resampClass to each asset (one column per resolution)"""
        
        log, tmp_dir, out_dir, ofp, resname, write = self._func_setup('arsc',  subdir=True,ext='.pkl', **kwargs)
        
        if bbox is None: bbox=self.bbox
        start=now()
        #=======================================================================
        # load classification masks
        #=======================================================================

        
        """
        view(pd.read_pickle(pick_fp))
        """ 
        
        log.info('on %i catMasks'%len(cm_fp_d))
        
        
        for k,v in cm_fp_d.items():
            assert os.path.exists(v), k
            
        #get extents from this
        rbnds = sgeo.box(*get_ds_attr(v, 'bounds'))
        
        #=======================================================================
        # harmonize extents
        #=======================================================================
        """minimum intersection between the 3 bounds""" 
        bbox1 = get_multi_intersection([sgeo.box(*gpd.read_file(finv_fp).total_bounds),
                                        rbnds.buffer(-k*2, resolution=1), #conservative to handle window rounding 
                                        bbox])
        
        """
        plt.plot(*rbnds.buffer(-k, resolution=1).exterior.xy)
        """
            
        #=======================================================================
        # load asset data         
        #=======================================================================
        gdf = gpd.read_file(finv_fp, bbox=bbox1).rename_axis('fid')
        assert len(gdf)>0
        abnds = sgeo.box(*gdf.total_bounds) #bouds
 
        if not abnds.within(bbox1):
            """can happen when an asset intersects the bbox"""
            log.warning('asset bounds  not within bounding box \n    %s\n    %s'%(
                        abnds.bounds, bbox1.bounds))
        
        log.info('loaded %i feats (w/ aoi: %s) from \n    %s'%(
            len(gdf), type(bbox1).__name__, os.path.basename(finv_fp)))
        
        assert gdf.crs==self.crs, 'crs mismatch'
        assert len(gdf)>0
        
        """
        
        tuple(np.round(gdf.total_bounds, 1).tolist())
        
        """
        #=======================================================================
        # get downscales
        #=======================================================================
        if centroids:
            
            rdf = self.get_rlays_samp_pts(cm_fp_d, gdf.set_geometry(gdf.centroid),
                                   logger=log)
            

            
        else:
            res_dx = self.get_assetRsc(cm_fp_d, gdf, 
                                       bbox=bbox1, #asset bounds may go beyond raster
                                       logger=log)
            
            """todo: need to collapse this down to one dsc column per scale"""
        
        #=======================================================================
        # write
        #=======================================================================
        #replace with strings
        rdf = rdf.replace({v:k for k,v in self.cm_int_d.items()})
        
        assert rdf.columns.name=='scale'
        assert np.array_equal(rdf.columns.values, np.array(list(cm_fp_d.keys())))
            
        rdf.to_pickle(ofp)
 
        log.info('finished in %.2f wrote %s to \n    %s'%((now()-start).total_seconds(), str(rdf.shape), ofp))
        
        
        return ofp
 
                
 
    def build_layerSamps(self, rlay_fp_d, finv_fp,
                         layName='wd',
                         centroids=True,
                       bbox=None,
                       prec=None,
                        **kwargs):
        """join grid values to each asset (one column per resolution)"""
        start=now()
        log, tmp_dir, out_dir, ofp, resname, write = self._func_setup('lsamp_%s'%layName,  subdir=True,ext='.pkl', **kwargs)
        
        if bbox is None: bbox=self.bbox
        if prec is None: prec=self.prec
        
        #=======================================================================
        # load classification masks
        #=======================================================================

 
        
        log.info('on %i \'%s\' layers'%(len(rlay_fp_d), layName))        
        for k,v in rlay_fp_d.items():
            assert os.path.exists(v), k
            
        #get extents from this
        rbnds = sgeo.box(*get_ds_attr(v, 'bounds'))
        
        #=======================================================================
        # harmonize extents
        #=======================================================================
        """minimum intersection between the 3 bounds""" 
        bbox1 = get_multi_intersection([sgeo.box(*gpd.read_file(finv_fp).total_bounds),
                                        rbnds.buffer(-k*2, resolution=1), #conservative to handle window rounding 
                                        bbox])
        
        """
        plt.plot(*rbnds.buffer(-k, resolution=1).exterior.xy)
        """
            
        #=======================================================================
        # load asset data         
        #=======================================================================
        gdf_raw = gpd.read_file(finv_fp, bbox=bbox1).rename_axis('fid')
        assert len(gdf_raw)>0
        abnds = sgeo.box(*gdf_raw.total_bounds) #bouds
 
        if not abnds.within(bbox1):
            """can happen when an asset intersects the bbox"""
            log.warning('asset bounds  not within bounding box \n    %s\n    %s'%(
                        abnds.bounds, bbox1.bounds))
        
        log.info('loaded %i feats (w/ aoi: %s) from \n    %s'%(
            len(gdf_raw), type(bbox1).__name__, os.path.basename(finv_fp)))
        
        assert gdf_raw.crs==self.crs, 'crs mismatch'
        assert len(gdf_raw)>0
        
        
        #=======================================================================
        # compute
        #=======================================================================
        if centroids:
 
            
            rdf = self.get_rlays_samp_pts(rlay_fp_d, gdf_raw.set_geometry(gdf_raw.centroid), 
 
                                   logger=log)
        else:
            raise IOError('not implemented')
 
 
 

        
        #=======================================================================
        # write
        #=======================================================================
        res_dx = pd.concat({layName:rdf}, axis=1, names=['layer'])
        res_dx.to_pickle(ofp)
        log.info('finished in %.2f wrote %s to \n    %s'%((now()-start).total_seconds(), str(res_dx.shape), ofp))
        
        #=======================================================================
        # write geo
        #=======================================================================
        if write:
            rdf.columns = rdf.columns.astype(str)
            ofpi = os.path.join(out_dir, resname+'.gpkg')
            gdf_raw.join(rdf.astype(np.float32)).to_file(ofpi, driver='GPKG', index=False, engine='fiona')
            
            log.info('wrote  to \n    %s'%(ofpi))
        
        
        return ofp


        
        
        
        
        
    
        
 