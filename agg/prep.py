'''
Created on May 9, 2022

@author: cefect
'''
import os, sys
import numpy as np
from qgis.core import QgsCoordinateReferenceSystem, QgsMapLayerStore
from hp.hyd import HQproj
from agg.hyd.hscripts import StudyArea
from definitions import proj_lib, base_resolution
np.random.seed(100)

class RandomSA(StudyArea, HQproj):
    def __init__(self,
                 resolution=None,
                 extent_fp=None,
                 **kwargs):
        
        if resolution is None:
            resolution=base_resolution
        self.resolution=resolution
        
        super().__init__(base_resolution=resolution, **kwargs)
        
        

        
        self.set_extent(extent_fp)
        
    def set_extent(self,
                    raw_fp,
                    logger=None,
                    ):
        if logger is None: logger=self.logger
        log=logger.getChild('set_extents')
        #=======================================================================
        # set extents
        #=======================================================================
        mstore = QgsMapLayerStore()
        raw_rlay = self.get_layer(raw_fp, mstore=mstore)
        
        #assert self.rlay_get_resolution(raw_rlay)==self.resolution
        
        #=======================================================================
        # wrap
        #=======================================================================
        self.extent=raw_rlay.extent()
        log.info('set extent from %s'%raw_rlay.name())
        mstore.removeAllMapLayers()
        
        return self.extent
    
    def build_rand(self, 
                   resolution=None,
                   out_dir=None,
                   layname='layer',
                   logger=None,
                   **kwargs):
        if logger is None: logger=self.logger
        log=logger.getChild('build_rand')
        if resolution is None: resolution=self.resolution
        if out_dir is None: out_dir=self.out_dir
        
        fp =  self.randomuniformraster(resolution,  extent=self.extent,
                             output=os.path.join(out_dir, '%s.tif'%layname), **kwargs)
        
        log.info('built %s'%fp)
        return self.get_layer(fp)
    
 
 

def build_random_proj( #construct a random project
        base_proj='obwb',
        extent_fp=r'C:\LS\02_WORK\NRC\2112_Agg\04_CALC\hyd\OBWB\aoi\aoi01_rand_0511.gpkg',
        #init kwargs
        init_kwargs = dict(trim=False,overwrite=True),

        ):
 
    print('from %s'%base_proj)
    
 
    
    with RandomSA(tag='rand', name='prep',extent_fp=extent_fp,
                   **init_kwargs, **proj_lib[base_proj]) as ses:
        
        #get the random layers
        dem = ses.build_rand(layname='dem_'+ses.longname)
        
        wse_raw = ses.build_rand(layname='wse_raw_'+ses.longname)
        
        #gw filter the wse
        wsegw = ses.wse_remove_gw(wse_raw, dem_rlay=dem)
        
        print(wsegw.source())
        
        
        
 

def convert_wse(
        out_dir=r'C:\LS\10_OUT\2112_Agg\outs\prep\0514',
        compression='med',
        studyArea_l=None,
        proj_lib=None,
        ):
    
    if proj_lib is None:
        from definitions import proj_lib
        
        
    if not os.path.exists(out_dir):os.makedirs(out_dir)
    
    if not studyArea_l is None:
        proj_lib = {k:v for k,v in proj_lib.items() if k in studyArea_l}
        
    res_lib = dict()
    for studyArea, pars_d in proj_lib.items():
        wse_fp_d = pars_d['wse_fp_d']
        dem_fp = pars_d['dem_fp_d'][base_resolution]
        
        crs = QgsCoordinateReferenceSystem('EPSG:%i' % pars_d['EPSG'])
        
        res_lib[studyArea] = dict()
        
        with HQproj(dem_fp=dem_fp, out_dir=out_dir, crs=crs,
                    base_resolution=base_resolution,
                    overwrite=True) as ses:
            for lvl, fp in wse_fp_d.items():
                raw_rlay = ses.get_layer(fp, mstore=ses.mstore)
                
                rlay = ses.wse_remove_gw(raw_rlay, out_dir=ses.temp_dir)
                
                #write w/ compression
                ses.rlay_write(rlay, out_dir=os.path.join(out_dir, studyArea),
                               compression=compression)
        
        
def check_projLib(
        proj_lib=None):
    
    if proj_lib is None:
        from definitions import proj_lib
        
    #===========================================================================
    # loop on study areas
    #===========================================================================
    err_d = dict()
    for studyArea, d in proj_lib.items():
 
        #=======================================================================
        # check layer containers
        #=======================================================================
        for k0, fp_d in {k:v for k,v in d.items() if k in ['wse_fp_d', 'dem_fp_d']}.items():
            for k1, fp in fp_d.items():
                if not os.path.exists(fp):
                    err_d['%s.%s.%s'%(studyArea, k0, k1)] = 'bad filepath: %s'%fp
 
        
        #=======================================================================
        # check individual layers
        #=======================================================================
        for k, fp in {k:v for k,v in d.items() if k in ['finv_fp', 'aoi']}.items():
            if not os.path.exists(fp):
                err_d['%s.%s'%(studyArea, k)] = 'bad filepath: %s'%fp
        
        
    #===============================================================================
    # wrap
    #===============================================================================
    if len(err_d)>0:
        print('got %i errors'%len(err_d))
        for k,v in err_d.items():
            print('%s:    %s'%(k,v))
            
    else:
        print('no errors')
            
        

if __name__ == "__main__":
    #check_projLib()
    convert_wse(studyArea_l=['LMFRA', 'Calgary'])
    print('finished')