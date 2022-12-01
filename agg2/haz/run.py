'''
Created on Aug. 28, 2022

@author: cefect
'''
import os, pathlib, pprint, webbrowser
from definitions import proj_lib
from hp.basic import get_dict_str, now, today_str
import shapely.geometry as sgeo
import numpy as np
import pandas as pd
idx = pd.IndexSlice

from rasterio.crs import CRS

import rioxarray
from dask.diagnostics import ProgressBar 
import dask

from hp.pd import view


res_fp_lib = {'r9':
              {
                'direct':{  
                    'agg': 'C:\\LS\\10_IO\\2112_Agg\\outs\\agg2\\r9\\SJ\\direct\\20220921\\agg\\SJ_r9_direct_0921_agg.pkl',
                    'aggXR':r'C:\LS\10_IO\2112_Agg\outs\agg2\r9\SJ\direct\20220923\aggXR\SJ_r9_direct_0923_aggXR.nc',
                    'diffs': r'C:\LS\10_IO\2112_Agg\outs\agg2\r9\SJ\direct\20220922\diffs\SJ_r9_direct_0922_diffs.pkl',
                    'catMasks': 'C:\\LS\\10_IO\\2112_Agg\\outs\\agg2\\r9\\SJ\\direct\\20220921\\cMasks\\SJ_r9_direct_0921_cMasks.pkl'
                    },
                'filter':{  
                    'agg': 'C:\\LS\\10_IO\\2112_Agg\\outs\\agg2\\r9\\SJ\\filter\\20220921\\agg\\SJ_r9_filter_0921_agg.pkl',
                    'diffs': r'C:\LS\10_IO\2112_Agg\outs\agg2\r9\SJ\filter\20220922\diffs\SJ_r9_filter_0922_diffs.pkl',
                    'catMasks': 'C:\\LS\\10_IO\\2112_Agg\\outs\\agg2\\r9\\SJ\\filter\\20220921\\cMasks\\SJ_r9_filter_0921_cMasks.pkl'
                }},
            'r10':
              {
                'direct':{  
                    'agg': 'C:\\LS\\10_IO\\2112_Agg\\outs\\agg2\\r9\\SJ\\direct\\20220921\\agg\\SJ_r9_direct_0921_agg.pkl',
 
                    },
                'filter':{  
                    'agg': 'C:\\LS\\10_IO\\2112_Agg\\outs\\agg2\\r9\\SJ\\filter\\20220921\\agg\\SJ_r9_filter_0921_agg.pkl',
 
                }},
              'r11':{
                  'direct':{
                      'agg':r'C:\LS\10_IO\2112_Agg\outs\agg2\r11\SJ\direct\20221006\agg\SJ_r11_direct_1006_agg.pkl',
                      'catMasks':r'C:\LS\10_IO\2112_Agg\outs\agg2\r11\SJ\direct\20221006\cMasks\SJ_r11_direct_1006_cMasks.pkl',
                      'aggXR':r'C:\LS\10_IO\2112_Agg\outs\agg2\r11\SJ\direct\20220930\_xr',
                      'diffXR':r'C:\LS\10_IO\2112_Agg\outs\agg2\r11\SJ\direct\20221015\s12XR\xr',
                      },
                  'filter':{
                      'agg':r'C:\LS\10_IO\2112_Agg\outs\agg2\r11\SJ\filter\20221006\agg\SJ_r11_filter_1006_agg.pkl',
                      'catMasks':r'C:\LS\10_IO\2112_Agg\outs\agg2\r11\SJ\filter\20221006\cMasks\SJ_r11_filter_1006_cMasks.pkl',
                      'aggXR':r'C:\LS\10_IO\2112_Agg\outs\agg2\r11\SJ\filter\20220930\_xr',
                      'diffXR':r'C:\LS\10_IO\2112_Agg\outs\agg2\r11\SJ\filter\20221014\s12XR\xr',
                      }},
              'dev':{  
                  'filter':{
                       'aggXR': 'C:\\LS\\10_IO\\2112_Agg\\outs\\agg2\\t\\SJ\\filter\\20221013\\_xr',
                       'diffXR':r'C:\LS\10_IO\2112_Agg\outs\agg2\t\SJ\filter\20221014\s12XR\xr',
                       },
                  'direct':{  
                       'aggXR': 'C:\\LS\\10_IO\\2112_Agg\\outs\\agg2\\t\\SJ\\direct\\20221013\\_xr',
                       }
                       }
              }
        



def run_haz_agg2XR(method='direct',
            fp_d={},
            case_name = 'SJ',
            dsc_l=[1,  2**5, 2**6, 2**7, 2**8, 2**9],
 
            proj_d=None,
                 **kwargs):
    """hazard/raster run for agg2 xarray"""
 
    #===========================================================================
    # extract parametesr
    #===========================================================================
    #project data   
    if proj_d is None: 
        proj_d = proj_lib[case_name] 
    wse_fp=proj_d['wse_fp_d']['hi']
    dem_fp=proj_d['dem_fp_d'][1] 
    crs = CRS.from_epsg(proj_d['EPSG'])
    #===========================================================================
    # run model
    #===========================================================================
    from agg2.haz.scripts import UpsampleSessionXR as Session    
    #execute
    with Session(case_name=case_name,method=method,crs=crs, nodata=-9999, dsc_l=dsc_l, **kwargs) as ses:
 
        log = ses.logger
        #=======================================================================
        # build aggregated layers
        #=======================================================================
 
        if not 'agg' in fp_d:
            fp_d['agg'] = ses.run_agg(dem_fp, wse_fp, method=method)
            ses._clear()
            
        
        #pre-processed base rasters
        base_fp_d = pd.read_pickle(fp_d['agg']).iloc[0, :].to_dict()
        
 
        if not 'aggXR' in fp_d:
            fp_d['aggXR'] = ses.build_downscaled_aggXR(pd.read_pickle(fp_d['agg']))
            
        
        #=======================================================================
        # deltas
        #=======================================================================
        """only for plots"""
        if not 'diff' in fp_d:
            ds = ses.get_ds_merge(fp_d['aggXR'])
            diff_ds = ses.get_s12XR(ds, write=True)
            #ses.write_rasters_XR(diff_ds, prefix='s12')
            return
 
        
        #=======================================================================
        # category masks
        #=======================================================================
        if not ('catMasks' in fp_d) or not ('cmXR' in fp_d):            
            fp_d['cmXR'], fp_d['catMasks'] = ses.run_catMasksXR(base_fp_d['dem'], base_fp_d['wse'], write_tif=True)            
            ses._clear()
            

 
        log.info(f'finished on \n\n'+
                 pprint.pformat(fp_d, width=30, indent=3, compact=True, sort_dicts =False))
 
                
    return ses.xr_dir

 
def build_vrt(pick_fp,layName_l = ['dem', 'wd', 'wse'],out_dir=None, **kwargs):
    if out_dir is  None:
        out_dir = os.path.join(os.path.dirname(pick_fp), 'vrt')
    from agg2.haz.scripts import UpsampleSession as Session
    with Session(out_dir=out_dir, **kwargs) as ses:
        log = ses.logger
        df = pd.read_pickle(pick_fp)
        
        #=======================================================================
        # loop and write each vrt
        #=======================================================================
        res_d = dict()
 
        for layName in layName_l:
            if pick_fp.endswith('cMasks.pkl'):
                fp_d = df['fp'].to_dict()
                layName='catMask'
            else:
                fp_d = df[layName].dropna().to_dict()
                
            
            
            ofpi = ses.build_vrts(fp_d,ofp = os.path.join(out_dir, f'{layName}_{len(fp_d):03d}.vrt'))
            
            log.info('    for \'%s\' compiled %i into a vrt: %s'%(layName, len(fp_d), os.path.basename(ofpi)))
            
            res_d['%s'%(layName)] = ofpi
            
            if layName=='catMask':break
        #=======================================================================
        # wrap
        #=======================================================================
        log.info('finished on \n'+pprint.pformat(res_d, width=30, indent=3, compact=True, sort_dicts =False))
        
    return res_d
        
 
        
 




def SJ_run(run_name='r9',method='direct',**kwargs):
    return run_haz_agg2XR(case_name='SJ', fp_d = res_fp_lib[run_name][method], method=method, run_name=run_name,
                          bbox=sgeo.box(2484736.000, 7435776.000, 2493952.000, 7443968.000),
                          **kwargs)


def SJ_dev(run_name='t',method='direct',**kwargs):
    return run_haz_agg2XR(case_name='SJ', fp_d = {}, method=method, run_name=run_name, 
                        dsc_l=[1,  2**3, 2**4],
                        bbox = sgeo.box(2492040.000, 7436320.000, 2492950.000, 7437130.000),
                        **kwargs)

if __name__ == "__main__": 
    start = now()
 
    #scheduler='single-threaded'
    scheduler='threads'
    with dask.config.set(scheduler=scheduler):
        print(scheduler)
          
        #xr_dir = SJ_dev(method='filter')
       
        xr_dir = SJ_run(method='direct',run_name='r11')
    
    #===========================================================================
    # for method in ['filter', 'direct']:
    #     build_vrt(res_fp_lib['r11'][method]['catMasks'])
    #===========================================================================
 
 
 
    print('finished in %.2f'%((now()-start).total_seconds())) 
