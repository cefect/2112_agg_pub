'''
Created on Sep. 6, 2022

@author: cefect
'''
import pprint, os
from definitions import proj_lib
from pyproj.crs import CRS
import pandas as pd
from hp.basic import get_dict_str, now
from agg2.haz.run import res_fp_lib as haz_fp_lib
from agg2.expo.scripts import ExpoSession
start=now()
 
 
def get_picks(run_name, method):
    """retrieve keys from hazard results"""
    fp_d=haz_fp_lib['r11'][method]    
    return fp_d['catMasks'], fp_d['agg']


res_fp_lib = {'r8':{
                'direct':{
                    'catMasks': r'C:\LS\10_IO\2112_Agg\outs\agg2\r8\SJ\direct\20220917\cMasks\SJ_r8_direct_0917_cMasks.pkl',
                    'arsc':r'C:\LS\10_IO\2112_Agg\outs\agg2\r8\SJ\direct\20220918\arsc\SJ_r8_direct_0918_arsc.pkl',
                    'wd':r'C:\LS\10_IO\2112_Agg\outs\agg2\r8\SJ\direct\20220918\lsamp_wd\SJ_r8_direct_0918_lsamp_wd.pkl',
                    'wse':r'C:\LS\10_IO\2112_Agg\outs\agg2\r8\SJ\direct\20220918\lsamp_wse\SJ_r8_direct_0918_lsamp_wse.pkl',
                    },
                'filter':{
                    'catMasks':'C:\\LS\\10_IO\\2112_Agg\\outs\\agg2\\r5\\SJ\\filter\\20220909\\cMasks\\SJ_r5_filter_0909_cMasks.pkl',
                    'arsc':None, #only need to compute this once
                    'wd':r'C:\LS\10_IO\2112_Agg\outs\agg2\r7\SJ\filter\20220911\lsamp_wd\SJ_r7_filter_0911_lsamp_wd.pkl',
                    'wse':r' C:\LS\10_IO\2112_Agg\outs\agg2\r7\SJ\filter\20220911\lsamp_wse\SJ_r7_filter_0911_lsamp_wse.pkl'
                    }
                },
            'r11':{
                'direct':{  
                     'arsc': 'C:\\LS\\10_IO\\2112_Agg\\outs\\agg2\\r11\\SJ\\direct\\20221006\\arsc\\SJ_r11_direct_1006_arsc.pkl',
                       'wd': 'C:\\LS\\10_IO\\2112_Agg\\outs\\agg2\\r11\\SJ\\direct\\20221006\\lsamp_wd\\SJ_r11_direct_1006_lsamp_wd.pkl',
                       'wse': 'C:\\LS\\10_IO\\2112_Agg\\outs\\agg2\\r11\\SJ\\direct\\20221006\\lsamp_wse\\SJ_r11_direct_1006_lsamp_wse.pkl'
                       },
                'filter':{ 
                    'arsc': 'C:\\LS\\10_IO\\2112_Agg\\outs\\agg2\\r11\\SJ\\filter\\20221006\\arsc\\SJ_r11_filter_1006_arsc.pkl',
                    'wd': 'C:\\LS\\10_IO\\2112_Agg\\outs\\agg2\\r11\\SJ\\filter\\20221006\\lsamp_wd\\SJ_r11_filter_1006_lsamp_wd.pkl',
                    'wse': 'C:\\LS\\10_IO\\2112_Agg\\outs\\agg2\\r11\\SJ\\filter\\20221006\\lsamp_wse\\SJ_r11_filter_1006_lsamp_wse.pkl'
                       }
                 
                }}

def run_expo(
        finv_fp,
        cm_pick_fp, agg_pick_fp,
        fp_d={}, 
 
        **kwargs):
    """run exposure calcs on an asset layer and the results from agg2.haz.run"""    
 
    
    with ExpoSession(  nodata=-9999, **kwargs) as ses:
        
        #=======================================================================
        # resample class on each asset
        #=======================================================================
        if not 'arsc' in fp_d:
             
            cm_fp_d = pd.read_pickle(cm_pick_fp)['fp'].to_dict()        
            fp_d['arsc'] = ses.build_assetRsc(cm_fp_d, finv_fp)
            
        #=======================================================================
        # sample layers
        #=======================================================================
        df_raw = pd.read_pickle(agg_pick_fp)
        for layName in ['wd','wse']:
            if not layName in fp_d:
                
                #load this data                
                assert layName in df_raw.columns, 'requested layer \'%s\' is missing from the pick:\n    %s'%(layName,agg_pick_fp)
 
                fp_d[layName] = ses.build_layerSamps(df_raw[layName].to_dict(), finv_fp,  layName=layName,write=True,)
                
    print('finished w/ \n%s'%(pprint.pformat(fp_d, width=30, indent=3, compact=True, sort_dicts =False)))
            
 
            

     
def SJ_expo_dev(method='direct', **kwargs):
    cm_pick_fp, agg_pick_fp = get_picks('r11', method)
    return SJ_expo_run(
        run_name='dev',
        aoi_fp=r'C:\LS\02_WORK\NRC\2112_Agg\04_CALC\hyd\SaintJohn\aoi\aoiT03_0906.geojson',
        cm_pick_fp=cm_pick_fp, agg_pick_fp=agg_pick_fp,
        **kwargs)
    
def SJ_expo_run(method='direct', run_name='r11', case_name='SJ',  
                agg_pick_fp=None, cm_pick_fp=None, **kwargs):
    
    proj_d = proj_lib[case_name] 
 
    crs = CRS.from_epsg(proj_d['EPSG'])
    
    finv_fp = proj_d['finv_fp']
    
    if agg_pick_fp is None:
        cm_pick_fp, agg_pick_fp = get_picks('r11', method)
 
    
    return run_expo(finv_fp, cm_pick_fp, agg_pick_fp,  method=method, 
                    fp_d = res_fp_lib[run_name][method], run_name=run_name,case_name=case_name ,crs=crs, **kwargs)

#===============================================================================
# def SJ_r6_0910(
#         method='filter',
#         fp_lib = {
#                 'direct':{
#                     'catMasks': r'C:\LS\10_IO\2112_Agg\outs\agg2\r8\SJ\direct\20220917\cMasks\SJ_r8_direct_0917_cMasks.pkl',
#                     'arsc':r'C:\LS\10_IO\2112_Agg\outs\agg2\r8\SJ\direct\20220918\arsc\SJ_r8_direct_0918_arsc.pkl',
#                     'wd':r'C:\LS\10_IO\2112_Agg\outs\agg2\r8\SJ\direct\20220918\lsamp_wd\SJ_r8_direct_0918_lsamp_wd.pkl',
#                     'wse':r'C:\LS\10_IO\2112_Agg\outs\agg2\r8\SJ\direct\20220918\lsamp_wse\SJ_r8_direct_0918_lsamp_wse.pkl',
#                     },
#                 'filter':{
#                     'catMasks':'C:\\LS\\10_IO\\2112_Agg\\outs\\agg2\\r5\\SJ\\filter\\20220909\\cMasks\\SJ_r5_filter_0909_cMasks.pkl',
#                     'arsc':None, #only need to compute this once
#                     'wd':r'C:\LS\10_IO\2112_Agg\outs\agg2\r7\SJ\filter\20220911\lsamp_wd\SJ_r7_filter_0911_lsamp_wd.pkl',
#                     'wse':r' C:\LS\10_IO\2112_Agg\outs\agg2\r7\SJ\filter\20220911\lsamp_wse\SJ_r7_filter_0911_lsamp_wse.pkl'
#                     }
#                 },
#         **kwargs):
#     return run_expo(fp_d=fp_lib[method], case_name = 'SJ', method=method,run_name='r8', **kwargs)
#===============================================================================

if __name__ == "__main__":
    
    #SJ_expo_dev()
    SJ_expo_run(method='filter', run_name='r11')
 
    
    print('finished in %.2f'%((now()-start).total_seconds()))