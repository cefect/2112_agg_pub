'''
Created on Mar. 25, 2022

@author: cefect
'''
import os
proj_dir = r'C:\LS\09_REPOS\02_JOBS\2112_Agg\cef'
src_dir = proj_dir
src_name='agg'

model_pars_fp = r'C:\LS\10_IO\2112_Agg\ins\hyd\model_pars\hyd_modelPars_0419.xls'

#CanFlood format vfunc lib
cf_vfuncLib_fp = r'C:\LS\10_IO\2112_Agg\ins\vfunc\CanFlood_curves_0414.xls'

proj_lib =     { #studyArea data for hyd.runr
            'Ok':{
                  'EPSG': 2955, 
                 'finv_fp': r'C:\LS\10_IO\2112_Agg\ins\hyd\obwb\inventory\microsoft_CanBldgFt_obwb_0715.gpkg', 
                 
                 #just for testing?
                 'aoi':r'C:\LS\02_WORK\NRC\2112_Agg\04_CALC\hyd\OBWB\aoi\obwb_aoiT01.gpkg',
                 
               'wse_fp_d':{ #10x10
                    'low':r'C:\LS\10_IO\2112_Agg\ins\hyd\obwb\wsl\wse_sB_1223\10\noGW\wse_sB_0100_1218_10_noGW.tif',  
                     'mid':r'C:\LS\10_IO\2112_Agg\ins\hyd\obwb\wsl\wse_sB_1223\10\noGW\wse_sB_0200_1218_10_noGW.tif',
                      'hi':r'C:\LS\10_IO\2112_Agg\ins\hyd\obwb\wsl\wse_sB_1223\10\noGW\wse_sB_0500_1218_10_noGW.tif',
                     },
               'dem_fp_d':{
 
                    10:r'C:\LS\10_IO\2112_Agg\ins\hyd\obwb\dem\obwb_NHC2020_DEM_20210804_10_aoi08.tif'
                    #10:r'C:\LS\10_IO\2112_Agg\ins\hyd\obwb\dem\obwb_NHC2020_DEM_20210804_10_aoi07_0419.tif',
                       },
                    }, 
            
            'noise':{ #generated using prep.py
                  'EPSG': 2955, 
                  #manual w/ random points and hex grid
                 'finv_fp': r'C:\LS\10_IO\2112_Agg\ins\hyd\noise\0512\finv_noise_0512.gpkg', 
                 'aoi':r'C:\LS\02_WORK\NRC\2112_Agg\04_CALC\hyd\OBWB\aoi\aoi01_rand_0511.gpkg',
                 
               'wse_fp_d':{ #10x10
                     'hi':r'C:\LS\10_IO\2112_Agg\ins\hyd\noise\20220511\wse_raw_prep_rand_0511_noGW.tif',
                     },
               'dem_fp_d':{
                    10:r'C:\LS\10_IO\2112_Agg\ins\hyd\noise\20220511\dem_prep_rand_0511.tif'
                       },
                    }, 
            
            'LM': {
                'EPSG': 3005, 
                'finv_fp': r'C:\LS\10_IO\2112_Agg\ins\hyd\LMFRA\finv\microsoft_CanBldgFt_LM_0715.gpkg', 
                'aoi':r'C:\LS\02_WORK\NRC\2112_Agg\04_CALC\hyd\LMFRA\aoi\LMFRA_aoiT01_0119.gpkg',
                
                'wse_fp_d':{ #10x10
                      'hi':r'C:\LS\10_IO\2112_Agg\ins\hyd\LMFRA\wsl\noGW_0514\AG4_Fr_0500_WL_simu_0415_aoi09_0304_noGW.tif', 
                      'mid':r'C:\LS\10_IO\2112_Agg\ins\hyd\LMFRA\wsl\noGW_0514\AG4_Fr_0200_WL_simu_0415_aoi09_0419_noGW.tif',
                      'low':r'C:\LS\10_IO\2112_Agg\ins\hyd\LMFRA\wsl\noGW_0514\AG4_Fr_0100_WL_simu_0415_aoi09_0304_noGW.tif',
                      },
                'dem_fp_d':{
                     #1:r'C:\LS\10_IO\2112_Agg\ins\hyd\LMFRA\dem\LMFRA_NHC2019_dtm_01_aoi09_0304.tif', #not exactly 1x1
                     #5:r'C:\LS\10_IO\2112_Agg\ins\hyd\LMFRA\dem\LMFRA_NHC2019_dtm_05_aoi09_0304.tif',
                     10:r'C:\LS\10_IO\2112_Agg\ins\hyd\LMFRA\dem\LMFRA_NHC2019_dtm_10_aoi09_0419.tif'
                      },
                    }, 
              
            'Calgary': {
                'EPSG': 3776, 
                'finv_fp': r'C:\LS\10_IO\2112_Agg\ins\hyd\Calgary\finv\microsoft_CanBldgFt_calgary_0715.gpkg', 
    
                'aoi':r'C:\LS\02_WORK\NRC\2112_Agg\04_CALC\hyd\Calgary\aoi\calgary_aoiT01_0119.gpkg',
   
                'wse_fp_d':{ #10x10
                    'hi':r'C:\LS\10_IO\2112_Agg\ins\hyd\Calgary\wse\sc0\noGW_0514\IBI_2017CoC_s0_0500_170729_aoi01_0304_noGW.tif', 
                    'mid':r'C:\LS\10_IO\2112_Agg\ins\hyd\Calgary\wse\sc0\noGW_0514\IBI_2017CoC_s0_0200_170729_aoi01_0304_noGW.tif',
                    'low':r'C:\LS\10_IO\2112_Agg\ins\hyd\Calgary\wse\sc0\noGW_0514\IBI_2017CoC_s0_0100_170729_aoi01_0304_noGW.tif',
                      },
                'dem_fp_d':{
                    2:r'C:\LS\10_IO\2207_fpolish\ins\calgary\dem\CoC_CS_DEM_2019_r02_aoi04_0819.tif'
 
    
                      },
                        }, 
                        
            'SJ': {
                'EPSG': 2953, 
                'finv_fp': r'C:\LS\10_IO\2112_Agg\ins\hyd\SJ\microsoft_aoi17_0819.geojson',
                'wse_fp_d':{ #1x1
                    'hi':r'C:\LS\10_IO\2112_Agg\ins\hyd\SJ\GeoNB_LSJ_FloodExtent2018_aoi17_fnd_noGW_r01_0828.tif', 
 
                      },
                'dem_fp_d':{
                    1:r'C:\LS\10_IO\2112_Agg\ins\hyd\SJ\NBDNR2015_r01_aoi17_fnd_e2953_0909.tif',
                    #===========================================================
                    # 2:r'C:\LS\10_IO\2112_Agg\ins\hyd\SaintJohn\dem\NBDNR2015_r02_aoi17_fnd_e2953_0819.tif',
                    # 16:r'C:\LS\10_IO\2112_Agg\ins\hyd\SaintJohn\dem\NBDNR2015_r16_aoi17_fnd_e2953_0819.tif'
                    #===========================================================
 
    
                      },
                            }, 

            #===================================================================
            # 'dP': {
            #     'EPSG': 2950, 
            #     'finv_fp': 'C:\\LS\\10_IO\\2112_Agg\\ins\\hyd\\dP\\finv\\microsoft_20210506_aoi03_0116.gpkg', 
            #     'dem': 'C:\\LS\\10_IO\\2112_Agg\\ins\\hyd\\dP\\dem\\HRDEM_CMM2_0722_fild_aoi03_0116.tif', 
 
            #     'aoi':r'C:\LS\02_WORK\NRC\2112_Agg\04_CALC\hyd\dP\aoiT01_202220118.gpkg',
            #     },
            #===================================================================
            }

logcfg_file=r'C:\LS\09_REPOS\01_COMMON\coms\logger.conf'

root_dir=r'C:\LS\10_IO\2112_Agg'
wrk_dir=root_dir

base_resolution=10

#max_cores = max(multiprocessing.cpu_count(), 8)