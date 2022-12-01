'''
Created on May 12, 2022

@author: cefect

TODO: 
    fix aggLevel=1 constructor
    compute some radius or longest dimension of aggregated assets
        sqrt(area)?
    compute actual child count
'''

 
import os, datetime, math, pickle, copy, sys
import numpy as np
np.random.seed(100)

start = datetime.datetime.now()
print('start at %s' % start)
from agg.hydE.hydE_scripts import ExpoRun


def run( #run a basic model configuration
        #=======================================================================
        # #generic
        #=======================================================================
        tag='tag',
        name='hydE',
        overwrite=True,
        trim=False,

        #=======================================================================
        # write control
        #=======================================================================
        write=True,
        exit_summary=True,
        write_lib=True, #enter the results into the library
 
 
        #=======================================================================
        # #data
        #=======================================================================
        studyArea_l = None, #convenience filtering of proj_lib
        proj_lib = None,
        
        #running from teh catalog
        catalog_fp=None,index_col=None,dkey_skip_l=['difrlay_lib'],
        
        #=======================================================================
        # session pars
        #=======================================================================
        prec=3,        

        #=======================================================================
        # #parameters
        #=======================================================================
        #raster downSampling and selection  (StudyArea.get_raster())
        iters=3, #resolution iterations
        dsampStage='pre', downSampling='Average', severity = 'hi', 
        
        
        #aggregation
        aggType = 'convexHulls', aggIters = 3,
        
        #sampling (geo). see Model.build_sampGeo()
        sgType = 'poly', 
        
        #sampling (method). see Model.build_rsamps()
        samp_method = 'zonal', zonal_stat='Mean',  # stats to use for zonal. 2=mean
        
        #outputting
        compression='med',

        #=======================================================================
        # debug
        #=======================================================================
        phase_l=['depth', 'expo'],

        **kwargs):
    print('START run w/ %s.%s and '%(name, tag))
 
    #===========================================================================
    # study area filtering
    #===========================================================================
    if proj_lib is None:
        from definitions import proj_lib
    
    if not studyArea_l is None:
        print('filtering studyarea to %i: %s'%(len(studyArea_l), studyArea_l))
        miss_l = set(studyArea_l).difference(proj_lib.keys())
        assert len(miss_l)==0, 'passed %i studyAreas not in proj_lib: %s'%(len(miss_l), miss_l)
        proj_lib = {k:v for k,v in proj_lib.items() if k in studyArea_l}
        
    
    #indexing parameters for catalog
    id_params={
                **dict(downSampling=downSampling, dsampStage=dsampStage, severity=severity),
                **dict(aggType=aggType, samp_method=samp_method)}
    #===========================================================================
    # execute
    #===========================================================================
    with ExpoRun(tag=tag,proj_lib=proj_lib,overwrite=overwrite, trim=trim, name=name,
                     write=write,exit_summary=exit_summary,prec=prec,phase_l=phase_l,
                 bk_lib = {
                     'drlay_lib':dict( severity=severity, downSampling=downSampling, dsampStage=dsampStage, iters=iters),
                     'finv_agg_lib':dict(aggType=aggType, iters=aggIters),
                     'finv_sg_lib':dict(sgType=sgType),
                     'rsamps':dict(samp_method=samp_method, zonal_stat=zonal_stat),
                     'res_dx':dict(),
                     'dataExport':dict(compression=compression, id_params=id_params)
                     },
                 **kwargs) as ses:
        
        #=======================================================================
        # precompiole from catalog
        #=======================================================================
        if not catalog_fp is None:
            ses.compileFromCat(catalog_fp=catalog_fp,id_params=id_params, index_col=index_col, dkey_skip_l=dkey_skip_l)
        #=======================================================================
        # call each phase
        #=======================================================================
        if 'depth' in phase_l:
            ses.runDownsample()
        
        if 'diff' in phase_l:
            ses.runDiffs()
            
        if 'expo' in phase_l:
            ses.runExpo()
 
        #=======================================================================
        # write results to library
        #=======================================================================
        if write_lib:
            ses.write_lib(id_params=id_params)

    print('\nfinished %s'%tag)
    
    return 


def dev():
    return run(
        trim=True, name='hydEdev',
        tag='dev',
 
 
        compiled_fp_d={
        #=======================================================================
        # 'drlay_lib':r'C:\LS\10_OUT\2112_Agg\outs\hydEdev\dev\20220515\working\drlay_lib_hydEdev_dev_0515.pickle',
        # 'noData_cnt':r'C:\LS\10_OUT\2112_Agg\outs\hydEdev\dev\20220515\working\noData_cnt_hydEdev_dev_0515.pickle',
        # 'rstats':r'C:\LS\10_OUT\2112_Agg\outs\hydEdev\dev\20220515\working\rstats_hydEdev_dev_0515.pickle',
        # 'wetStats':r'C:\LS\10_OUT\2112_Agg\outs\hydEdev\dev\20220515\working\wetStats_hydEdev_dev_0515.pickle',
        # 'gwArea':r'C:\LS\10_OUT\2112_Agg\outs\hydEdev\dev\20220515\working\gwArea_hydEdev_dev_0515.pickle',
        #=======================================================================

 
 
             },
        studyArea_l=['obwb'],
        #catalog_fp=r'C:\LS\10_OUT\2112_Agg\lib\hydEdev\hydEdev_run_index.csv',
        #phase_l=['depth', 'expo']
        )

def r01(**kwargs):
    rkwargs = dict(
        iters=8, downSampling='Average',dsampStage='pre',
        aggIters=5,samp_method='zonal',
        write_lib=True, 
        catalog_fp=r'C:\LS\10_OUT\2112_Agg\lib\hydE01\hydE01_run_index.csv',
        #phase_l=['depth']
        )    
    return run(name='hydE01', **{**rkwargs, **kwargs})

def r02(**kwargs):
    rkwargs = dict(
        iters=8, downSampling='Average',dsampStage='pre',
        aggIters=8,samp_method='zonal',
        write_lib=True, 
        catalog_fp=r'C:\LS\10_OUT\2112_Agg\lib\hydR02\hydR02_run_index_copy.csv',
        index_col=list(range(5)), #loading a RastRun catalog
        #phase_l=['depth']
        )    
    return run(name='hydE02', **{**rkwargs, **kwargs})

def pre_cvh():
    return r02(
        tag='pre_cvh', aggType='convexHulls',
        compiled_fp_d={
                    'finv_agg_lib':r'C:\LS\10_OUT\2112_Agg\outs\hydE02\pre_cvh\20220516\working\finv_agg_lib_hydE02_pre_cvh_0516.pickle',
        'faggMap':r'C:\LS\10_OUT\2112_Agg\outs\hydE02\pre_cvh\20220516\working\faggMap_hydE02_pre_cvh_0516.pickle',
        'rsamps':r'C:\LS\10_OUT\2112_Agg\outs\hydE02\pre_cvh\20220516\working\rsamps_hydE02_pre_cvh_0516.pickle',
        'rsampStats':r'C:\LS\10_OUT\2112_Agg\outs\hydE02\pre_cvh\20220516\working\rsampStats_hydE02_pre_cvh_0516.pickle',
        'rsampErr':r'C:\LS\10_OUT\2112_Agg\outs\hydE02\pre_cvh\20220516\working\rsampErr_hydE02_pre_cvh_0516.pickle',
            }
        )
    
if __name__ == "__main__": 
    
    #dev()
    pre_cvh()
 

    tdelta = datetime.datetime.now() - start
    print('finished in %s' % (tdelta))