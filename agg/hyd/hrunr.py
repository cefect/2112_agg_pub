'''
Created on Jan. 16, 2022

@author: cefect

running hyd models (from python IDE)

explore errors in impact estimates as a result of aggregation using hyd model depths
    let's use hp.coms, but not Canflood
    using damage function csvs from figurero2018 (which were pulled from a db)
    intermediate results only available at Session level (combine Study Areas)
    
    
key questions
    how do errors relate to depth?
    how much total error might I expect?
'''

#===============================================================================
# imports-----------
#===============================================================================
import os, datetime, math, pickle, copy, sys
import qgis.core
import pandas as pd
import numpy as np

np.random.seed(100)
#===============================================================================
# import scipy.stats 
# import scipy.integrate
# print('loaded scipy: %s'%scipy.__version__)
#===============================================================================

start = datetime.datetime.now()
print('start at %s' % start)

 
#===============================================================================
# custom imports--------
#===============================================================================
from agg.hyd.hscripts import Model, ModelStoch, get_all_pars, view, Error
from definitions import model_pars_fp
#===========================================================================
# #presets
#===========================================================================
 



    
#===============================================================================
# FUNCTIONS-------
#===============================================================================
def get_modelIDs( #retrieve thoe modelIDs from the rparamtere file
        pars_fp=model_pars_fp):
 
        
    assert os.path.exists(pars_fp), 'bad model_pars_fp: %s'%pars_fp
    mid_l = pd.read_excel(pars_fp, comment='#')['modelID'].dropna().astype(int).tolist()
    print('%i modelIDs found in %s'%(len(mid_l), os.path.basename(pars_fp)))
    return mid_l
    
    
def get_pars(#retrieving and pre-checking parmeter values based on model ID
            modelID,
            #file with preconfigrued runs
             pars_fp = model_pars_fp,
             ):
    
    """todo... use the HydSession"""
    #===========================================================================
    # load pars file
    #===========================================================================
    from numpy import dtype #needed.  not sure why though
    
    #pars_df_raw.dtypes.to_dict()
    #pars_df_raw = pd.read_csv(pars_fp, index_col=False, comment='#')
    pars_df_raw= pd.read_excel(pars_fp, comment='#')
    
    
    #remove notes comumns
    bxcol = pars_df_raw.columns.str.startswith('~')
    if bxcol.any():
        print('dropping %i/%i columns flagged as notes'%(bxcol.sum(), len(bxcol)))
        
        
    pars_df1 = pars_df_raw.loc[:, ~bxcol].dropna(how='all', axis=1).dropna(subset=['modelID'], how='any').infer_objects()
    
    
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
    assert pars_df.index.name == 'modelID'
 
    assert modelID in pars_df.index, 'failed to find %i in index'%modelID
    
    #===========================================================================
    # check
    #===========================================================================
    #possible paramater combinations
    pars_lib = copy.deepcopy(Model.pars_lib)
 
    
    #value check
    for id, row in pars_df.iterrows():
        """replacing nulls so these are matched'"""
        #bx = pars_df.fillna('nan').eq(pre_df.fillna('nan').loc[id, :])
    
        """
        view(pars_df.join(bx.sum(axis=1).rename('match_cnt')).sort_values('match_cnt', ascending=False))
        """
        
        #check each parameter value
        for varnm, val in row.items():
            
            #skippers
            if varnm in ['tag']:
                continue 
 
            allowed_l = pars_lib[varnm]['vals']
            
            if not val in allowed_l:
                
                raise Error(' modelID=%i \'%s\'=\'%s\' not in allowed set\n    %s'%(
                id, varnm, val, allowed_l))
        
    #type check
    for varnm, dtype in pars_df.dtypes.items():
        if varnm in ['tag']:continue
        v1 = pars_lib[varnm]['vals'][0]
        
        if isinstance(v1, str):
            assert dtype.char=='O'
        elif isinstance(v1, int):
            assert 'int' in dtype.name
        elif isinstance(v1, float):
            assert 'float' in dtype.name
 
 
    #===========================================================================
    # get kwargs
    #===========================================================================
    raw_d = pars_df.loc[modelID, :].to_dict()
    
    #renamp types
    """
    this is an artifact of loading parameters from pandas
    not very nice.. but not sure how else to preserve the type checks"""
    for k,v in copy.copy(raw_d).items():
        if isinstance(v, str):
            continue
        elif 'int' in type(v).__name__:
            raw_d[k] = int(v)
            
        elif 'float' in type(v).__name__:
            raw_d[k] = float(v)
        
    
    return raw_d
     
    
def run_autoPars( #run a pre-defined model configuration
        modelID=0,
        **kwargs):
    print('START on %i w/ %s'%(modelID, kwargs))
    #retrieve preconfigured parameters
    model_pars = get_pars(modelID)
    
    #reconcile passed parameters
    for k,v in copy.copy(model_pars).items():
        if k in kwargs:
            if not v==kwargs[k]:
                print('WARNING!! passed parameter \'%s\' conflicts with pre-loaded value...replacing'%(k))
                model_pars[k] = kwargs[k] #overwrite these for reporting
 
        
    
    return run(
        modelID=modelID,
        cat_d=copy.deepcopy(model_pars),
        **{**model_pars, **kwargs} #overwrites model_pars w/ kwargs (where theres a conflict)
        )
    
def run_auto_dev( #special dev runner
        iters=3, trim=True, name='hyd5_dev',**kwargs):
    
    return run_autoPars(iters=iters, trim=trim, name=name, **kwargs)
    
 

def run( #run a basic model configuration
        #=======================================================================
        # #generic
        #=======================================================================
        tag='r2_base',
        name='hyd5',
        overwrite=True,
        trim=False,
        
        #=======================================================================
        # write control
        #=======================================================================
        write=True,
        exit_summary=True,
        write_lib=True, #enter the results into the library
        write_summary=True, #write the summary sheet
        #=======================================================================
        # #data
        #=======================================================================
        studyArea_l = None, #convenience filtering of proj_lib
        proj_lib = None,
        
        #=======================================================================
        # session pars
        #=======================================================================
        prec=3,        

        #stochasticity
        iters=50,
        
        #=======================================================================
        # #parameters
        #=======================================================================
        #aggregation
        aggType = 'none', aggLevel = 0,
        
        #down scaling (asset values)
        tval_type = 'rand', normed=True, #generating true asset values
        dscale_meth='none', #downscaling to the aggreated finv
        
        #sampling (geo). see Model.build_sampGeo()
        sgType = 'poly', 
        #sampling (method). see Model.build_rsamps()
        samp_method = 'zonal', zonal_stat='Mean',  # stats to use for zonal. 2=mean
        
        #raster downSampling and selection  (StudyArea.get_raster())
        dsampStage='none', resolution=5, downSampling='none', 
        severity = 'hi', 
        
 
        #vfunc selection
        vid = 798, 
        
        #=======================================================================
        # meta
        #=======================================================================
        cat_d={},
        
 
        **kwargs):
    print('START run w/ %s.%s and iters=%i'%(name, tag, iters))
    #===========================================================================
    # parameter logic override
    #===========================================================================
    """these overrides are an artifact of having overly flexible parameters
    move to model_retrieve?"""
    if not tval_type in ['rand']:
        iters=1
        
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
    #===========================================================================
    # execute
    #===========================================================================
    with ModelStoch(tag=tag,proj_lib=proj_lib,overwrite=overwrite, trim=trim, name=name,
                    iters=iters,write=write,exit_summary=exit_summary,prec=prec,
                 bk_lib = {
                     'finv_agg_d':dict(aggLevel=aggLevel, aggType=aggType),
                     
                     'drlay_d':dict( severity=severity, resolution=resolution, downSampling=downSampling, dsampStage=dsampStage),

                     'rsamps':dict(samp_method=samp_method, zonal_stat=zonal_stat),
                     
                     'finv_sg_d':dict(sgType=sgType),
                     
                     'tvals_raw':dict(normed=normed, tval_type=tval_type),
                     'tvals':dict( dscale_meth=dscale_meth),
                     'rloss':dict(vid=vid),
                                          
                     },
                 **kwargs) as ses:
        
        #special library override for dev runs
        if tag=='dev':
            lib_dir = os.path.join(ses.out_dir, 'lib')
            if not os.path.exists(lib_dir):os.makedirs(lib_dir)
        else:
            lib_dir = None
        
        #execute parts in sequence
        ses.run_dataGeneration()
        ses.run_intersection()
        ses.run_lossCalcs()
        
        #write results
        if write_summary:
            ses.write_summary()
        if write_lib:
            ses.write_lib(lib_dir=lib_dir, cat_d=cat_d)
 
        
        data_d = ses.data_d
        ofp_d = ses.ofp_d
        
    print('\nfinished %s'%tag)
    
    return data_d, ofp_d


 
 
def dev():
    
    return run(
        tag='dev',modelID = 1,
        compiled_fp_d = {
        'finv_agg_d':r'C:\LS\10_OUT\2112_Agg\outs\hyd5\dev\20220414\working\finv_agg_d_hyd5_dev_0414.pickle',
        'finv_agg_mindex':r'C:\LS\10_OUT\2112_Agg\outs\hyd5\dev\20220414\working\finv_agg_mindex_hyd5_dev_0414.pickle',
        'tvals_raw':r'C:\LS\10_OUT\2112_Agg\outs\hyd5\dev\20220414\working\tvals_raw_hyd5_dev_0414.pickle',
        'tvals':r'C:\LS\10_OUT\2112_Agg\outs\hyd5\dev\20220414\working\tvals_hyd5_dev_0414.pickle',
        'drlay_d':r'C:\LS\10_OUT\2112_Agg\outs\hyd5\dev\20220414\working\drlay_d_hyd5_dev_0414.pickle',
        'finv_sg_d':r'C:\LS\10_OUT\2112_Agg\outs\hyd5\dev\20220414\working\finv_sg_d_hyd5_dev_0414.pickle',
        'rsamps':r'C:\LS\10_OUT\2112_Agg\outs\hyd5\dev\20220414\working\rsamps_hyd5_dev_0414.pickle',
            },
        
 
        iters=3,
        #=======================================================================
        # aggType='gridded', 
        # aggLevel=0,
        # 
        # dsampStage='wsl',
        # resolution=100, downSampling='Average',
        #=======================================================================
 
        trim=True,
        studyArea_l = ['Calgary'],
        overwrite=True,
        vid=1001,
        )
    
    
    
 
 
if __name__ == "__main__": 
    
    #dev()
 
    output=run_auto_dev(modelID=78, write=False, write_lib=False,
                        compiled_fp_d={ 
     
                            },
                        #studyArea_l = ['obwb'],
                        )
 
 
#===============================================================================
#     output=run_autoPars(modelID=27, write=True, write_lib=False,
#                         name='hyd5_dev', 
#                         #studyArea_l = ['obwb'],
#                         compiled_fp_d={
#         'finv_agg_d':r'C:\LS\10_OUT\2112_Agg\outs\hyd5_dev\a50r0_aws\20220412\working\finv_agg_d_hyd5_dev_a50r0_aws_0412.pickle',
#         'finv_agg_mindex':r'C:\LS\10_OUT\2112_Agg\outs\hyd5_dev\a50r0_aws\20220412\working\finv_agg_mindex_hyd5_dev_a50r0_aws_0412.pickle',
#         'tvals_raw':r'C:\LS\10_OUT\2112_Agg\outs\hyd5_dev\a50r0_aws\20220412\working\tvals_raw_hyd5_dev_a50r0_aws_0412.pickle',
# 
#                             })
#===============================================================================
 
    
 
        
 
    tdelta = datetime.datetime.now() - start
    print('finished in %s' % (tdelta))