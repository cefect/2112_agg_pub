'''
Created on Aug. 22, 2022

@author: cefect
'''

from definitions import proj_lib
from agg.hydR.dsc.scripts import runr
 


def SJ_0821():
    proj_name = 'dsc'
    proj_d = proj_lib['SJ']
    return runr(
        proj_name=proj_name, run_name='SJ_r1',
        downscale=50,
        wse_fp=proj_d['wse_fp_d']['hi'],        
        dem_fp=proj_d['dem_fp_d'][1]
        )
    

if __name__ == "__main__":
    SJ_0821()