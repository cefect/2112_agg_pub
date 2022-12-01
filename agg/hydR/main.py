'''
Created on May 8, 2022

@author: cefect
'''
import sys, argparse, datetime
start = datetime.datetime.now()
 
from agg.hydR.hydR_runr import run

if __name__ == "__main__":
    print(sys.argv)
 
    #===========================================================================
    # setup argument parser 
    #===========================================================================
    parser = argparse.ArgumentParser(prog='hydR',description='execute hydR.models')
    #add arguments
    parser.add_argument('-tag','-t', help='run label', type=str)
    parser.add_argument("-name",'-n', help='name for the run group', type=str, default='hydR') #this defaults to None if not passed
    parser.add_argument("-write",'-w', help='flag for writing intermediate data', action='store_false') #defaults to True
    parser.add_argument("-write_lib",'-wl', help='flag for storing results to library', action='store_false') #defaults to True
    parser.add_argument("-catalog_fp",help='optional file path of catalog', type=str, default=None)
    
    #build_drlays2
    parser.add_argument("-iters",'-i', help='resolution iterations', type=int, default=7)  
    parser.add_argument("-dsampStage",help='raster downsampling stage', type=str, default='pre')  
    parser.add_argument("-downSampling",help='raster downsampling GDAL.warp method', type=str, default='Average')
    parser.add_argument("-sequenceType",help='raster downsampling sequence key', type=str, default='none')
    
    
    
    
    args = parser.parse_args()
    kwargs = vars(args)
    print('parser got these kwargs: \n    %s'%kwargs) #print all the parsed arguments in dictionary form
 
    print('\n\nSTART (tag=%s) \n\n\n\n'%kwargs['tag'])
    run(**kwargs)
    
    tdelta = datetime.datetime.now() - start
    print('finished in %s' % (tdelta))
    
