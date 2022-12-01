'''
Created on Feb. 24, 2022

@author: cefect

executing hyd.model from command line

see batch scripts
'''
import sys, argparse
from agg.hyd.hrunr import run_autoPars, run_auto_dev

if __name__ == "__main__":
    print(sys.argv)
    #===========================================================================
    # setup argument parser 
    #===========================================================================
    parser = argparse.ArgumentParser(prog='hyd',description='execute hyd.models')
    #add arguments
    parser.add_argument('modelID', help='specify the model code', type=int)
    parser.add_argument("-name",'-n', help='name for the run group', type=str, default='hyd') #this defaults to None if not passed
    parser.add_argument("-dev",'-d', help='flag for dev runs', action='store_true')
    parser.add_argument("-write",'-w', help='flag for writing intermediate pickels', action='store_true') #defaults to False
    
    
    args = parser.parse_args()
    kwargs = vars(args)
    print('parser got these kwargs: \n    %s'%kwargs) #print all the parsed arguments in dictionary form
    
    dev = kwargs.pop('dev')
    print('\n\nSTART (dev=%s) \n\n\n\n'%dev)
    if dev:
        run_auto_dev(**kwargs)
        
    else:
        run_autoPars(**kwargs)