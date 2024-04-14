#!/usr/bin/python 

import platform
from numpy import ctypeslib,empty,array,exp,ascontiguousarray
from ctypes import c_float,c_double,c_int

def chi2_dist(X,Y=None,K=None):
    if X.dtype==c_float:
       datatype=c_float
    else:
       datatype=c_double
    
    X = ascontiguousarray(X,datatype) # make array continguous
    n,xdim = X.shape
    if Y==None:
        m,ydim = n,xdim
    else:
        Y = ascontiguousarray(Y,datatype)
        m,ydim = Y.shape

    if xdim != ydim:
        print "Input dimension mismatch: %d != %d", (xdim, ydim)
        raise RuntimeError
    
    if (K==None):
        K = empty( (n,m), datatype)
    elif (K.shape != (n,m)):
        K = empty( (n,m), datatype)

    try:
        chi2lib = ctypeslib.load_library("libchi2.so",".") # adapt path to library location
    except:
        print "Unable to load chi2-library"
        raise RuntimeError
        
    if Y == None:
        if datatype==c_float:
            chi2_routine = chi2lib.chi2sym_distance_float
        else:
            chi2_routine = chi2lib.chi2sym_distance_double
    
        chi2_routine.restype = datatype
        chi2_routine.argtypes = [c_int, \
            c_int, ctypeslib.ndpointer(dtype=datatype, ndim=2, flags='C_CONTIGUOUS'), \
                   ctypeslib.ndpointer(dtype=datatype, ndim=2, flags='C_CONTIGUOUS') ]
        meanK = chi2_routine(dim, n, X, K)
    else:
        if datatype==c_float:
            chi2_routine = chi2lib.chi2_distance_float
        else:
            chi2_routine = chi2lib.chi2_distance_double
    
        chi2_routine.restype = datatype
        chi2_routine.argtypes = [c_int, \
            c_int, ctypeslib.ndpointer(dtype=datatype, ndim=2, flags='C_CONTIGUOUS'), \
            c_int, ctypeslib.ndpointer(dtype=datatype, ndim=2, flags='C_CONTIGUOUS'), \
                   ctypeslib.ndpointer(dtype=datatype, ndim=2, flags='C_CONTIGUOUS') ]
        meanK = chi2_routine(dim, n, X, m, Y, K)
    
    return K,meanK


def chi2_kernel(X,Y=None,K=None,oldmeanK=None):
    K,meanK = chi2_dist(X,Y,K)
    if (oldmeanK == None):
        K = exp(-0.5*K/meanK)
    else:    
        K = exp(-0.5*K/oldmeanK)
    return K,meanK


if __name__ == "__main__":
    from numpy import mean,log
    from numpy.random import random_integers
    from time import time
    dim=321 
    xsize=2001
    ysize=1003
    X1 = array( random_integers(1, 10, xsize*dim), c_float)
    X1.shape = (-1,dim)
    X2 = array( random_integers(1, 10, ysize*dim), c_float)
    X2.shape = (-1,dim)
    
    before = time()
    K,meanK = chi2_kernel(X1)
    print "chi2, symmetric %dx%d, %d-dim float: runtime %f s" % (xsize,xsize,dim,time()-before)
    print "mean(log(K)) = %f (should be -0.5)" % mean(mean(log(K)))
    
    before = time()
    K,meanK = chi2_kernel(X1,X2)
    print "chi2, asymmetric %dx%d, %d-dim float: runtime %f s" % (xsize,ysize,dim,time()-before)
    print "mean(log(K)) = %f (should be -0.5)" % mean(mean(log(K)))

    X1 = array( random_integers(1, 10, xsize*dim), c_double)
    X1.shape = (-1,dim)
    X2 = array( random_integers(1, 10, ysize*dim), c_double)
    X2.shape = (-1,dim)
    
    before = time()
    K,meanK = chi2_kernel(X1)
    print "chi2,symmetric %dx%d, %d-dim double: runtime %f s" % (xsize,xsize,dim,time()-before)
    print "mean(log(K)) = %f (should be -0.5)" % mean(mean(log(K)))

    before = time()
    K,meanK = chi2_kernel(X1,X2)
    print "chi2, asymmetric %dx%d, %d-dim double: runtime %f s" % (xsize,ysize,dim,time()-before)
    print "mean(log(K)) = %f (should be -0.5)" % mean(mean(log(K)))
