#include <mex.h>
#include "chi2double.h"
#include "chi2float.h"

/*
  computes the chi??? distance between the input arguments
  d(X,Y) = sum ((X(i)-Y(i))???)/(X(i)+Y(i))
*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    int i,j,dim,ptsA, ptsB, k,kptsA,kptsB;
    void *vecA, *vecB, *pA,*pB, *dist;
    mwIndex *ir, *jc, *ir2, *jc2;

	if (nrhs == 0)
	{
		mexPrintf("Usage: d = chi2_mex(X,Y);\n");
		mexPrintf("where X and Y are matrices of dimension [dim,npts]\n");
		mexPrintf("\nExample\n a = rand(2,10);\n b = rand(2,20);\n d = chi2_mex(a,b);\n");
		return;
	}

	if (nrhs != 2){
		mexPrintf("2 input arguments expected: A, B");
		return;
	}

	if (mxGetNumberOfDimensions(prhs[0]) != 2 || mxGetNumberOfDimensions(prhs[1]) != 2)
	{
		mexPrintf("inputs must be two dimensional");
		return;
	}
    
    
    mxClassID the_class = mxGetClassID(prhs[0]);
    if (the_class != mxGetClassID(prhs[1])){
        mexErrMsgTxt("Two histograms need to have the same precision!\n");
    }
    
    if (the_class != mxDOUBLE_CLASS && the_class != mxSINGLE_CLASS) {
        mexErrMsgTxt("Histograms should have single/double precision!\n");
    }
    
	if (mxIsSparse(prhs[0]) && !mxIsSparse(prhs[1])){
		mexErrMsgTxt("Right now only works with both dense or both sparse X and Y matrices!\n");
	}

	ptsA = mxGetN(prhs[0]);
	ptsB = mxGetN(prhs[1]);
	dim = mxGetM(prhs[0]);

	if (mxIsSparse(prhs[0]))
	{
/* Sparse routine. In this case no need to resort to intrinsics */
/* dims * npts, so each column is an example */
	        void *nums, *nums2;
		ir = mxGetIr(prhs[0]);
		jc = mxGetJc(prhs[0]);
		ir2 = mxGetIr(prhs[1]);
		jc2 = mxGetJc(prhs[1]);
		if (the_class == mxDOUBLE_CLASS){
	    	    nums = mxGetPr(prhs[0]);
		    nums2 = mxGetPr(prhs[1]);
		}
		else{
		    nums = mxGetData(prhs[0]);
		    nums2 = mxGetData(prhs[1]);
		}
		plhs[0] = mxCreateDoubleMatrix(ptsA,ptsB,mxREAL);
		dist = (double *)mxGetPr(plhs[0]);
/* Now we just find matches in the ir and jc of the two */
		chi2_sparse_distance(ir, jc, nums, ptsA, ir2, jc2, nums2, ptsB, dist);
	}
	else
	{
		if (the_class == mxDOUBLE_CLASS){
		    vecA = mxGetPr(prhs[0]);
		    vecB = mxGetPr(prhs[1]);
		}
		else{
		    vecA = mxGetData(prhs[0]);
		    vecB = mxGetData(prhs[1]);
		}

		if (dim != mxGetM(prhs[1]))
		{
			mexPrintf("Dimension mismatch");
			return;
		}

	    const mwSize ndims[2] = {ptsA, ptsB};
/*    printf("%d %d\n", ndims[0],ndims[1]); */
    
/*    mxArray* mxdist = mxCreateNumericArray(2, ndims,the_class,mxREAL);    
    dist = (double *)mxGetData(mxdist);*/
        if (the_class == mxSINGLE_CLASS)
	{
            plhs[0] = mxCreateNumericArray(2, ndims, the_class, mxREAL);
            dist = mxGetData(plhs[0]);
            chi2_distance_float(dim,ptsB,(float *)vecB,ptsA,(float *)vecA,(float *)dist);
	}
        else          
 	{
            plhs[0] = mxCreateDoubleMatrix(ptsA,ptsB,mxREAL);
            dist = mxGetPr(plhs[0]);
            chi2_distance_double(dim,ptsB,(double *)vecB,ptsA,(double *)vecA,(double *)dist); 
	}
            
/*    plhs[0] = mxdist;*/
	}    
	return;
}
