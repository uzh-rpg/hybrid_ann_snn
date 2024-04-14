// anna PHOGDESCRIPTOR, rewritten in C++ by Andreas Mueller

#include <mex.h>
#include <iostream>
#include <boost/multi_array.hpp>
#include <cassert>
#include <boost/date_time/posix_time/posix_time.hpp>

//IN:
//	bh - matrix of bin histogram values
//	bv - matrix of gradient values 
//   L - number of pyramid levels
//   bin - number of bins
//
//OUT:
//	p - pyramid histogram of oriented gradients (phog descriptor)
using namespace boost::posix_time;

typedef boost::multi_array_ref<double,2> double_2d_ref;
typedef boost::multi_array_ref<double,1> double_1d_ref;
typedef boost::multi_array<double,3> double_3d;

void mexFunction(int nOut, mxArray *pOut[],
		 int nIn, const mxArray *pIn[]) {
	//ptime startTime = microsec_clock::universal_time();
	if((nIn != 4) || (nOut != 1))
		mexErrMsgTxt("Usage: features = phog(bin_hist,grad_val,pyramid_lvl,num_bins)");

	if (!mxIsDouble(pIn[0]) || mxGetNumberOfDimensions(pIn[0]) != 2) {
			mexErrMsgTxt("Usage: masks must be a logical 3-tensor");
		}
	 
	const mwSize *dims= mxGetDimensions(pIn[0]);
	const mwSize *dims_grad = mxGetDimensions(pIn[1]);

	if (dims[0]!=dims_grad[0] || dims[1] !=dims_grad[1]){
			mexErrMsgTxt("Histogram values and gradient values must have same size!");
	}
	if (dims[0]*dims[1] == 0){
			mexErrMsgTxt("No edges!");
	}

	int rows= dims[0];
	int cols = dims[1];

	int num_layers = mxGetScalar(pIn[2]);
	if (num_layers<0 || num_layers>3) {
			mexErrMsgTxt("Number of pyramid levels needs to be between 0 and 3");
	}

	int num_bins = mxGetScalar(pIn[3]);
	int featurelen = 0;
	// calculate feature length from level of pyramid: bin on first, 4 on second, 16 on third and 64 on fourth level
	for(int i=0; i<num_layers+1; i++) {
		featurelen+=num_bins*pow(4,i);	
	}

	// create output matrices, wrap pointers in boost multiarrays

	double_2d_ref histogram(mxGetPr(pIn[0]),boost::extents[cols][rows]);
	double_2d_ref gradient(mxGetPr(pIn[1]),boost::extents[cols][rows]);

	pOut[0]=mxCreateDoubleMatrix(featurelen,1,mxREAL);
	double_1d_ref feature(mxGetPr(pOut[0]),boost::extents[featurelen]);
	std::fill(feature.begin(),feature.end(),0.0);

	double_3d temp_matrix(boost::extents[num_bins][cols][rows]);

	//ptime endTime = microsec_clock::universal_time();	
	//mexPrintf("[Init]Elapsed time: %6.3f seconds\n", time_period(startTime, endTime).length().total_milliseconds()/1000.f);
	//startTime = microsec_clock::universal_time();

	// level 0

	double* temp_dat=temp_matrix.data(); //pointer arithmetic is significantly faster than boost multi_array
	double* feature_dat=feature.data();
	double* gradient_dat=gradient.data();
	double* histogram_dat=histogram.data();

	for(int bin=0; bin<num_bins; bin++){
		for (int i=0; i<cols; i++)
			for (int j=0; j<rows; j++){
				//double new_val=gradient[i][j]*(histogram[i][j]==bin+1);
				double new_val=gradient_dat[i*rows +j]*(histogram_dat[i*rows + j]==bin+1);
				temp_dat[bin*rows*cols + rows*i +j]=new_val;
				//temp_mat[bin][i][j]=new_val;
				feature_dat[bin]+=new_val;
			}
	}
	//endTime = microsec_clock::universal_time();
	//mexPrintf("[1st Level]Elapsed time: %6.3f seconds\n", time_period(startTime, endTime).length().total_milliseconds()/1000.f);
	int counter=num_bins; //start here with higher layers
	//startTime = microsec_clock::universal_time();
	// I apologize. That wasn't my idea...
	for (int layer=1;layer<=num_layers; layer++) {
		int x = floor(cols/pow(2,layer));
		int y = floor(rows/pow(2,layer));
		int xx=0;
		int counterx=0;
		while (xx+x <= cols){
			int yy=0;
			int countery=0;
			while(yy+y <= rows){
				for(int bin=0;bin<num_bins;bin++){
					for(int i=xx; i<xx+x; i++){
						for(int j=yy; j<yy+y; j++)
							feature_dat[counter]+=temp_dat[bin*rows*cols+i*rows+j];
							//feature[counter]+=temp_matrix[bin][i][j];
						}
					counter++;
				}
				yy+=y;
				countery++;
				if (countery==pow(2,layer))
					break;
			}
			xx+=x;
			counterx++;
			if (counterx==pow(2,layer))
				break;
		}

	}
	//endTime = microsec_clock::universal_time();
	//mexPrintf("[higher Level]Elapsed time: %6.3f seconds\n", time_period(startTime, endTime).length().total_milliseconds()/1000.f);
}

