#include "chi2_sparse_distance.h"

void chi2_sparse_distance(mwIndex *ir, mwIndex *jc, double *nums, int npoints, mwIndex *ir2, mwIndex *jc2, double *nums2, int npoints2, double *dist)
{
	int i,j, k1, k2;
	int low1, low2, high1, high2;
	double val;
	for(i=0;i<npoints;i++)
	{
		low1 = (int)jc[i], high1 = (int)jc[i+1];
		for(j=0;j<npoints2;j++)
		{
			val = 0;
			low2 = (int)jc2[j], high2 = (int)jc2[j+1];
			k1 = low1, k2 = low2;
			while(k1 < high1 && k2 < high2)
			{
				/* Equal index, then perform chi2 */
				if (ir[k1]==ir2[k2])
				{
					val += (nums[k1] - nums2[k2]) * (nums[k1] - nums2[k2]) / (nums[k1] + nums2[k2]);
					k1++, k2++;
				}
				else 
				{
					if (ir[k1] < ir2[k2])
					{
						val += nums[k1];
						k1++;
					}
					else
					{
						val += nums2[k2];
						k2++;
					}
				}
			}
			if (k1 < high1)
			{
				for(;k1<high1;k1++)
					val += nums[k1];
			}
			if (k2 < high2)
			{
				for(;k2<high2;k2++)
					val += nums2[k2];
			}
			dist[i*npoints2+j] = val;
		}
	}
}
