#include <stdlib.h>
#include <math.h>
#include "mat.h"
#include "matrix.h"
#include "chi2float.h"

#define NGAMMAS 5

double expsum(const float *dist, int anum, int bnum, int astart, double cur_gamma)
{
  int i,j;
  double sum = 0.0;
  for (i=0;i<anum;i++)
    for (j=0;j<bnum;j++)
      sum += exp(-cur_gamma * dist[(astart + i)* bnum + j]);
  return sum;
}

int main(int argc, char **argv) {
  MATFile *pmat, *pmatout;
  mxArray *mxtrainfeats, *mxtrain_bag_start, *mxdist;
  const char *file = "TrainFeats.mat";
  const char *fileout = "Kernels.mat";
  const char *features_var = "TrainFeats";
  const char *bag_start_var = "train_bag_start";
  const double gammas[] = {0.001,0.005,0.01,0.05,0.1};
  mxArray *mxks[NGAMMAS];
  double *ks[NGAMMAS];
  float *dist, *trainfeats;
  double *train_bag_start;
  unsigned int points, dim, nbags, cur = 0, ptsB, ptsC, maxpts;
  unsigned int i,j,k;

  pmat = matOpen(file, "r");
  if (pmat == NULL) {
    printf("Error opening file %s\n", file);
    return(EXIT_FAILURE);
  }
  pmatout = matOpen(fileout, "w");
  if (pmatout == NULL) {
    printf("Error creating file %s\n", fileout);
    return(EXIT_FAILURE);
  }
  mxtrainfeats = matGetVariable(pmat, features_var);
  if (mxtrainfeats == NULL) {
    printf("Error reading existing features matrix (TrainFeats)\n");
    return(EXIT_FAILURE);
  }
  mxtrain_bag_start = matGetVariable(pmat,bag_start_var);
  if (mxtrain_bag_start == NULL) {
    printf("Error reading existing bag start matrix (train_bag_start)\n");
    return(EXIT_FAILURE);
  }
  /* Suppose trainfeats is single precision and has already been transposed */
  trainfeats = (float *)mxGetData(mxtrainfeats);
  train_bag_start = mxGetPr(mxtrain_bag_start);

  points = mxGetN(mxtrainfeats);
  dim = mxGetM(mxtrainfeats);
  nbags = mxGetM(mxtrain_bag_start);
  
  /* Find the maximal number of points first */
  maxpts = 0;
  for(i=0;i<nbags;i++)
  {
    if(i < nbags - 1)
      ptsB = train_bag_start[i+1] - train_bag_start[i];
    else
      ptsB = points - train_bag_start[i] + 1;
    if(ptsB > maxpts)
      maxpts = ptsB;
  }

  mxdist = mxCreateNumericMatrix(maxpts,points,mxSINGLE_CLASS,mxREAL);
  dist = (float *)mxGetData(mxdist);
  for(i=0;i<NGAMMAS;i++)
  {
    mxks[i] = mxCreateDoubleMatrix(nbags,nbags,mxREAL);
    ks[i] = mxGetPr(mxks[i]);
  }
  for(i=0;i<nbags;i++)
  {
    if(i%5 == 0)
      printf("Processing image: %d\n", i+1);
    cur = train_bag_start[i]-1;
    if(i < nbags - 1)
      ptsB = train_bag_start[i+1] - cur - 1;
    else
      ptsB = points - cur;
    chi2_distance_float(dim,points,trainfeats,ptsB,&trainfeats[cur*dim],dist);
    for(j=i;j<nbags;j++)
    {
      if(j < nbags - 1)
        ptsC = train_bag_start[j+1] - train_bag_start[j];
      else
        ptsC = points - train_bag_start[j] + 1;
      for(k=0;k<NGAMMAS;k++)
        ks[k][j*nbags + i] = ks[k][i*nbags + j] = expsum(dist, ptsC, ptsB, train_bag_start[j]-1, gammas[k]);
    }
  }
  /* Write ks out */
  matPutVariable(pmatout, "K1", mxks[0]);
  matPutVariable(pmatout, "K2", mxks[1]);
  matPutVariable(pmatout, "K3", mxks[2]);
  matPutVariable(pmatout, "K4", mxks[3]);
  matPutVariable(pmatout, "K5", mxks[4]);
  matClose(pmat);
  matClose(pmatout);
}
