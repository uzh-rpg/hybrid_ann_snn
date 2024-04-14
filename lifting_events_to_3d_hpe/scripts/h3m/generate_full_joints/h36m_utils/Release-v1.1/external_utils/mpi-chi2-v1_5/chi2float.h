/* calculate the chi2-distance between two vectors/histograms of unknown alignment/size*/
float chi2_float(const int dim, const float* const x, const float* const y);

/* calculate the chi2-distance matrix between a set of vectors/histograms. */
float chi2sym_distance_float(const int dim, const int nx, const float* const x, 
                             float* const K);

/* calculate the chi2-distance matrix between two sets of vectors/histograms. */
float chi2_distance_float(const int dim, const int nx, const float* const x, 
                          const int ny, const float* const y, float* const K);
