/* calculate the chi2-distance between two vectors/histograms of unknown alignment/size*/
double chi2_double(const int dim, const double* const x, const double* const y);

/* calculate the chi2-distance matrix between a set of vectors/histograms. */
double chi2sym_distance_double(const int dim, const int nx, const double* const x, 
                               double* const K);

/* calculate the chi2-distance matrix between two sets of vectors/histograms. */
double chi2_distance_double(const int dim, const int nx, const double* const x, 
                          const int ny, const double* const y, double* const K);
