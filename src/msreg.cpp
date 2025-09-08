#include <RcppArmadillo.h>
#include <Rcpp.h>
#include <iostream>
#include <cstdlib>  // For std::exit()


// [[Rcpp::depends(RcppArmadillo)]]

// MSREG Class Definition
class MSREG {
public:
  // Public variables
  arma::vec _y1;
  arma::mat _X1, _X2, _Z1, _Z2, _W, _X2_md;
  std::vector<arma::uvec*> _idx_md; // Vector of pointers to store neighbor indices
  arma::vec _b; // Coefficients
  arma::mat _V; // Variance-Covariance Matrix
  int _n, _m, _p;
  bool _cons;
  std::string  _estimator, _vce, _metric;
  int _nn, _order;
  Rcpp::DataFrame df1, df2;
  
  
  // Default constructor
  MSREG() : _cons(true), _estimator("onestep"), _vce("vi"), _metric("mahalanobis"), _nn(1), _order(1) {}
  
  // Parameterized constructor
  MSREG(bool cons, const std::string& estimator, const std::string& vce, const std::string& metric, int nn, double order)
    : _cons(cons), _estimator(estimator), _vce(vce), _metric(metric), _nn(nn), _order(order) {}
  
  // Public Methods
  void load_data(const Rcpp::DataFrame& df1,
                 const Rcpp::DataFrame& df2,
                 const std::string& st_y,
                 const std::vector<std::string>& st_x1,
                 const std::vector<std::string>& st_x2,
                 const std::vector<std::string>& st_z);
  
  void get_data();
  void compute();
  Rcpp::List post_results();
  void get_W();
  void match_X2();
  void get_metric(arma::mat& A);
  void match_zi(const arma::vec& sidx, const arma::rowvec& zi, const arma::mat& A, arma::uvec*& ptr_id);
  void norm_xax(const arma::rowvec& zi, const arma::mat& A, arma::vec& res);
  void onestep();
  void twostep();
  void adjust_y(const arma::vec& b0, arma::vec& y_adj);
  void power_series(int order, const arma::mat& Z_orig, arma::mat& Zp);
  void rmcoll(arma::mat& Z, arma::uvec& p);
  
  // Utility Functions
  void list_class();
  
private:
  
  //arma::vec _y1;
  //arma::mat _X1, _X2, _Z1, _Z2, _W, _X2_md;
  
  // Private Utility Functions
  void msols(const arma::vec& y1, const arma::mat& W, arma::vec& b, arma::mat& V);
  void _msreg_tr_norm(const arma::mat& A, const arma::rowvec& z, arma::vec& res);
  void _msreg_reorder_S2(const arma::mat& Z2, const arma::mat& X2, size_t m);
  void _msreg_msii(const std::string& vce, const arma::vec& y1, const arma::mat& W,
                   const arma::mat& X1, const arma::mat& X2, const arma::mat& Z1, const arma::mat& Z2,
                   double n, double m, double cons, double nn, arma::vec& b, arma::mat& V);
  void _msreg_get_Sigma(const arma::mat& X1, const arma::mat& X2, const arma::mat& Z1,
                        double cons, double m, arma::mat& Sigma);
  void _msreg_msii_b(const arma::vec& y1, const arma::mat& W, double n, double nn,
                     const arma::mat& Sigma, arma::vec& b, arma::mat& invPw);
  void _msreg_msii_V(const std::string& vce, const arma::vec& y1, const arma::mat& W,
                     const arma::mat& X1, const arma::mat& X2, const arma::mat& Z1, const arma::mat& Z2,
                     const arma::mat& Sigma, const arma::vec& b, double n, double m, double cons, double nn,
                     const arma::mat& invPw, arma::mat& V);
  arma::mat _msreg_msii_Gamma(int l, int m, const arma::mat& X2, const arma::mat& Sigma2, const arma::vec& b2);
  //arma::mat norm_xax(const arma::rowvec& zi, const arma::mat& A, arma::vec& res);
};



// Register the Rcpp module
RCPP_MODULE(msreg) {
  Rcpp::class_<MSREG>("MSREG")
  .constructor() // Default constructor
  .constructor<bool, std::string, std::string, std::string, int, int>() // Parameterized constructor
  .method("load_data", &MSREG::load_data, "Load data from data frame objects")
  .method("get_data", &MSREG::get_data)
  .method("compute", &MSREG::compute)
  .method("post_results", &MSREG::post_results)
  //.method("get_W", &MSREG::get_W)
  //.method("match_X2", &MSREG::match_X2)
  //.method("get_metric", &MSREG::get_metric)
  //.method("onestep", &MSREG::onestep)
  //.method("twostep", &MSREG::twostep)
  //.method("adjust_y", &MSREG::adjust_y)
  .method("list_class", &MSREG::list_class);
}


// Public Method Implementations

void MSREG::list_class() {
  Rcpp::Rcout << "Summary of objects in the MSREG class:" << std::endl;
  Rcpp::Rcout << "Constant included: " << (_cons ? "Yes" : "No") << std::endl;
  Rcpp::Rcout << "Estimator: " << _estimator << std::endl;
  Rcpp::Rcout << "Metric: " << _metric << std::endl;
  Rcpp::Rcout << "VCE: " << _vce << std::endl;
  Rcpp::Rcout << "Number of Neighbors (nn): " << _nn << std::endl;
  Rcpp::Rcout << "Order: " << _order << std::endl;
}


// [[Rcpp::depends(RcppArmadillo)]]

void MSREG::load_data(const Rcpp::DataFrame& df1,
                      const Rcpp::DataFrame& df2,
                      const std::string& st_y,
                      const std::vector<std::string>& st_x1,
                      const std::vector<std::string>& st_x2,
                      const std::vector<std::string>& st_z) {
  // Read dependent variable y1 from df1
  _y1 = arma::vec(Rcpp::as<Rcpp::NumericVector>(df1[st_y]));
  _n = _y1.n_rows;
  
  if (_n == 0) {
    Rcpp::stop("Cohort dataset is empty");
  } 
  
  // Read X1 matrix from df1 (if specified) or create an empty matrix
  if (!st_x1.empty()) {
    arma::mat X1(_n, st_x1.size());
    for (size_t i = 0; i < st_x1.size(); ++i) {
      X1.col(i) = Rcpp::as<arma::vec>(df1[st_x1[i]]);
    }
    _X1 = X1; // No transposition needed, directly rows x cols
  } else {
    _X1 = arma::zeros<arma::mat>(_n, 0); // Empty matrix with rows matching _y1
  }

  // Read X2 matrix from df2
  _m = df2.nrows();
  if (_m == 0) {
    Rcpp::stop("Survey dataset is empty");
  } 
  
  if (!st_x2.empty()) {
    arma::mat X2(_m, st_x2.size());
    for (size_t i = 0; i < st_x2.size(); ++i) {
      X2.col(i) = Rcpp::as<arma::vec>(df2[st_x2[i]]);
    }
    _X2 = X2; // No transposition needed, directly rows x cols
  }
  
  // Read Z1 matrix from df1
  if (!st_z.empty()) {
    arma::mat Z1(_n, st_z.size());
    for (size_t i = 0; i < st_z.size(); ++i) {
      Z1.col(i) = Rcpp::as<arma::vec>(df1[st_z[i]]);
    }
    _Z1 = Z1; // No transposition needed, directly rows x cols
  }
  
  // Read Z2 matrix from df2
  if (!st_z.empty()) {
    arma::mat Z2(_m, st_z.size());
    for (size_t i = 0; i < st_z.size(); ++i) {
      Z2.col(i) = Rcpp::as<arma::vec>(df2[st_z[i]]);
    }
    _Z2 = Z2; // No transposition needed, directly rows x cols
  }
  
  //Rcpp::Rcout << _n << std::endl;
  //Rcpp::Rcout << _m << std::endl;
  Rcpp::Rcout << "Data successfully loaded from df1 and df2." << std::endl;
}

void MSREG::compute() {
  
  // Get data 
  get_data();

  // Compute b and V based on the specified method
  if (_estimator == "ols") {
    msols(_y1, _W, _b, _V);
  } else if (_estimator == "onestep") {
    onestep();
  } else if (_estimator == "twostep") {
    twostep();
  } else {
    throw std::invalid_argument("Unknown method: " + _estimator);
  }
  
 
}


// [[Rcpp::export]]
Rcpp::List MSREG::post_results() {

  if (_estimator == "ols") {
    
    // Return as named list with class
    Rcpp::List out = Rcpp::List::create(
      Rcpp::Named("coefficients") = _b,
      Rcpp::Named("VCV")       = _V
    );
    
    out.attr("class") = "msreg";
    return out;
    
  } else if (_estimator == "onestep") {
    onestep();
  } else if (_estimator == "twostep") {
    twostep();
  }
  
  Rcpp::stop("Invalid estimator: " + _estimator);
  
}


  
void MSREG::get_data() {

  // Step 2: Match X2
  match_X2();  // Matches X2 to some criteria 
  
  // Step 3: Get W matrix
  get_W();  // Constructs the W matrix 
  
}

void MSREG::get_W() {
  _W = arma::join_horiz(_X1, _X2_md, _Z1);
  _p = _W.n_cols;
  
  if (_cons) {
    _W = arma::join_horiz(_W, arma::ones<arma::mat>(_n, 1));
    _p += 1;
  }
  //Rcpp::Rcout << "_p in get_W(): " << _p << std::endl;
}


void MSREG::match_X2() {
  arma::mat A;                  // Matrix for metrics
  arma::rowvec zi;              // Row vector for the current point
  arma::vec sidx;               // Random indices for sorting
  arma::uvec* ptr_idi = nullptr; // Pointer to store neighbor indices
  
  // Step 1: Generate random sorting indices
  sidx = arma::randu<arma::vec>(_Z2.n_rows);
  
  // Step 2: Compute the metric matrix (assume get_metric is implemented)
  get_metric(A);
  
  // Step 3: Initialize output matrices
  _X2_md = arma::mat(_y1.n_rows, _X2.n_cols, arma::fill::none); // Placeholder for computed means
  std::vector<arma::uvec*> _idx_md; // Declare as a vector of pointers
  _idx_md.resize(_n, nullptr); // Resize the vector to match the required size
  
  // Step 4: Loop through rows of _Z1
  for (size_t i = 0; i < _Z1.n_rows; ++i) {
    // Extract the i-th row from _Z1
    zi = _Z1.row(i);
    
    // Call match_zi to find neighbors
    match_zi(sidx, zi, A, ptr_idi);
    
    // Store the pointer to neighbor indices in _idx_md
    _idx_md[i] = ptr_idi;
    
    //ptr_idi->t().print("ptr_idi");
    //Rcpp::Rcout << "ptr_n: " << ptr_idi->t() << std::endl;

    // Compute the mean of rows in _X2 corresponding to selected neighbors
    _X2_md.row(i) = arma::mean(_X2.rows(*ptr_idi), 0);
    // Rcpp::Rcout << "ptr_n: " << _X2_md.row(i) << std::endl;
    // Note: Memory for ptr_idi is now managed in _idx_md
  }
}

void MSREG::get_metric(arma::mat& A) {
  arma::mat Z = arma::join_vert(_Z1, _Z2);
  arma::rowvec Zbar = arma::mean(Z, 0);
  Z.each_row() -= Zbar;
  A = (Z.t() * Z) / Z.n_rows;

  if (_metric == "mahalanobis") {
    A = arma::inv_sympd(A);
  } else if (_metric == "euclidean") {
    A = arma::inv_sympd(arma::diagmat(A));
  }
  //A.print("A");
}


void MSREG::match_zi(const arma::vec& sidx, 
                     const arma::rowvec& zi, 
                     const arma::mat& A, 
                     arma::uvec*& ptr_id) {
  arma::uvec idx;       // Stores sorted indices
  arma::vec res;        // Stores computed distances
  size_t n_tie = 0;     // Number of ties where res == 0
  
  // Step 1: Compute norm_xax
  norm_xax(zi, A, res); // Compute the distance metric
  
  // Combine res and sidx into a matrix
  arma::mat combined(_m, 2);
  combined.col(0) = res;   // Primary key
  combined.col(1) = sidx;  // Secondary key
  
  // Create a vector of row indices
  arma::uvec row_indices = arma::regspace<arma::uvec>(1, _m); // 1 to _m
  
  // Sort row indices based on lexicographical order of rows in `combined`
  std::sort(row_indices.begin(), row_indices.end(), [&](size_t i, size_t j) {
    --i; --j; // Convert 1-based indices to 0-based for Armadillo
    if (combined(i, 0) < combined(j, 0)) return true;   // Compare primary key
    if (combined(i, 0) == combined(j, 0)) return combined(i, 1) < combined(j, 1); // Secondary key
    return false;
  });
  
  // Use `row_indices` as the permutation vector
  idx = row_indices - 1; // Convert back to 0-based for Armadillo usage
  
   
  // Step 4: Count ties (where res == 0)
  n_tie = arma::sum(res == 0);
  
  // Step 5: Select first K nearest neighbors, adjusting for ties
  if (n_tie <= 1) {
    ptr_id = new arma::uvec(idx.head(_nn)); // Assign top `_nn` indices
  } else {
    ptr_id = new arma::uvec(idx.head(_nn + n_tie - 1)); // Include ties
  }
  
  //Rcpp::Rcout << "idx.n_rows: " << idx.n_rows << std::endl;
  //Rcpp::Rcout << "_m: " << _m << std::endl;
  //Rcpp::Rcout << "ptr_n: " << ptr_id << std::endl;
  //Rcpp::Rcout << "n_tie: " << n_tie << std::endl;
 
}

void MSREG::norm_xax(const arma::rowvec& zi, const arma::mat& A, arma::vec& res) {
  arma::mat diff = _Z2.each_row() - zi;
  res = arma::sqrt(arma::sum((diff * A) % diff, 1));
  //res.print("res");
}

void MSREG::onestep() {
  _msreg_msii(_vce, _y1, _W, _X1, _X2, _Z1, _Z2, _n, _m, _cons, _nn, _b, _V);
}

void MSREG::twostep() {
  arma::vec y1_adj, b0;
  arma::mat X2_orig = _X2, Z2_orig = _Z2;
  arma::mat placeholder_V;
  
  _msreg_msii(_vce, _y1, _W, _X1, X2_orig, _Z1, Z2_orig, _n, _m, _cons, _nn, b0, placeholder_V);
  adjust_y(b0, y1_adj);
  _msreg_msii(_vce, y1_adj, _W, _X1, _X2, _Z1, _Z2, _n, _m, _cons, _nn, _b, _V);
}

void MSREG::adjust_y(const arma::vec& b0, arma::vec& y_adj) {
  arma::mat Zp_1 = _Z1, Zp_2 = _Z2, B_p, g_Z1, g_Z2;
  arma::uvec p;
  arma::vec b2, lbda = arma::zeros<arma::vec>(_n);
   
  // Power series for Z1
  power_series(_order, _Z1, Zp_1);
  rmcoll(Zp_1, p);
  Zp_1 = arma::join_horiz(arma::ones<arma::mat>(_n, 1), Zp_1);
  
  // Power series for Z2
  power_series(_order, _Z2, Zp_2);
  Zp_2 = Zp_2.cols(p);
  Zp_2 = arma::join_horiz(arma::ones<arma::mat>(_m, 1), Zp_2);
  
  B_p = arma::inv_sympd(Zp_2.t() * Zp_2) * (Zp_2.t() * _X2);
  
  g_Z1 = Zp_1 * B_p;
  g_Z2 = Zp_2 * B_p;
  
  b2 = b0.subvec(_X1.n_cols, _X1.n_cols + _X2.n_cols - 1);
  
  for (size_t i = 0; i < _n; ++i) {
    //double g_Z2_mean = ; // Use row with the single index
     lbda[i] = arma::dot(g_Z1.row(i) - arma::mean(g_Z2.rows(*_idx_md[i]), 0), b2);
  }
  
  y_adj = _y1 - lbda;
}

void MSREG::power_series(int order, const arma::mat& Z_orig, arma::mat& Zp) {
  if (order == 1) {
    Zp = Z_orig;
  } else {
    arma::mat tmp;
    power_series(order - 1, Z_orig, Zp);
    
    for (size_t j = 0; j < Z_orig.n_cols; ++j) {
      tmp = arma::join_horiz(tmp, Z_orig.col(j) % Zp);
    }
    
    Zp = arma::join_horiz(Zp, tmp);
  }
}

void MSREG::rmcoll(arma::mat& Z, arma::uvec& p) {
  arma::mat G = arma::inv_sympd(Z.t() * Z);
  p = arma::find(arma::diagvec(G) != 0); // Select columns with non-zero diagonal
  Z = Z.cols(p);
}

void MSREG::msols(const arma::vec& y1, const arma::mat& W, arma::vec& b, arma::mat& V) {
  arma::mat invWpW = arma::inv_sympd(W.t() * W);
  b = invWpW * (W.t() * y1);
  //b.print("beta_ols");
  arma::vec eps = y1 - W * b;
  arma::mat uhat = W.each_col() % eps;
  V = invWpW * (uhat.t() * uhat) * invWpW;
  //V.print("VCV_ols");
}

void MSREG::_msreg_msii(
    const std::string& vce, const arma::vec& y1, const arma::mat& W,
    const arma::mat& X1, const arma::mat& X2, const arma::mat& Z1, const arma::mat& Z2,
    double n, double m, double cons, double nn, arma::vec& b, arma::mat& V) {
  
  arma::mat Sigma;
  arma::mat invPw;
  
  // reorder X2 w.r.t Z2
  _msreg_reorder_S2(Z2, X2, m);
  
  // Get Sigma
  _msreg_get_Sigma(X1, X2, Z1, cons, m, Sigma);
  
  // get point estimate
  _msreg_msii_b(y1, W, n, nn, Sigma, b, invPw);
  
  // get V estimate
  _msreg_msii_V(vce, y1, W, X1, X2, Z1, Z2, Sigma, b, n, m, cons, nn, invPw, V);
}

void MSREG::_msreg_reorder_S2(const arma::mat& Z2, const arma::mat& X2, size_t _m) {
  arma::uvec p(_m, arma::fill::zeros);   // Permutation vector
  arma::vec res;                        // Vector for storing distances
  arma::vec sidx = arma::randu<arma::vec>(_m); // Random indices for tie-breaking
  arma::uvec used(_m, arma::fill::zeros);     // Logical vector to track used indices
  arma::uvec idx;       // Stores sorted indices
  
  // Step 1: Find the row with the smallest value in the first column of _Z2
  //double zmin = _Z2.col(0).min();
  p[0] = arma::index_min(Z2.col(0)); // Row index with the smallest value
  used[p[0]] = 1; // Mark this index as used
  
  // Step 2: Reorder rows based on trace norms
  for (size_t j = 1; j < _m; ++j) {
    arma::rowvec z_jm1 = Z2.row(p[j - 1]); // Previous row
    _msreg_tr_norm(Z2, z_jm1, res);        // Compute distances to all rows
    
    // Perform lexicographical sorting of (res, sidx)
    arma::mat combined(_m, 2);
    combined.col(0) = res;
    combined.col(1) = sidx;
    
    
    // Create a vector of row indices
    arma::uvec row_indices = arma::regspace<arma::uvec>(1, _m); // 1 to _m
    
    // Sort row indices based on lexicographical order of rows in `combined`
    std::sort(row_indices.begin(), row_indices.end(), [&](size_t i, size_t j) {
      --i; --j; // Convert 1-based indices to 0-based for Armadillo
      if (combined(i, 0) < combined(j, 0)) return true;   // Compare primary key
      if (combined(i, 0) == combined(j, 0)) return combined(i, 1) < combined(j, 1); // Secondary key
      return false;
    });
    
    // Use `row_indices` as the permutation vector
    arma::uvec pres = row_indices - 1; // Convert back to 0-based for Armadillo usage
    
    // Find the first unused row in `pres`
    for (size_t k = 0; k < pres.n_elem; ++k) {
      if (!used[pres[k]]) { // Check if index `pres[k]` has been used
        p[j] = pres[k];
        used[p[j]] = 1; // Mark as used
        break;
      }
    }
  }
  
  // Step 3: Reorder _Z2 and _X2 based on `p`
  _Z2 = _Z2.rows(p);
  _X2 = _X2.rows(p);
}


void MSREG::_msreg_tr_norm(const arma::mat& A, const arma::rowvec& z, arma::vec& res) {
  arma::mat diff = A.each_row() - z;        // Row-wise differences
  res = arma::sqrt(arma::sum(arma::square(diff), 1)); // Compute Euclidean distance
}

void MSREG::_msreg_get_Sigma(
    const arma::mat& X1, const arma::mat& X2, const arma::mat& Z1,
    double cons, double m, arma::mat& Sigma) {
  
  arma::mat S2 = arma::zeros<arma::mat>(X2.n_cols, X2.n_cols);
  
  for (size_t j = 1; j < m; ++j) {
    arma::rowvec D_X2 = X2.row(j) - X2.row(j - 1);
    S2 += D_X2.t() * D_X2;
  }
  S2 /= (2 * (m - 1));
  
  size_t d1 = X1.n_cols;
  size_t d2 = X2.n_cols;
  size_t dz = Z1.n_cols + cons;
  
  Sigma = arma::zeros<arma::mat>(d1, d1);
  Sigma = arma::join_horiz(Sigma, arma::zeros<arma::mat>(d1, d2));
  Sigma = arma::join_horiz(Sigma, arma::zeros<arma::mat>(d1, dz));
  
  Sigma = arma::join_vert(Sigma, arma::join_horiz(arma::zeros<arma::mat>(d2, d1), S2));
  Sigma = arma::join_vert(Sigma, arma::join_horiz(arma::zeros<arma::mat>(dz, d1 + d2), arma::zeros<arma::mat>(dz, dz)));
  Sigma.print("Sigma");
  
  }


void MSREG::_msreg_msii_b(
    const arma::vec& y1, const arma::mat& W, double n, double nn,
    const arma::mat& Sigma, arma::vec& b, arma::mat& invPw) {
  
  arma::mat Rw = (W.t() * y1) / n;
  arma::mat Qw = (W.t() * W) / n;
  arma::mat Pw = Qw - Sigma / nn;
  
  invPw = arma::inv(Pw);
  b = invPw * Rw;
}

void MSREG::_msreg_msii_V(
    const std::string& vce, const arma::vec& y1, const arma::mat& W,
    const arma::mat& X1, const arma::mat& X2, const arma::mat& Z1, const arma::mat& Z2,
    const arma::mat& Sigma, const arma::vec& b, double n, double m, double cons, double nn,
    const arma::mat& invPw, arma::mat& V) {
  
  arma::vec ehat = y1 - W * b;
  arma::mat uhat = W.each_col() % ehat;
  arma::mat O_11A = (uhat.t() * uhat) / n;
  
  size_t d1 = X1.n_cols;
  size_t d2 = X2.n_cols;
  size_t dz = Z1.n_cols + cons;
  arma::mat Sigma2 = Sigma.submat(d1, d1, d1 + d2 - 1, d1 + d2 - 1);
  arma::vec b2 = b.subvec(d1, d1 + d2 - 1);
  
  arma::mat G1 = _msreg_msii_Gamma(1, m, X2, Sigma2, b2);
  arma::mat G0 = _msreg_msii_Gamma(0, m, X2, Sigma2, b2);
  
  arma::mat O_22 = arma::join_horiz(arma::zeros<arma::mat>(d1, d1), G0 + 2 * G1);
  O_22 = arma::join_horiz(O_22, arma::zeros<arma::mat>(d1 + d2, dz));
  O_22 = O_22 / std::pow(nn, 2);
  O_22 = 0.5 * (O_22 + O_22.t());
  
  arma::mat Om = O_11A;
  
  if (vce == "vi") {
    V = invPw * Om * invPw / n;
  } else if (vce == "vii") {
    V = invPw * O_11A * invPw / n;
  } else if (vce == "viii") {
    V = invPw * O_22 * invPw / m;
  } else {
    throw std::runtime_error("Misspecified vce()");
  }
}

arma::mat MSREG::_msreg_msii_Gamma(int l, int m, const arma::mat& X2, const arma::mat& Sigma2, const arma::vec& b2) {
  arma::mat G = arma::zeros<arma::mat>(X2.n_cols, X2.n_cols);
  for (int j = 1; j < m; ++j) {
    arma::rowvec D_X2_j = X2.row(j) - X2.row(j - 1);
    arma::rowvec D_X2_jl = X2.row(j - l) - X2.row(j - l - 1);
    
    arma::mat Gj = (D_X2_j.t() * D_X2_j) / 2 - Sigma2;
    arma::mat Gjl = (D_X2_jl.t() * D_X2_jl) / 2 - Sigma2;
    
    G += Gj * Gjl.t();
  }
  return G / (m - 1);
}




