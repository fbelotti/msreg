\name{msreg-package}
\alias{msreg-package}
\alias{msreg}
\docType{package}
\title{
  Consistent estimation of linear regression models using matched data
}
\description{
    msreg is the R implementation of the Stata msreg command.
    It fits a linear regression model when some of the covariates are missing in one sample
    and one needs to use a matched sample for estimation.  The ordinary least-squares (OLS)
    estimator using the matched sample is inconsistent in this setting.  msreg can
    consistently estimate the model coefficients using matched sample indirect inference
    estimators developed in Hirukawa and Prokhorov (2018).
}
\usage{
  msreg(formula, cohort_data, survey_data, est, vcov, nneighbor = 3, order = 2)
}
\arguments{
  \item{formula}{an object of class \code{"formula"} (or one that can be coerced to that class): 
    a symbolic description of the model to be fitted. The details of model specification 
    are given under \emph{Details}.}
  \item{cohort_data}{a data frame (or object coercible by 
    \code{as.data.frame} to a data frame) containing the variables from the cohort data to be included in the model. Generally, the dependent variable, some regressors specific to the cohort, and some variables in common with the survey dataset to be used as matching variable.}
  \item{survey_data}{a data frame (or object coercible by 
    \code{as.data.frame} to a data frame) containing the variables from the survey data to be included in the model. These are the covariate of interest and some variables in common with the cohort dataset to be used as matching variable.}
\item{est}{specify the type of estimator. It can be "ols", "onestep" or "twostep". Default is "ols".}
\item{nneighbor}{specify the number of matches per observation. Default is 1.}
\item{metric}{specify the distance matrix used as the weight matrix in a quadratic form that transforms the multiple distances into a single distance measure. It can be "mahalanobis" or "euclidean". Default is "mahalanobis".}
\item{order}{specify the order of polynomial in the power-series approximation.}
\item{list}{display a list with all function's arguments passed to the class.}
}
\details{
Models for \code{msreg} are specified symbolically. A typical model has the form 
\code{dependent ~ cohort_indep | matching_vars | survey_main_variable} where \code{dependent} is the (numeric) response vector,  
\code{cohort_indep} is a series of terms which specifies a linear predictor for \code{dependent}, \code{matching_vars} is a series of variables (maximum 5 variables) which are used to match \code{survey_main_variable} into the cohort data frame.

A \code{cohort_indep} specification of the form \code{first + second} indicates all the terms in 
\code{first} together with all the terms in \code{second} with duplicates removed. Same for \code{matching_vars}.

}
\value{
As for the variance-covariance matrix type:
\item{vi}{VI-type variance-covariance assuming sample-size ratio converges to a nonzero constant; the default.}
\item{vii}{VII-type variance-covariance assuming sample-size ratio converges to zero.}
\item{viii}{VIII-type variance-covariance assuming sample-size ratio converges to infinity.}
}
\author{
Federico Belotti.

Maintainer: Federico Belotti <ambda76@econometrics.it>
}
\references{
  Hirukawa, M., and A. Prokhorov (2018). Consistent estimation of linear regression models
  using matched data. \emph{Journal of Econometrics}, \strong{203}, 344–358.
  \url{https://doi.org/10.1016/j.jeconom.2017.07.006}
}
\keyword{ package }
\seealso{
  \code{lm()}
}
\examples{
  \dontrun{
  fit <- msreg(model, cohort_data, svy_data, est = "ols", vcov = "vi", nneighbor = 3, order = 2 )
  summary(fit)
  }   
}
