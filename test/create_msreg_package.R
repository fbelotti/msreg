
# Clear environment
rm(list = ls())

library(Rcpp)
library(RcppArmadillo)

# Set working directory to your package folder
wd = "/Users/federico/Dropbox/Projects/LONGITOOLS/microsim/data"
setwd(wd)

# Use this only for creating the initial package structure 
#Rcpp.package.skeleton(name = "msreg", module = TRUE)

# Define the package directory
pkg_dir <- "/Users/federico/Dropbox/Projects/LONGITOOLS/microsim/data/msreg"
# Define the package tarball (version 1.0.1)
pkg_tarball <- "msreg_1.0.1.tar.gz"

# Unisntall the previous version 
remove.packages("msreg")

# Run R CMD build
build_command <- sprintf("R CMD build %s", pkg_dir)
system(build_command) 

# Install the package
install_command <- sprintf("R CMD INSTALL %s", pkg_tarball)
system(install_command)



