
.onLoad <- function(libname, pkgname) {
  loadModule("msreg", TRUE)
}

.onAttach <- function(libname, pkgname) {
  packageStartupMessage("Loading msreg dependencies")
  library(Formula)
  packageStartupMessage("Formula ...loaded")
}
