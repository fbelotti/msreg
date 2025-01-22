
# Restart R
#.rs.restartR()

# Clear environment
rm(list = ls())

# Set working directory to your package folder
wd = "/Users/federico/Dropbox/Projects/LONGITOOLS/microsim/data"
setwd(wd)



#library(Rcpp)
library(dplyr)
library(foreign)
library(msreg)

#sourceCpp("/Users/federico/Dropbox/Projects/LONGITOOLS/microsim/data/msreg/src/msreg.cpp")
#source("/Users/federico/Dropbox/Projects/LONGITOOLS/microsim/data/msreg/R/msreg.R")

cohort_data <- read.dta(file = paste0(wd, "/dta/nfbc_annual_msreg_ge_bw.dta"), convert.factors = FALSE)
eusilc_6_data <- read.dta(file = paste0(wd, "/dta/msreg_eusilc_6.dta"), convert.factors = FALSE)
eusilc_6_data <- eusilc_6_data[c("job_dad_2", "job_mom_2", "edu_mom", "ln_equi_hh_income")]

# Define the cohort and survey data.frame with the appropriate variables
# Actually, thanks to the command parsing, it is not needed to select the right variables
# They will be selected by the specified formula
#gesta_cohort <- select(cohort_data, gesta, female, age_mom, job_dad_2, job_mom_2, edu_mom)
#gesta_eusilc_6 <- select(eusilc_6_data, ln_equi_hh_income, job_dad_2, job_mom_2, edu_mom)

# Define the model
model <- gesta ~ age_mom + female | job_dad_2 + job_mom_2 + edu_mom | ln_equi_hh_income

msreg(model, cohort_data, eusilc_6_data, est = "ols", vcov = "vi", nneighbor = 3, order = 2 )
 
 


