

print.msreg <- function(x, ...) {
  cat("Call:\n<MSREG Model>\n\n")
  cat("Coefficients:\n")
  print(x$coefficients)
  invisible(x)
}

summary.msreg <- function(object, ...) {
  coefs <- object$coefficients
  V <- object$VCV
  se <- sqrt(diag(V))
  tval <- coefs / se
  # TODO: pass the right DF
  df <- Inf  # or use actual df if known
  pval <- 2 * pt(-abs(tval), df)
  
  result <- cbind(
    Estimate = coefs,
    `Std. Error` = se,
    `t value` = tval,
    `Pr(>|t|)` = pval
  )
  
  cat("Summary of MSREG Model:\n\n")
  printCoefmat(result, P.values = TRUE, has.Pvalue = TRUE)
  invisible(result)
}



msreg <- function(formula, df_cohort, df_survey,
                  estimator = c("ols", "onestep", "twostep"),
                  vcov = c("vi", "vii", "viii"),
                  metric = c("mahalanobis", "euclidean"),
                  nneighbor = 1,
                  order = 1, 
                  list = FALSE) {
  
  # parse data
  # Ensure the cohort and survey data are a data frame
  if (!is.data.frame(df_cohort)) stop("Cohort data must be a data.frame!")
  if (!is.data.frame(df_survey)) stop("Survey data must be a data.frame!")

  
  # parse formula
  # Ensure the formula is of class 'formula'
  if (!inherits(formula, "formula")) stop("Input must be a formula.")
  
  # Parse the formula using the Formula package
  parsed_formula <- Formula(formula)
  
  # Extract each component
  yvar <- all.vars(formula(parsed_formula, lhs = 1))[1]  # Response variable
  xvars <- setdiff(all.vars(formula(parsed_formula, rhs = 1)), yvar) # First part of RHS
  zvars <- setdiff(all.vars(formula(parsed_formula, rhs = 2)), yvar)  # Second part of RHS
  missing_var <- setdiff(all.vars(formula(parsed_formula, rhs = 3)), yvar)  # Third part of RHS
  
  # Combine variables for validation
  main_model <- c(yvar, xvars, zvars)
  matching_model <- c(missing_var, zvars)
  
  # Check if variables exist in the data frame
  check_vars_main <- setdiff(main_model, names(df_cohort))
  if (length(check_vars_main) > 0) {
    stop(sprintf("The following variables are missing in the data frame: %s", paste(check_vars_main, collapse = ", ")))
  }
  check_vars_matching <- setdiff(matching_model, names(df_survey))
  if (length(check_vars_matching) > 0) {
    stop(sprintf("The following variables are missing in the data frame: %s", paste(check_vars_matching, collapse = ", ")))
  }
  
  # Check if the intercept (constant) is included
  check_constant <- terms(formula(parsed_formula, rhs = 1))  # Check the first RHS for the intercept
  has_constant <- attr(check_constant, "intercept") == 1     # 1 if intercept is included, 0 otherwise
  
  # parse estimator
  estimator <- match.arg(estimator)
  # parse vcov
  vcov <- match.arg(vcov)
  # parse vcov
  metric <- match.arg(metric)
  
  # Validate  nneighbor
  if (!is.null(nneighbor) && (!is.numeric(nneighbor) || length(nneighbor) != 1)) {
    stop("The nneighbor argument must be a single numeric value.")
  }
  # Validate  order
  if (!is.null(order) && (!is.numeric(order) || length(order) != 1)) {
    stop("The order argument must be a single numeric value.")
  }
  
  
  # Select only vars of interests in the two data.frames 
  # Remove rows with missing values
  e_df_cohort <- df_cohort[complete.cases(df_cohort[, c(yvar, xvars, zvars)]), ]
  e_df_survey <- df_survey[complete.cases(df_survey[, c(missing_var, zvars)]), ]
  e_df_cohort <- e_df_cohort[, c(yvar, xvars, zvars)]
  e_df_survey <- e_df_survey[, c(missing_var, zvars)]
  
  # Just check the parsed options in R
  # print(yvar)
  # print(xvars)
  # print(has_constant)
  # print(zvars)
  # print(missing_var)
  # print(estimator)
  # print(vcov)
  # print(metric)
  # print(nneighbor)
  # print(order)
  
  # Create an MSREG object with custom parameters
  msreg_obj <- new(MSREG, has_constant, estimator, vcov, metric, nneighbor, order)
  
  # Check that options are correctly passed to the class
  if (!missing(list) && isTRUE(list)) { 
    msreg_obj$list_class()
  }
  
  # load_data
  msreg_obj$load_data(df1 = e_df_cohort, df2 = e_df_survey, st_y = yvar,
                           st_x1 = c(xvars), st_x2 = missing_var,
                           st_z = c(zvars))
  
  # Run estimation
  set.seed(1234467879)
  msreg_obj$compute() 
  msreg_obj$post_results() 
  
}

