# x is not a required parameter because these methods support
# both autoregressive models and models with external regressors.
# Any data for x is expected to be passed in the dynamic arguments.
fit <- function(y, FUN, ...) { 
    arguments <- list(...)
    names(arguments) <- gsub("_", ".", names(arguments))
    arguments <- c(list(y = y), arguments)
	
    model.fit <- tryCatch({
		do.call(FUN, arguments, quote = TRUE)
	}, error = function(err) {
    # TODO: Stop Error Message from writing to StdErr
		#opt <- options(show.error.messages = FALSE)
		#on.exit(options(opt))
		stop(err)
	})
    # print( summary(model.fit) )
	return(model.fit)
}


predict <- function(model, FUN, ...) {
    arguments <- list(...)
    names(arguments) <- gsub("_", ".", names(arguments))
    arguments <- c(list(object = model), arguments)

    #pred <- try(do.call(FUN, arguments, quote = TRUE))
    #if (!inherits(pred, "try-error")) {
    #    return(pred$mean)
    #} else {
    #    return(NULL)
    #}

	pred <- tryCatch ({
		do.call(FUN, arguments, quote = TRUE)
	}, error = function(err) {
		print("Encountered Error in model prediction")
		#opt <- options(show.error.messages = FALSE)
		#on.exit(options(opt))
		stop()
	})
    
	return(list(pred$fitted, pred$mean, c(pred$upper)))
}

### special fit() and predict() for SARIMAX

fit_autoarima <- function(y, FUN, xreg, ...) { 
    arguments <- list(...)
    names(arguments) <- gsub("_", ".", names(arguments))
    arguments <- c(list(y = y, xreg = as.matrix(xreg)), arguments)
	
    model.fit <- tryCatch({
		do.call(FUN, arguments, quote = TRUE)
	}, error = function(err) {
		stop(err)
	})

	return(model.fit)
}

predict_autoarima <- function(model, FUN, xreg, ...) { 
    arguments <- list(...)
    names(arguments) <- gsub("_", ".", names(arguments))
    arguments <- c(list(object = model, xreg = as.matrix(xreg)), arguments)

	pred <- tryCatch ({
		do.call(FUN, arguments, quote = TRUE)
	}, error = function(err) {
		print("Encountered Error in model prediction")
		stop()
	})
    
	return(list(pred$fitted, pred$mean, c(pred$upper)))
}

#DLM
fit_dlm <- function(y, EST, FUN) {
  
  # Model building
  if (length(y) < frequency(y) *52 ){ # less than 1yr, no yearly seasonality
    if ( frequency(y) ==1 ){ # no weekly seasonality
      model.build <- function( p ){
        return( dlmModPoly( order = 2, dV = p[1], dW = p[2:3] ) ) }
    } else { # has weekly seasonality for daily data
      model.build <- function( p ){
        return( dlmModPoly( order = 2, dV = p[1], dW = p[2:3] )
                + dlmModSeas( frequency = frequency(y)) ) } }
  } else { # For MUID has more than 1yr of data, has yearly seasonality
    if (frequency(y) ==1){ # no weekly seasonality
      model.build <- function( p ) {
        return( dlmModPoly( order = 2, dV = p[1], dW = p[2:3] )
                + dlmModTrig( tau = frequency(y)*52, q = 3) ) }
    } else { # has both weekly and yearly seasonality
      model.build <- function( p ) {
        return(dlmModPoly( order = 2, dV = p[1], dW = p[2:3] )
               + dlmModSeas( frequency = frequency(y) )
               + dlmModTrig( tau = frequency(y)*52, q = 3) ) } }
  }
  
  arguments <- c(list(y, parm=c(1,1,1), build=model.build))
  
  model.mle <- tryCatch({
    do.call(EST, arguments, quote = TRUE) }, error = function(err) { stop(err) })
  
  arguments_2 <- c(list(y, model.build(model.mle$par) ) )
  
  model.fit <- tryCatch({
    do.call(FUN, arguments_2, quote = TRUE) }, error = function(err) { stop(err) })
  
  return(model.fit)
}

### Special predict() for DLM as it uses pred$f to access the forecasts
predict_dlm <- function(model, FUN, nAhead) {
    arguments <- c(list(model,nAhead=nAhead))

	pred <- tryCatch ({
		do.call(FUN, arguments, quote = TRUE)
	}, error = function(err) {
		print("Encountered Error in model prediction")
		stop()
	})
    
	return(c(pred$f))
}

