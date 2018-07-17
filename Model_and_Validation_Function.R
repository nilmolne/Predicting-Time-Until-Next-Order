
time_to_next_transaction <- function (params, x, t.x, T.cal) {
  
  # When will the next purchase be?
  time.to.next.transaction <- 1
  ntransactions <- 0
  
  while (1) {
    
    ntransactions <- 
    	pnbd.ConditionalExpectedTransactions(params, T.star = time.to.next.transaction, x, t.x, T.cal)
    
    if (ntransactions >= 1) {
    	
      break
      
    }
    
    time.to.next.transaction <- time.to.next.transaction + 1;
  }
  
  return(time.to.next.transaction)
  
}

############
############
############

validate_pnbd_model <- function (data, params) {
  
  # Vectors to store data as the code loops through all customers
  df.cust.id <- c()
  df.prediction.error <- c()
  df.prob.alive <- c()
  df.n.cal.transactions <- c()
  
  # Loop for all customers 
  for (i in 1:nrow(UK.data$holdout$cbt)) {
    
    cust.id <- row.names(UK.data$holdout$cbt)[i]
    df.cust.id <- c(df.cust.id, row.names(UK.data$holdout$cbt)[i])
    
    # Collect customer 'i' past behaviour
    x <- UK.data$cal$cbs[cust.id, 'x']
    t.x <- UK.data$cal$cbs[cust.id, 't.x']
    T.cal <- UK.data$cal$cbs[cust.id, 'T.cal']
    
    # Customers number of transactions during calibration 
    # +1 because 'x' is the number of repeat transactions in the calibration period
    df.n.cal.transactions <- c(df.n.cal.transactions, UK.data$cal$cbs[cust.id, 'x'] + 1)
    
    # Customers probability to be alive 
    df.prob.alive <- c(df.prob.alive, pnbd.PAlive(pnbd.params, x, t.x, T.cal))
    
    n.holdout.transactions <- UK.data$holdout$cbs[cust.id, 'x.star']
    
    # Check if customer did at least one transaction during holdout
    if (n.holdout.transactions == 0) {
      # Assign NA to error if no transactions were made
      df.prediction.error <- c(df.prediction.error, NA)
      
    } else {
      
      ##################################################
      # Model Output: Predict time to next transaction #
      ##################################################
      
      days.to.next.predicted.transaction <- 
        time_to_next_transaction(pnbd.params, x, t.x, T.cal)
      
      ##################################################
      #              End of Model Output               # 
      ##################################################
      
      # Look for when the transaction actually happened during holdout
      for (j in 1:ncol(UK.data$holdout$cbt)) {
        
        if (UK.data$holdout$cbt[cust.id, j] == 1) {
          
          days.to.next.transaction <- 
            as.Date(colnames(UK.data$holdout$cbt)[j]) - cal.end.date
          
          # Compute and store Error
          error <- abs(days.to.next.transaction - days.to.next.predicted.transaction)
          df.prediction.error <- c(df.prediction.error, error)
          
          break
          
        }
      }
    }
  }
  
  # Put results for every customer into a dataframe
  next.purchase.results <- 
    data.frame(df.cust.id,
               df.n.cal.transactions,
               df.prob.alive,
               df.prediction.error)
  
  return(next.purchase.results)
  
}