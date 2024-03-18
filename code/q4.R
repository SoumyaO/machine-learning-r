myFDA <- function(X,y){
  ########################################################
  # This function calculates the linear discriminant for binary
  # classification.
  # Input: Feature matrix, X (N by p) and label vector, y (N by 1)
  # Output: Linear discriminant, w (p by 1)
  ########################################################
  
  # separate the data into classes
  x1 = X[y == 1,]
  x2 = X[y == 2,]
  
  # Initialize a vector to store the means
  mu1 <- numeric(ncol(x1))
  mu2 <- numeric(ncol(x2))
  
  # Loop over each column and calculate the mean of the two classes
  for (i in 1:ncol(x1)) {
    sumcol = 0
    for (j in 1:nrow(x1)) {
      sumcol = sumcol + x1[j,i]
    }
    mu1[i] = sumcol/nrow(x1)
    
  }
  
  for (i in 1:ncol(x2)) {
    sumcol = 0
    for (j in 1:nrow(x2)) {
      sumcol = sumcol + x2[j,i]
    }
    mu2[i] = sumcol/nrow(x2)
  }
  
  # Print the mean of each class
  print(mu1)
  print(mu2)
  
  # compute within class covariance matrix for each class
  # Initialize a matrix to store the covariance values
  S1 <- matrix(0, ncol(x1), ncol(x1))
  S2 <- matrix(0, ncol(x1), ncol(x1))
  
 # Loop over each column and calculate the covariance with every other column
  for (i in 1:ncol(x1)) {
    for (j in 1:ncol(x1)) {
      sumcov = 0
      for (k in 1:nrow(x1)) {
        sumcov = sumcov + (x1[k,i] - mean(x1[,i])) * (x1[k,j] - mean(x1[,j]))
      }
      S1[i,j] = sumcov / (nrow(x1) - 1)
    }
  }
  for (i in 1:ncol(x2)) {
    for (j in 1:ncol(x2)) {
      sumcov = 0
      for (k in 1:nrow(x2)) {
        sumcov = sumcov + (x2[k,i] - mean(x2[,i])) * (x2[k,j] - mean(x2[,j]))
      }
      S2[i,j] = sumcov / (nrow(x2) - 1)
    }
  }
  
  # Print the covariance matrix
  print(S1)
  print(S2)
   
  # Calculate SW as S1+S2
  SW = S1 + S2
  
  # compute w
  w = solve(SW) %*% (mu1 - mu2) # px1
  
  # Set threshold for prediction
  threshold = t(w) %*% (mu1 + mu2) / 2
  
  result = list(w=w, th=threshold, mu1=mu1, mu2=mu2)
  return(result)
  
}

# ==============================
# Applying on GermanCredit data
# ==============================
library(ISLR)
library(caret)
data(GermanCredit)
GermanCredit[, c("Purpose.Vacation","Personal.Female.Single")] <- list(NULL)

# apply PCA on the data
data = prcomp(GermanCredit[,-10], scale. = TRUE)
# compute the proportion of variance explained by each principal component
data.var = data$sdev ^2
data.pve = data.var/sum(data.var)

plot(cumsum(data.pve), xlab = "Principal Component", ylab ="
Cumulative Proportion of Variance Explained", ylim = c(0,1),
     type = "b")

data_pc = data$x[,cumsum(data.pve) < 0.9]
data_pc = cbind(data_pc, GermanCredit$Class)

# split the dataset into training (70%) and test (30%) splits
set.seed(365)
training_sample_rows = createDataPartition(data_pc[,38], p = 0.7, list = FALSE, times = 1)
train=data_pc[training_sample_rows,] # training set - pick the rows computed above
test=data_pc[-training_sample_rows,] # test set - pick the rows not chosen above

# compute W using LDA
result = myFDA(train[,-38], as.numeric(train[,38]))

# Print w
print(result$w)
print(max(result$w))
print(min(result$w))

# Predict using W
pred = test[,-38] %*% result$w


gt = as.numeric(test[,38]) - 1

# compute threshold
threshold = result$th[1,1]

# Prediction on the mean of the classes to see where each class falls 
# based on the threshold

pred_mu1 = result$mu1  %*% result$w
pred_mu2 = result$mu2  %*% result$w

print(pred_mu1)
print(pred_mu2)

# Accuracy of the prediction
mean((pred<threshold) == gt)

# plot the top two principal components
plot(test[,1:2], col=as.numeric(test[,38]))

# plot the projected test data and the separation lines
plot(pred, col=as.numeric(test[,38]), xlab="sample index", ylab="projected value", main="LDA projections")
x = c(0,nrow(test))
y = c(threshold, threshold)
lines(x, y, type = "l", col="blue")
lines(x, c(mean(pred[test[,38]==2,]), mean(pred[test[,38]==2,])), type = "l", col="red")
lines(x, c(mean(pred[test[,38]==1,]), mean(pred[test[,38]==1,])), type = "l", col="black")



