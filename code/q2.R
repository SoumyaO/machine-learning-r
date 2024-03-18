# =======================
# Q 2.1 Data generation
# =======================
set.seed(365)
# function to generate 2D points in a circle/ring.
# - Inputs are the 
# - centre of the circle/ring (x_offset, y_offset)
# - min and max radii of the ring (min_r, max_r)
# - class label
# - number of points
generate_data = function (x_offset, y_offset, min_r, max_r, class, num_points) {
  # generate r and theta
  r = runif(num_points, min_r, max_r)
  theta = runif(num_points, 0, 2 * pi)
  
  # x = r*cos(theta) + x_offset
  feat1 = r * cos(theta) + x_offset
  # y = r*sin(theta) + y_offset
  feat2 = r * sin(theta) + y_offset
  
  # create a dataframe with the two features
  data_frame = data.frame(feat1, feat2)
  # add the class column
  data_frame$class = class
  # return the dataframe
  return(data_frame)
}

# generate three blobs of data
data1 = generate_data(0, 0, 0, 0.5, 1, 50)
data2 = generate_data(0, 0, 0.5, 1, 2, 50)
data3 = generate_data(0, 0, 1, 1.25, 3, 50)

# merge all of them into 1 dataframe
df1 = merge(data1, data2, all=TRUE)
df = merge(df1, data3, all=TRUE)

# plot the data
plot(df[,-3], col=as.integer(df[,3]))

# =========================
# Q 2.2 SVM classification
# =========================
library(caret)
library(e1071)

# split the dataset into training (50%) and test (50%)
training_sample_rows = createDataPartition(df$class, p = 0.5, list = FALSE, times = 1)
train=df[training_sample_rows,] # training set - pick the rows computed above
test=df[-training_sample_rows,] # test set - pick the rows not chosen above

# SVM with linear kernel
# ------------------------
# tuning parameters
grid_linear=expand.grid(C = c(0.01, 0.1, 1, 10))
fitControl=trainControl(method = "cv", number = 5)
# training
svm.linear=train(train[,-3], as.factor(train[,3]), method = "svmLinear",
                 trControl=fitControl,
                 preProcess = c("center", "scale"),
                 tuneGrid = grid_linear)
svm.linear
plot(svm.linear)
# Prediction on test
pred.linear=predict(svm.linear,test[,-3])
# Accuracy on the test set
mean(pred.linear == test[,3])

# plot the support vectors on the training dataset
plot(cmdscale(dist(train[,-3])),
     col = as.integer(train[,3]),
     pch = c("o","+")[1:150 %in% svm.linear$finalModel@SVindex + 1],
     main = "Linear kernel", 
     xlab = "feature1", 
     ylab="feature2")

# SVM with polynomial kernel
# ----------------------------
# tuning parameters
grid_poly=expand.grid(degree = c(2,3,4),
                      scale = c(0.01, 0.1, 1),
                      C = c(0.01, 0.1, 1, 10))
fitControl=trainControl(method = "cv", number = 5)
# training
svm.poly=train(train[,-3], as.factor(train[,3]), method = "svmPoly",
               trControl=fitControl,
               preProcess = c("center", "scale"),
               tuneGrid = grid_poly)
svm.poly
plot(svm.poly)
# Prediction on test set
pred.poly=predict(svm.poly,test[,-3])
# Accuracy on the test set
mean(pred.poly == test[,3])

# plot the support vectors on the training dataset
plot(cmdscale(dist(train[,-3])),
     col = as.integer(train[,3]),
     pch = c("o","+")[1:150 %in% svm.poly$finalModel@SVindex + 1],
     main = "Polynomial kernel", 
     xlab = "feature1", 
     ylab="feature2")

# SVM with RBF kernel
# --------------------
# tuning parameters
grid_radial=expand.grid(sigma = c(0.01, 0.1, 1,10),
                        C = c(0.01, 0.1, 1, 10))
fitControl=trainControl(method = "cv", number = 5)
# training
svm.radial=train(train[,-3], as.factor(train[,3]), method = "svmRadial",
               trControl=fitControl,
               preProcess = c("center", "scale"),
               tuneGrid = grid_radial)
svm.radial
plot(svm.radial)
# Prediction on test set
pred.radial=predict(svm.radial,test[,-3])
# Accuracy on the test set
mean(pred.radial == test[,3])

# plot the support vectors on the training dataset
plot(cmdscale(dist(train[,-3])),
     col = as.integer(train[,3]),
     pch = c("o","+")[1:150 %in% svm.radial$finalModel@SVindex + 1],
     main = "RBF kernel", 
     xlab = "feature1",
     ylab="feature2")


