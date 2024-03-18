#install.packages("ISLR")
library(ISLR)
# load the dataset
data(GermanCredit)

# install.packages("caret")
library(caret)

# check missing values
sum(is.na(GermanCredit))
# there are no missing values
str(GermanCredit[, 1:62])

# Check the number of unique values in each column
rapply(GermanCredit, function(x)length(unique(x)))

# remove columns with no distinct values
# these are the columns Purpose.Vacation and Personal.Female.Single
GermanCredit[, c("Purpose.Vacation","Personal.Female.Single")] <- list(NULL)

# split the dataset into training (70%) and test (30%) splits
set.seed(365)
training_sample_rows = createDataPartition(GermanCredit$Class, p = 0.7, list = FALSE, times = 1)
train=GermanCredit[training_sample_rows,] # training set - pick the rows computed above
test=GermanCredit[-training_sample_rows,] # test set - pick the rows not chosen above

# ==================================================
# Q 1.1: Fitting decision tree using caret library
# ==================================================
# 5-fold cross validation
fitcontrol=trainControl(method = "cv",
                        number = 5)
set.seed(365)
# checking 20 different alpha values to find the optimal alpha
GermanCredit.rpart=train(train[, -10],     # X
                  train[, 10],             # y
                  method = "rpart",        # method rpart
                  tuneLength=20,           # 20 values of alpha  
                  trControl = fitcontrol)

GermanCredit.rpart
plot(GermanCredit.rpart$finalModel)
text(GermanCredit.rpart$finalModel, pretty=1, cex=.8)
# optimal alpha = cp = 0.01190476
# optimal number of leaves based on above alpha =  10

# --> Predict accuracy on the test dataset
pred_test_dt = predict(GermanCredit.rpart$finalModel, test[,-10], type="class")

table(pred_test_dt, test[,10])
# accuracy
mean(pred_test_dt == test[,10])
# test error rate
mean(pred_test_dt != test[,10])

# Visualise tree using rattle library
library(rattle)
fancyRpartPlot(GermanCredit.rpart$finalModel)


# =====================
# Q 1.2: Random Forest
# =====================

# set number of trees
num_trees = 1000
# 5-fold cross validation
fitcontrol=trainControl(method = "cv",
                        number = 5)
set.seed(365)
# fit random forest, tune for the number of features using cross-validation
GermanCredit.rf=train(train[, -10],     # X
                      train[, 10],      # y
                      method = "rf",    # random forest classifier
                      ntree = num_trees,# number of trees in the forest
                      tuneLength=5,  
                      trControl = fitcontrol)
# Output
GermanCredit.rf
plot(GermanCredit.rf)
GermanCredit.rf$finalModel

# test error rate
pred_test_rf = predict(GermanCredit.rf$finalModel, test[,-10], type="class")

table(pred_test_rf, test[,10])
# accuracy
mean(pred_test_rf == test[,10])
# test error rate
mean(pred_test_rf != test[,10])

# Variable importance plot
plot(varImp(GermanCredit.rf,scale=FALSE), top =20)


# ==================
# Q 1.3: ROC curves
# ==================
# install.packages("pROC")
library(pROC)

# predict probabilities of the response being 1
pred_test_dt_prob = predict(GermanCredit.rpart$finalModel, test[,-10], type="prob")[,2]
pred_test_rf_prob = predict(GermanCredit.rf$finalModel, test[,-10], type="prob")[,2]

# Compute and plot the ROC curves
par(pty = "s")  # set layout to square
roc_dt = roc(test[,10], pred_test_dt_prob, plot=TRUE, legacy.axes=TRUE, xlab="False Positive Percentage", ylab="True Postive Percentage", col="#376a91", lwd=2, print.auc=TRUE)
plot.roc(test[,10], pred_test_rf_prob, col="#4dae21", lwd=2, print.auc=TRUE, add=TRUE, print.auc.y=0.4)
legend("bottomright", legend=c("Decision Tree", "Random Forest"), col=c("#376a91", "#4dae21"), lwd=2)
par(pty = "m")  # reset the layout

