library(caret)
library(pROC)

# Load dataset
data = read.csv("newthyroid.txt")
data$class = as.factor(data$class)

# split the dataset into training (70%) and test (30%) splits 10 times
set.seed(200)
num_times = 10
training_sample_rows = createDataPartition(data$class, p = 0.7, list = FALSE, times = num_times)

# create empty lists for storing the values of AUC for lda and knn
auc_lda = rep(0, num_times)
auc_knn = rep(0, num_times)
k_best = rep(0, num_times)

fp_knn = rep(0, num_times)
fn_knn = rep(0, num_times)
fp_lda = rep(0, num_times)
fn_lda = rep(0, num_times)

# Perform analysis for each of the partitions
for (i in 1:num_times) {
  train=data[training_sample_rows[,i],] # training set - pick the rows computed above
  test=data[-training_sample_rows[,i],] # test set - pick the rows not chosen above  
  
  # k-NN with 5-fold cross validation, AUC as the metric to find the best value of k
  # cross validation params
  fitControl = trainControl(method="cv",
                            number=5,
                            summaryFunction = twoClassSummary,
                            classProbs=TRUE)
  # Values of k will be from (3, 5, 7, 9, 11, 13, 15)
  kGrid=expand.grid(k=c(3, 5, 7, 9, 11, 13, 15))
  
  knnModel = train(train[,-1], 
                   train[,1], 
                   method="knn",
                   metric="ROC",
                   trControl=fitControl,
                   preProcess = c("center", "scale"),
                   tuneGrid=kGrid)
  knnModel
  k_best[i]=knnModel$bestTune$k
  
  # predict on the test set and get the confusion matrix
  knnPred = predict(knnModel, test[,-1])
  cm = confusionMatrix(knnPred, test[,1])
  
  # compute false positives and false negatives
  fp_knn[i] = cm$table[1,2]
  fn_knn[i] = cm$table[2,1]
  
  # compute ROC on the test set
  knnROC = roc(predictor=as.numeric(knnPred),
               response=test[,1], 
               direction="<")
  
  # Compute AUC
  auc_knn[i] = knnROC$auc
  print(knnROC$auc)
  
  # LDA
  # ----
  
  # LDA model on the dataset
  ldaModel=train(train[,-1],
                 train[,1], 
                 method = "lda",
                 trControl = trainControl(method = "none"),
                 preProcess = c("center", "scale")
                 )
  ldaModel$finalModel
  
  # Prediction on the test dataset and get the confusion matrix
  ldaPred=predict(ldaModel,test[,-1])
  cm = confusionMatrix(ldaPred, test[,1])
  
  # compute false positives and false negatives
  fp_lda[i] = cm$table[1,2]
  fn_lda[i] = cm$table[2,1]
  
  # compute ROC on the test set
  ldaROC = roc(predictor=as.numeric(ldaPred),
               response=as.numeric(test[,1]), 
               direction="<")
  
  # Compute AUC
  auc_lda[i] = ldaROC$auc
  print(knnROC$auc)
}

# Best k for the ten splits done in Knn
print(k_best)

# Print 10 AUCs of kNN, LDA
print(auc_knn)
print(auc_lda)

# Plot boxplot of AUCs over ten splits on the dataset of knn and LDA
aucs = data.frame(KNN=auc_knn, LDA=auc_lda)
boxplot(aucs, ylab="AUC", main="Box plot of AUC metric for 10 test set predictions")

# Mean AUC of knn, LDA
print(mean(auc_knn))
print(mean(auc_lda))

# Variance of AUC 
print(var(auc_knn))
print(var(auc_lda))

# false positives
print(fp_knn)
print(fp_lda)

print(mean(fp_knn))
print(mean(fp_lda))

# false negatives
print(fn_knn)
print(fn_lda)

print(mean(fn_knn))
print(mean(fn_lda))