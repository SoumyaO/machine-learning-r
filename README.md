# Machine Learning in R
[code](https://github.com/SoumyaO/machine-learning-r/blob/main/code) | [Report](https://github.com/SoumyaO/machine-learning-r/blob/main/Report.pdf)

## Keywords
Decision Trees, Random Forests, Support Vector Machines (SVM), Kernel Engineering, kNN Classifier, Linear Discriminant Analysis (LDA), Principal Component Analysis (PCA)

## Description
This project covered several topics (classification algorithms) in machine learning and focussed on implementation of these techniques in R.

- The first sub-project involved fitting a decision tree to financial data and determining optimal tree size through cross-validation. Cost complexity pruning was performed to avoid overfitting. This was followed by fitting a random forest classifier on the data and identifying the most relevant features. The best model of the two, a random forest with 30 features, was chosen by performing AUC-ROC analysis.

- The second sub-project involved generating a simulated dataset and fitting support vector machines with three different kernels - linear, polynomial and radial basis function (RBF) for classifying the simulated dataset. A dataset with samples in three concentric rings was chosen to compare and evaluate the strengths and weaknesses of each of the kernels.

- The third sub-project involved classfication of medical data on hyperthyroidism using a kNN classifier and an LDA classifier. 5-fold cross-validation was used to determine the value of k for the kNN classifier. The area under the curve was used to compare the classifiers on 10 random splits of the dataset. It was a better metric for comparison over accuracy as the dataset is unbalanced.

- The fourth sub-project involved applying PCA and Fisher's LDA on German Credit data. PCA was applied to remove features that are highly correlated and reduce the dimension to explain 90% of the variance in the data. Fisher's LDA was then applied on the chosen principal components for classification.
