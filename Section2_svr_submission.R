library(caret)
library(glmnet)
library(e1071)
library(xgboost)
library(ModelMetrics)


# 1 Load data
train_data = read.csv("data/train_data_R.csv", stringsAsFactors=TRUE)
test_data = read.csv("data/test_data_R.csv", stringsAsFactors=TRUE)
train_data = subset(train_data,select = -X)
train_data = subset(train_data,select = -Id)
test_id = test_data$Id
test_data = subset(test_data,select = -X)
test_data = subset(test_data,select = -Id)
test_data = subset(test_data,select = -log1p_SalePrice)


# 2 Fit on the whole data set to get best_features(top25)
nround = 100
param = list(max_depth=2,eta=0.3, silent=0)
X = subset(train_data,select = -log1p_SalePrice)
Y = train_data$log1p_SalePrice
X = as.matrix(X)
Y = as.matrix(Y)
dtrain = xgb.DMatrix(X,label=Y)
bst = xgb.train(params = param, data = dtrain, nrounds = nround, nthread = 2) 
importance_matrix = xgb.importance(model = bst)
best_features = importance_matrix$Feature[1:25]
train_data = train_data[,c(best_features,"log1p_SalePrice")]
test_data = test_data[,best_features]


# 3 Fitting SVM
eps_value = 0.1
kernel_value = "linear"
X = subset(train_data,select = -log1p_SalePrice)
Y = train_data$log1p_SalePrice
X = as.matrix(X)
Y = as.matrix(Y)
svm_model = svm(X,Y,eps=eps_value,kernel=kernel_value)
print("Fitting rmse is:")
print(rmse(predict(svm_model,X),Y))


# 4 Make prediction
submission_prediction = predict(svm_model,as.matrix(test_data))
SalePrice = exp(submission_prediction)-1
section2_svr_submission = cbind(test_id,SalePrice)
section2_svr_submission = as.data.frame(section2_svr_submission)
colnames(section2_svr_submission) = c("Id","SalePrice")
write.csv(section2_svr_submission,file = "data/output/section2_svr_submission.csv",row.names = FALSE)

