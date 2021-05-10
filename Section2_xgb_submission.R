library(caret)
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


# 2 Fitting on the best parameter:
nround = 100
gamma_value	= 0
min_child_weight_value = 0
param = list(gamma=gamma_value,min_child_weight=min_child_weight_value,max_depth=2,eta=0.3, silent=0)
X = X[,best_features]
Y = train_data$log1p_SalePrice
X = as.matrix(X)
Y = as.matrix(Y)
dtrain = xgb.DMatrix(X,label=Y)
bst = xgb.train(params = param, data = dtrain, nrounds = nround, nthread = 2)
print("Fitting error is:")
print(rmse(predict(bst,X),Y))


# 3 Make submission section2_xgb_submission
submission_prediction = predict(bst,as.matrix(test_data[,best_features]))
SalePrice = exp(submission_prediction)-1
section2_xgb_submission = cbind(test_id,SalePrice)
section2_xgb_submission = as.data.frame(section2_xgb_submission)
colnames(section2_xgb_submission) = c("Id","SalePrice")
write.csv(section2_xgb_submission,file = "data/output/section2_xgb_submission.csv",row.names = FALSE)


