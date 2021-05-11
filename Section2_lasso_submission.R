library(caret)
library(glmnet)
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


# 2 Fit on the data
X = subset(train_data,select=-log1p_SalePrice)
X = as.matrix(X)
lasso_model = glmnet(X, train_data$log1p_SalePrice, family="gaussian", lambda=0.2, alpha=0)
print("Fitting rmse is:")
print(rmse(predict(lasso_model,X),train_data$log1p_SalePrice))


# 3 Make prediction
submission_prediction = predict(lasso_model,as.matrix(test_data))
SalePrice = exp(submission_prediction)-1
section2_lasso_submission = cbind(test_id,SalePrice)
section2_lasso_submission = as.data.frame(section2_lasso_submission)
colnames(section2_lasso_submission) = c("Id","SalePrice")
write.csv(section2_lasso_submission,file = "data/output/section2_lasso_submission.csv",row.names = FALSE)





