library(caret)
library(glmnet)
library(e1071)
library(xgboost)
library(ModelMetrics)


# 0 Functions
# divide data into 5 folds
divide_data = function(data,rand_num){
  set.seed(rand_num)
  indexs = createFolds(1:dim(data)[1], k = 5, list = TRUE, returnTrain = FALSE)
  return(indexs)
  # test_data = data[indexs[[i]],]
  # train_data = data[-indexs[[i]],]
}


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


# 3 Tunning parameters for SVM
final_result = data.frame()
for (eps_value in seq(0,1,0.1)){
    for (kernel_value in c("linear")){
      err_vec_high = c()
      for (rd_state in 1:20){
      indexs = divide_data(train_data,rd_state)
      err_vec = c()
      for (i in 1:5){
        train_batch = train_data[-indexs[[i]],c(best_features,"log1p_SalePrice")]
        test_batch = train_data[indexs[[i]],c(best_features,"log1p_SalePrice")]
        X = subset(train_batch,select = -log1p_SalePrice)
        Y = train_batch$log1p_SalePrice
        X = as.matrix(X)
        Y = as.matrix(Y)
        svm_model = svm(X,Y,eps=eps_value,kernel=kernel_value)
        test_X = subset(test_batch,select = -log1p_SalePrice)
        test_Y = test_batch$log1p_SalePrice
        test_X = as.matrix(test_X)
        test_Y = as.matrix(test_Y)
        svm_prediction = predict(svm_model,test_X)
        batch_err = rmse(test_Y,svm_prediction)
        err_vec = c(err_vec,batch_err)
      }
      mean_cv_err = mean(err_vec)
      err_vec_high = c(err_vec_high,mean_cv_err)
      }
    tem_result = c(eps_value,kernel_value,mean(err_vec_high))
    print(tem_result)
    final_result = rbind(final_result,tem_result)
    }
}
colnames(final_result)=c("eps_value","kernel_type","cross-validation error")
write.csv(final_result,file = "data/output/section2_svr_final_result.csv")


                       
