library(caret)
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
train_data = subset(train_data,select = -X)
train_data = subset(train_data,select = -Id)


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


# 2 Test xgb(max_depth=2)
final_result = data.frame()
for (gamma_value in c(0,0.3,0.6,1)){
  for (min_child_weight_value in c(0,0.3,0.6,1)){
    result_err_vec = c()
    for (rd_state in 1:20){
    indexs = divide_data(train_data,rd_state)
    nround = 100
    param = list(gamma=gamma_value,min_child_weight=min_child_weight_value,max_depth=2,eta=0.3, silent=0)
    err_vec = c()
    # Prepare index[i]
    for (i in 1:5){
      train_batch = train_data[-indexs[[i]],c(best_features,"log1p_SalePrice")]
      test_batch = train_data[indexs[[i]],c(best_features,"log1p_SalePrice")]
      X = subset(train_batch,select = -log1p_SalePrice)
      Y = train_batch$log1p_SalePrice
      X = as.matrix(X)
      Y = as.matrix(Y)
      dtrain = xgb.DMatrix(X,label=Y)
      
      bst = xgb.train(params = param, data = dtrain, nrounds = nround, nthread = 2) 
      test_X = subset(test_batch,select = -log1p_SalePrice)
      test_Y = test_batch$log1p_SalePrice
      test_X = as.matrix(test_X)
      test_Y = as.matrix(test_Y)
      xgb_prediction = predict(bst,test_X)
      batch_err = rmse(test_Y,xgb_prediction)
      err_vec = c(err_vec,batch_err)
    }
    result_err_vec = c(result_err_vec,mean(err_vec))
    }
    result = c(gamma_value,min_child_weight_value,mean(result_err_vec))
    final_result = rbind(final_result,result)
  }
}
colnames(final_result)=c("gamma_value","min_child_weight_value","cross-validation error")
write.csv(final_result,file = "data/output/section2_xgb_depth2_final_result.csv")


# 3 Test xgb(max_depth=3)
final_result = data.frame()
for (gamma_value in c(0,0.3,0.6,1)){
  for (min_child_weight_value in c(0,0.3,0.6,1)){
    result_err_vec = c()
    for (rd_state in 1:20){
      indexs = divide_data(train_data,rd_state)
      nround = 100
      param = list(gamma=gamma_value,min_child_weight=min_child_weight_value,max_depth=3,eta=0.3, silent=0)
      err_vec = c()
      # Prepare index[i]
      for (i in 1:5){
        train_batch = train_data[-indexs[[i]],c(best_features,"log1p_SalePrice")]
        test_batch = train_data[indexs[[i]],c(best_features,"log1p_SalePrice")]
        X = subset(train_batch,select = -log1p_SalePrice)
        Y = train_batch$log1p_SalePrice
        X = as.matrix(X)
        Y = as.matrix(Y)
        dtrain = xgb.DMatrix(X,label=Y)
        
        bst = xgb.train(params = param, data = dtrain, nrounds = nround, nthread = 2) 
        test_X = subset(test_batch,select = -log1p_SalePrice)
        test_Y = test_batch$log1p_SalePrice
        test_X = as.matrix(test_X)
        test_Y = as.matrix(test_Y)
        xgb_prediction = predict(bst,test_X)
        batch_err = rmse(test_Y,xgb_prediction)
        err_vec = c(err_vec,batch_err)
      }
      result_err_vec = c(result_err_vec,mean(err_vec))
    }
    result = c(gamma_value,min_child_weight_value,mean(result_err_vec))
    final_result = rbind(final_result,result)
  }
}
colnames(final_result)=c("gamma_value","min_child_weight_value","cross-validation error")
write.csv(final_result,file = "data/output/section2_xgb_depth3_final_result.csv")


