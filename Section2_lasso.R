library(caret)
library(glmnet)


# 1 Load data
train_data = read.csv("data/train_data_R.csv", stringsAsFactors=TRUE)
test_data = read.csv("data/test_data_R.csv", stringsAsFactors=TRUE)
train_data = subset(train_data,select = -X)
train_data = subset(train_data,select = -Id)
test_id = test_data$Id
test_data = subset(test_data,select = -X)
test_data = subset(test_data,select = -Id)
test_data = subset(test_data,select = -log1p_SalePrice)


# 2 lasso parameter tunning
result = list()
glmnetTuningGrid = expand.grid(alpha = seq(0, 1, 0.2),
                               lambda = seq(0, 1, 0.2))
for (i in 1:20){
  print(i)
  set.seed(i)
  myControl = trainControl(method = "cv", number = 5, verboseIter = FALSE)
  cv_result = train(log1p_SalePrice ~ ., 
                        data = train_data,
                        method = "glmnet",
                        trControl = myControl,
                        tuneGrid = glmnetTuningGrid)
  result[[i]] = cv_result$results
}

section2_lasso_final_result = result[[1]][,c("alpha","lambda","RMSE")]
section2_lasso_final_result$RMSE = 0
for (i in 1:20){
  section2_lasso_final_result$RMSE = section2_lasso_final_result$RMSE + result[[i]]$RMSE
}
section2_lasso_final_result$RMSE = section2_lasso_final_result$RMSE / 20
write.csv(section2_lasso_final_result,file = "data/output/section2_lasso_final_result.csv")





