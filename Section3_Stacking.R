# 1 Load data
lasso_submission= read.csv("data/output/section2_lasso_submission.csv", stringsAsFactors=TRUE)
svr_submission= read.csv("data/output/section2_svr_submission.csv", stringsAsFactors=TRUE)
xgb_submission= read.csv("data/output/section2_xgb_submission.csv", stringsAsFactors=TRUE)


# 2 lasso+xgb
stacking = svr_submission
stacking$SalePrice = 0
for (model_weight in c(0.2,0.4,0.6,0.8)){
  stacking$SalePrice = model_weight*lasso_submission$SalePrice+(1-model_weight)*xgb_submission$SalePrice
  save_str = paste0("data/output/section3_",as.character(model_weight),"_lasso_xgb.csv")
  write.csv(stacking,file = save_str,row.names = FALSE)
}


# 3 svr+xgb
stacking = svr_submission
stacking$SalePrice = 0
for (model_weight in c(0.2,0.4,0.6,0.8)){
  stacking$SalePrice = model_weight*svr_submission$SalePrice+(1-model_weight)*xgb_submission$SalePrice
  save_str = paste0("data/output/section3_",as.character(model_weight),"_svr_xgb.csv")
  write.csv(stacking,file = save_str,row.names = FALSE)
}
