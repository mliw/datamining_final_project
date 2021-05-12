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


# 3 Plot features
ggplot(train_data)+geom_point(alpha = 0.5,aes(x = above_and_ground_area, y = log1p_SalePrice),color='darkblue')
ggsave("pics/best_feature.png")

ggplot(train_data)+geom_point(alpha = 0.5,aes(x = YrSold, y = log1p_SalePrice),color='darkred')
ggsave("pics/worst_feature.png")
