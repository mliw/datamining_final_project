library(bazar)
library(tidyverse)
library(Hmisc)
library(dummies)


# 1 Load data
train_data = read.csv("data/original_data/train.csv", stringsAsFactors=TRUE)
train_data$log1p_SalePrice = log(train_data$SalePrice+1)
test_data = read.csv("data/original_data/test.csv", stringsAsFactors=TRUE)
rownames(test_data) = test_data$Id
rownames(train_data) = train_data$Id
train_id = train_data$Id
test_id = test_data$Id
# Define NAs to merge
test_data$log1p_SalePrice = NA
test_data$SalePrice = NA
all_data = rbind(train_data,test_data)
all_data = subset(all_data, select = -SalePrice)
train_data = subset(train_data, select = -SalePrice)
print("all_data size is")
print(dim(all_data))
# At this stage we get all_data. all_data combine train and test data together
# and log1p_SalePrice is prediction target.
train_test = all_data


# 2 Impute missing feature
mis_num = colSums(is.na(all_data))
print(mis_num[mis_num>0])
# 2.1 Impute LotFrontage ########Question 1!
train_test_summary = train_test %>%
  group_by(Neighborhood) %>%
  summarise(median_LotFrontage=median(LotFrontage,na.rm=TRUE))
train_test_summary = as.data.frame(train_test_summary)
rownames(train_test_summary) = train_test_summary$Neighborhood
train_test_summary = subset(train_test_summary,select = -Neighborhood)
for (i in 1:dim(train_test)[1]){
  if (is.na(train_test[i,"LotFrontage"])){
    train_test[i,"LotFrontage"] = train_test_summary[train_test[i,"Neighborhood"],]
  }
}
# 2.2 Alley
train_test$Alley = impute(train_test$Alley,"None")
# 2.3 MasVnrType
train_test$MasVnrType = impute(train_test$MasVnrType,"None")
# 2.4 MasVnrArea
train_test$MasVnrArea = impute(train_test$MasVnrArea,median)
# 2.5 BsmtQual
train_test$BsmtQual = impute(train_test$BsmtQual,"no")
# 2.6 BsmtCond
train_test$BsmtCond = impute(train_test$BsmtCond,"no")
# 2.7 BsmtExposure
train_test$BsmtExposure = impute(train_test$BsmtExposure,"nobase")
# 2.8 BsmtFinType1
train_test$BsmtFinType1 = impute(train_test$BsmtFinType1,"nobase")
# 2.9 BsmtFinType2
train_test$BsmtFinType2 = impute(train_test$BsmtFinType2,"nobase")
# 2.10 Electrical
getmode <- function(v) {
  v = v[!is.na(v)]
  uniqv <- unique(v)
  uniqv[which.max(tabulate(match(v, uniqv)))]
}
Electrical_mode = getmode(train_test$Electrical)
train_test$Electrical = impute(train_test$Electrical,Electrical_mode)
# 2.11 FireplaceQu
FireplaceQu_mode = getmode(train_test$FireplaceQu)
train_test$FireplaceQu = impute(train_test$FireplaceQu,FireplaceQu_mode)
# 2.12 GarageType
train_test$GarageType = impute(train_test$GarageType,"nogarage")
# 2.13 GarageYrBlt
train_test$GarageYrBlt = impute(train_test$GarageYrBlt,median)
# 2.14 GarageFinish
train_test$GarageFinish = impute(train_test$GarageFinish,"nogarage")
# 2.15 GarageQual
train_test$GarageQual = impute(train_test$GarageQual,"nogarage")
# 2.16 GarageCond
train_test$GarageCond = impute(train_test$GarageCond,"nogarage")
# 2.17 Fence
train_test$Fence = impute(train_test$Fence,"nofence")
# 2.18 MiscFeature
train_test$MiscFeature = impute(train_test$MiscFeature,"None")
# We have 16 missing features at this time
# ['MSZoning','Utilities','Exterior1st','Exterior2nd','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF',
#  'BsmtFullBath','BsmtHalfBath','KitchenQual','Functional','GarageCars','GarageArea','PoolQC','SaleType']
# 2.19 MSZoning
MSZoning_mode = getmode(train_test$MSZoning)
train_test$MSZoning = impute(train_test$MSZoning,MSZoning_mode)
# 2.20 Utilities
Utilities_mode = getmode(train_test$Utilities)
train_test$Utilities = impute(train_test$Utilities,Utilities_mode)
# 2.21 Exterior1st
Exterior1st_mode = getmode(train_test$Exterior1st)
train_test$Exterior1st = impute(train_test$Exterior1st,Exterior1st_mode)
# 2.22 Exterior2nd
Exterior2nd_mode = getmode(train_test$Exterior2nd)
train_test$Exterior2nd = impute(train_test$Exterior2nd,Exterior2nd_mode)
# 2.23 BsmtFinSF1
train_test$BsmtFinSF1 = impute(train_test$BsmtFinSF1,median)
# 2.24 BsmtFinSF2
train_test$BsmtFinSF2 = impute(train_test$BsmtFinSF2,median)
# 2.25 BsmtUnfSF
train_test$BsmtUnfSF = impute(train_test$BsmtUnfSF,median)
# 2.26 TotalBsmtSF
train_test$TotalBsmtSF = impute(train_test$TotalBsmtSF,median)
# 2.27 BsmtFullBath
BsmtFullBath_mode = getmode(train_test$BsmtFullBath)
train_test$BsmtFullBath = impute(train_test$BsmtFullBath,BsmtFullBath_mode)
# 2.28 BsmtHalfBath
BsmtHalfBath_mode = getmode(train_test$BsmtHalfBath)
train_test$BsmtHalfBath = impute(train_test$BsmtHalfBath,BsmtHalfBath_mode)
# 2.29 KitchenQual
KitchenQual_mode = getmode(train_test$KitchenQual)
train_test$KitchenQual = impute(train_test$KitchenQual,KitchenQual_mode)
# 2.30 Functional
Functional_mode = getmode(train_test$Functional)
train_test$Functional = impute(train_test$Functional,Functional_mode)
# 2.31 GarageCars
train_test$GarageCars = impute(train_test$GarageCars,median)
# 2.32 GarageArea
train_test$GarageArea = impute(train_test$GarageArea,0)
# 2.33 PoolQC
train_test$PoolQC = impute(train_test$PoolQC,"nopool")
# 2.34 SaleType
SaleType_mode = getmode(train_test$SaleType)
train_test$SaleType = impute(train_test$SaleType,SaleType_mode)
mis_num = colSums(is.na(train_test))
print(mis_num[mis_num>0])
# At this point, we have finished imputing missing features.


# 3 Design new features
# 3.1 area_per_car
area_per_car = train_test$GarageArea/train_test$GarageCars
area_per_car[is.na(area_per_car)] = 0
train_test$area_per_car = area_per_car
# 3.2 above_and_ground_area
train_test$above_and_ground_area = train_test$TotalBsmtSF+train_test$GrLivArea
# 3.3 Total_Bathrooms
train_test$Total_Bathrooms = train_test$FullBath+ 0.5*train_test$HalfBath+
                                   train_test$BsmtFullBath+ 0.5*train_test$BsmtHalfBath
# 3.4 one_and_two
train_test$one_and_two = train_test$X1stFlrSF+train_test$X2ndFlrSF
# 3.5 Total_Porch_Area
train_test$Total_Porch_Area = train_test$OpenPorchSF + train_test$X3SsnPorch + train_test$EnclosedPorch + train_test$ScreenPorch + train_test$WoodDeckSF


# 4 Delete outliers
train_test_save = train_test
tem_train = train_test[train_id,]
logi_0 = tem_train$log1p_SalePrice>=12.25 & tem_train$OverallQual==4
logi_1 = tem_train$log1p_SalePrice<=11.5 & tem_train$OverallQual==7
logi_2 = tem_train$log1p_SalePrice<=12.5 & tem_train$OverallQual==10
logi_3 = tem_train$log1p_SalePrice<=12.5 & tem_train$above_and_ground_area>=6000
logi_4 = tem_train$log1p_SalePrice<=11 & tem_train$ExterQual=="Gd"
logi_5 = tem_train$log1p_SalePrice<=12.5 & tem_train$one_and_two>=4000
logi_6 = tem_train$log1p_SalePrice<=11.5 & tem_train$GarageArea>=1200
logi_7 = tem_train$log1p_SalePrice<=12.5 & tem_train$TotalBsmtSF>=5000
logi = logi_0 |logi_1 |logi_2 |logi_3 |logi_4 |logi_5 |logi_6 |logi_7 
maintain_index = train_id[!logi]
tem_train = train_test[maintain_index,]
tem_test = train_test[test_id,]
final_combination = rbind(tem_train,tem_test)


# 5 Output data
select_rows = function(data,vec){
  result = c()
  for (i in 1:dim(data)[1]){
    if (data$Id[i] %in% vec){
      result = c(result,i)
    }
  }
  return(result)
}

train_test_new <- dummy.data.frame(final_combination, sep = "_")
train_data = train_test_new[select_rows(train_test_new,maintain_index),]
test_data =  train_test_new[select_rows(train_test_new,test_id),]
write.csv(train_data,file = "data/train_data_R.csv")
write.csv(test_data,file = "data/test_data_R.csv")


# 6 Check missing numbers
mis_num = colSums(is.na(train_data))
print(mis_num[mis_num>0])
mis_num = colSums(is.na(test_data))
print(mis_num[mis_num>0])
print(dim(test_data))
