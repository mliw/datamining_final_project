import pandas as pd
import numpy as np


if __name__=="__main__":
    
    # 1 Load the data
    train_data = pd.read_csv("data/original_data/train.csv")
    train_data["log1p_SalePrice"] = np.log1p(train_data.SalePrice)
    test_data = pd.read_csv("data/original_data/test.csv")
    train_data.index = train_data['Id']
    test_data.index = test_data['Id']
    train_id = train_data['Id']
    test_id = test_data['Id']
    del(train_data['Id'])
    del(test_data['Id'])
    all_data = pd.concat((train_data, test_data))
    all_data["log1p_SalePrice"] = np.log1p(all_data.SalePrice)
    del(all_data['SalePrice'])
    del(train_data['SalePrice'])
    print("all_data size is : {}".format(all_data.shape))
    train_test = all_data.copy()
    
    # 2 Impute missing feature
    missing_features = all_data.columns[all_data.isnull().sum(axis = 0)>0]
    """
    We would fill 18 missing features at this time
    ['LotFrontage','Alley','MasVnrType','MasVnrArea','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',
 'Electrical','FireplaceQu','GarageType','GarageYrBlt','GarageFinish','GarageQual','GarageCond','Fence','MiscFeature']
    """   
    train_test["LotFrontage"] = train_test.groupby("Neighborhood")["LotFrontage"].transform(
        lambda x: x.fillna(x.median())) 
    
    train_test["Alley"] = train_test["Alley"].fillna("None")   
    train_test["MasVnrType"] = train_test["MasVnrType"].fillna("None")
    train_test["MasVnrArea"] = train_test["MasVnrArea"].fillna(train_test["MasVnrArea"].median())    
    train_test["BsmtQual"] = train_test["BsmtQual"].fillna("no")     
    train_test["BsmtCond"] = train_test["BsmtCond"].fillna("no")     
    train_test["BsmtExposure"] = train_test["BsmtExposure"].fillna("nobase")     
    train_test["BsmtFinType1"] = train_test["BsmtFinType1"].fillna("nobase") 
    train_test["BsmtFinType2"] = train_test["BsmtFinType2"].fillna("nobase") 
    train_test["Electrical"] = train_test["Electrical"].fillna(train_test["Electrical"].mode()[0]) 
    train_test["FireplaceQu"] = train_test["FireplaceQu"].fillna(train_test["FireplaceQu"].mode()[0]) 
    train_test["GarageType"] = train_test["GarageType"].fillna("nogarage") 
    train_test["GarageYrBlt"] = train_test["GarageYrBlt"].fillna(train_test["GarageYrBlt"].median()) 
    train_test["GarageFinish"] = train_test["GarageFinish"].fillna("nogarage") 
    train_test["GarageQual"] = train_test["GarageQual"].fillna("nogarage") 
    train_test["GarageCond"] = train_test["GarageCond"].fillna("nogarage") 
    train_test["Fence"] = train_test["Fence"].fillna("nofence") 
    train_test["MiscFeature"] = train_test["MiscFeature"].fillna("None") 

    remaining_missing = list(train_test.isnull().sum().index[train_test.isnull().sum()>0])
    """
    We have 16 missing features at this time
['MSZoning','Utilities','Exterior1st','Exterior2nd','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF',
 'BsmtFullBath','BsmtHalfBath','KitchenQual','Functional','GarageCars','GarageArea','PoolQC','SaleType']
    """
    train_test["MSZoning"] = train_test["MSZoning"].fillna(train_test["MSZoning"].mode()[0])
    train_test["Utilities"] = train_test["Utilities"].fillna(train_test["Utilities"].mode()[0])
    train_test["Exterior1st"] = train_test["Exterior1st"].fillna(train_test["Exterior1st"].mode()[0])
    train_test["Exterior2nd"] = train_test["Exterior2nd"].fillna(train_test["Exterior2nd"].mode()[0])
    train_test["BsmtFinSF1"] = train_test["BsmtFinSF1"].fillna(train_test["BsmtFinSF1"].median())
    train_test["BsmtFinSF2"] = train_test["BsmtFinSF2"].fillna(train_test["BsmtFinSF2"].median())
    train_test["BsmtUnfSF"] = train_test["BsmtUnfSF"].fillna(train_test["BsmtUnfSF"].median())
    train_test["TotalBsmtSF"] = train_test["TotalBsmtSF"].fillna(train_test["TotalBsmtSF"].median())
    train_test["BsmtFullBath"] = train_test["BsmtFullBath"].fillna(train_test["BsmtFullBath"].mode()[0])
    train_test["BsmtHalfBath"] = train_test["BsmtHalfBath"].fillna(train_test["BsmtHalfBath"].mode()[0])
    train_test["KitchenQual"] = train_test["KitchenQual"].fillna(train_test["KitchenQual"].mode()[0])
    train_test["Functional"] = train_test["Functional"].fillna(train_test["Functional"].mode()[0])
    train_test["GarageCars"] = train_test["GarageCars"].fillna(train_test["GarageCars"].median())
    train_test["GarageArea"] = train_test["GarageArea"].fillna(0)
    train_test["PoolQC"] = train_test["PoolQC"].fillna("nopool")
    train_test["SaleType"] = train_test["SaleType"].fillna(train_test["SaleType"].mode()[0])
    print("The number of nan is {}".format(train_test.isnull().sum().sum()))


    # 3 Design new features
    print("="*30)
    print("Divide features into different categories:")
    print("""
    {'OverallQual':Rates the overall material and finish of the house}
    {'Neighborhood':Physical locations within Ames city limits}
    {'GarageCars':"Size of garage in car capacity",'GarageArea': "Size of garage in square feet",'GarageFinish': "Interior finish of the garage" \
        'GarageYrBlt': "Year garage was built", 'GarageType': "Garage location"}
    {'ExterQual':Evaluates the quality of the material on the exterior,'BsmtQual': Evaluates the height of the basement \
        'Foundation': "Type of foundation"}
    {'KitchenQual': Kitchen quality}
    {'GrLivArea': Above grade (ground) living area square feet,'TotRmsAbvGrd':"Total rooms above grade (does not include bathrooms)", \
        'FullBath': "Full bathrooms above grade",'1stFlrSF': "First Floor square feet",'TotalBsmtSF: "Total square feet of basement area"}
    {'YearBuilt': "Original construction date",'YearRemodAdd': "Remodel date (same as construction date if no remodeling or additions)"}
    {'MSSubClass': "Identifies the type of dwelling involved in the sale."}
    {'Fireplaces': "Number of fireplaces"}
    """)
    print("="*30)

    # 1.4.1 Explore garage
    garage_words = ['GarageType','GarageYrBlt','GarageFinish','GarageCars','GarageArea','GarageQual','GarageCond']   
    tem_garage = train_test[garage_words]
    def area_per_car(clause):
        result = clause['GarageArea'] / clause['GarageCars'] if clause['GarageCars']!=0 else 0
        return result
    train_test["area_per_car"] = train_test.apply(area_per_car,axis = 1)

    # 1.4.2 Explore area
    train_test["above_and_ground_area"] = train_test["TotalBsmtSF"]+train_test["GrLivArea"]
    train_test['Total_Bathrooms'] = (train_test['FullBath'] + (0.5 * train_test['HalfBath']) +
                               train_test['BsmtFullBath'] + (0.5 * train_test['BsmtHalfBath']))
    train_test["one_and_two"] = train_test["1stFlrSF"]+train_test["2ndFlrSF"]
    train_test['Total_Porch_Area'] = (train_test['OpenPorchSF'] + train_test['3SsnPorch'] + train_test['EnclosedPorch'] + train_test['ScreenPorch'] + train_test['WoodDeckSF'])

    # 1.5 Delete outliers
    train_test_save = train_test.copy()
    tem_train = train_test.loc[train_id,:]
    tem_train["log1p_SalePrice"] = train_data.loc[train_id,"log1p_SalePrice"]
     
    logi_0 = np.logical_and(tem_train.log1p_SalePrice>=12.25, tem_train.OverallQual==4)
    logi_1 = np.logical_and(tem_train.log1p_SalePrice<=11.5, tem_train.OverallQual==7)    
    logi_2 = np.logical_and(tem_train.log1p_SalePrice<=12.5, tem_train.OverallQual==10)
    logi_3 = np.logical_and(tem_train.log1p_SalePrice<=12.5, tem_train.above_and_ground_area>=6000)   
    logi_4 = np.logical_and(tem_train.log1p_SalePrice<=11, tem_train.ExterQual=="Gd")     
    logi_5 = np.logical_and(tem_train.log1p_SalePrice<=12.5, tem_train.one_and_two>=4000)    
    logi_6 = np.logical_and(tem_train.log1p_SalePrice<=11.5, tem_train.GarageArea>=1200)       
    logi_7 = np.logical_and(tem_train.log1p_SalePrice<=12.5, tem_train.TotalBsmtSF>=5000)    
    logi = logi_0
    
    for i in range(8):
        exec("""logi = np.logical_or(logi,logi_"""+str(i)+""")""")

    tem_train.drop(index = tem_train.index[logi],axis = 0,inplace=True)
    tem_test = train_test.loc[test_id,:]
    train_id = tem_train.index
    test_id = tem_test.index
    train_test = pd.concat([tem_train,tem_test],axis = 0)
    train_test = pd.get_dummies(train_test)
    
    train_data = train_test.loc[train_id,:]
    test_data = train_test.loc[test_id,:]
    
    train_data.to_csv("data/train_data_python.csv")
    test_data.to_csv("data/test_data_python.csv")   

