
def main():

    import numpy as np
    import pandas as pd
    import xgboost as xgb # good tutorial -> https://www.datacamp.com/community/tutorials/xgboost-in-python
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import mean_squared_error

    train_data = pd.read_csv('./data/train.csv')
    test_data = pd.read_csv('./data/test.csv')

    #print(train_data.columns)
    #print(test_data.head())

    X_train = train_data.drop(['id','target'], axis = 1).copy()
    y_train = train_data['target'].copy()

    X_test = test_data.drop(['id'], axis = 1).copy()
    #y_test = test_data['target'].copy()  // We dont have target into test dataset
    
    print(X_train.head())
    print(y_train.head())
    print(X_test.head())

    '''
        Running XGBoost using default params
        XGBoost Doc: https://xgboost.readthedocs.io/en/latest/

        Step 1 : Set HyperParams: 
            - `learning_rate`: step size shrinkage used to prevent overfitting. Range is [0,1]
            - `max_depth`: determines how deeply each tree is allowed to grow during any boosting round
            - `subsample`: percentage of samples used per tree. Low value can lead to underfitting
            - `colsample_bytree`: percentage of features used per tree. High value can lead to overfitting
            - `n_estimators`: number of trees you want to build
            - `objective`: determines the loss function to be used like reg:linear for regression problems, reg:logistic for classification problems with only decision, binary:logistic for classification problems with probability

        Step 2: Instance XGBoostClassifier, fit and predict
        Step 3: Using Cross-Validation:
            - Convert the dataset into an optimized data structure called `Dmatrix` that XGBoost supports and gives it acclaimed performance and efficiency gains
            - Instance XGBoost cross-validation ,  fit and predict 
    '''
    # Setup Params - Step 1
    params = {  'objective':"reg:logistic",
                'colsample_bytree': 0.3,
                'learning_rate': 0.1,
                'max_depth': 5, 
                'alpha': 10
            }

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=123)
    
    # Step 2 
    xg_class = xgb.XGBClassifier(**params)
    xg_class.fit(X_train,y_train)

    preds = xg_class.predict(X_test)
    print('preds: %s', preds)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    print("RMSE: %f" % (rmse))

    # Using Cross-Validation - Step 3
    data_dmatrix = xgb.DMatrix(data=X_train,label=y_train)

    cv_results = xgb.cv(dtrain=data_dmatrix, params=params, nfold=3,
                    num_boost_round=50,early_stopping_rounds=10,metrics="rmse", as_pandas=True, seed=123)

    print(cv_results.head())
    print((cv_results["test-rmse-mean"]).tail(1))


    '''
        Running Logistic Regression
    '''

    '''
        Running Random Forest  
    '''




if __name__ == "__main__":
    main()