import lightgbm as lgb


# https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html
def get_model(train, feature_set):
    model = lgb.LGBMRegressor(
    n_estimators=2000,
    learning_rate=0.01,
    max_depth=5,
    num_leaves=2**5-1,
    colsample_bytree=0.1
    )
    model.fit(
        train[feature_set],
        train["target"]
    )
    return model