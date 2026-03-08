from sklearn.tree import DecisionTreeRegressor

def create_train_test(daily_df):
    X = daily_df.drop(columns=['CRASH DATE','DAILY_INJURIES'])
    y = daily_df['DAILY_INJURIES']
    split_index = int(len(daily_df)*0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
    return X_train, X_test, y_train, y_test

def train_dtr(X_train, y_train):
    dtr = DecisionTreeRegressor(
        criterion="squared_error",
        max_depth=4,
        min_samples_split=0.1,
        min_samples_leaf=10,
        random_state=123
    )
    dtr.fit(X_train, y_train)
    return dtr