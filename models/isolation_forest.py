from sklearn.ensemble import IsolationForest

def isoforest(x_train, x_test, estimators, contamination, max_samples):
    model = IsolationForest(n_estimators=estimators, contamination=contamination, max_samples=max_samples, random_state=42)
    
    model.fit(x_train)
    y_pred_raw = model.predict(x_test)
    y_pred = [1 if x == -1 else 0 for x in y_pred_raw]
    
    return model, y_pred