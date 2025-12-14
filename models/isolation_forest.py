from sklearn.ensemble import IsolationForest

def isoforest(X_train, X_test, estimators, contamination, random_state, n_jobs):
    iso_forest = IsolationForest(n_estimators=estimators, contamination=contamination, random_state=random_state, n_jobs=n_jobs)
    
    iso_forest.fit(X_train)
    y_pred_raw = iso_forest.predict(X_test)
    y_pred = [1 if x == -1 else 0 for x in y_pred_raw]
    
    return iso_forest, y_pred