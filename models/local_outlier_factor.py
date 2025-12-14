from sklearn.neighbors import LocalOutlierFactor

def lof(x_train, x_test, neighbors, metric, contamination):
    lof = LocalOutlierFactor(n_neighbors=neighbors, novelty=True, metric=metric, contamination=contamination)
    
    lof.fit(x_train)
    y_pred_raw = lof.predict(x_test)
    y_pred = [1 if x == -1 else 0 for x in y_pred_raw]
    
    return lof, y_pred