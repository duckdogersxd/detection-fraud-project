from sklearn.neighbors import LocalOutlierFactor

def lof(x_train, x_test, neighbors, metric, contamination):
    model = LocalOutlierFactor(n_neighbors=neighbors, novelty=True, metric=metric, contamination=contamination)
    
    model.fit(x_train)
    y_pred_raw = model.predict(x_test)
    y_pred = [1 if x == -1 else 0 for x in y_pred_raw]
    
    return model, y_pred