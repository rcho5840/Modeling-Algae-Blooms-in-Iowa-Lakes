from sklearn.preprocessing import StandardScaler

def scaling(data, cols):
    scaler = StandardScaler()
    data[cols] = scaler.fit_transform(data[cols])
    return data