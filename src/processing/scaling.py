from sklearn.preprocessing import StandardScaler

def scaling(data, cols):
    scaler = StandardScaler()
    scaler.fit(data[cols])
    data[cols] = scaler.transform(data[cols])
    return data