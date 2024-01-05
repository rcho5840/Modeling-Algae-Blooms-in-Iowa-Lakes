from sklearn.preprocessing import StandardScaler

def scaling(data, data2, cols):
    scaler = StandardScaler()
    data[cols] = scaler.fit_transform(data[cols])
    data2[cols] = scaler.transform(data2[cols])
    return data, data2