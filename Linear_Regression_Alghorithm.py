import numpy as np

class LinearRegression:

    def __init__(self,learning_rate,epochs):
        self.alpha = learning_rate
        self.epochs = epochs
        self.w = None
        self.b = None


    def fit(self,x,y):

        n_samples,n_features = x.shape

        self.w = np.zeros(n_features)
        self.b = 0

        for epochs in range(self.epochs):

            y_pred = np.dot(x,self.w) + self.b

            error = y_pred - y

            dw = (2 / n_samples) * np.dot(x.T,error)
            db = (2 / n_samples) * np.sum(error)

            self.w -= self.alpha * dw
            self.b -= self.alpha * db

    def prediction(self,x):

        return np.dot(x,self.w) + self.b

class StandardScaler:

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self,x):

        self.mean = np.mean(x,axis=0)
        self.std = np.std(x,axis=0)


    def transform(self,x):

        return (x - self.mean) / self.std

    def fit_transform(self,x):
        self.fit(x)
        return self.transform(x)

X_train = np.array([[2104, 5, 3],
    [1600, 3, 3],
    [2400, 15, 3],
    [1416, 20, 2],
    [3000, 10, 4],
    [1985, 4, 4],
    [1534, 12, 3],
    [852, 35, 2],
    [1200, 15, 3]])

Y_train = np.array([399900, 329900, 369000, 232000, 539900, 299900, 314900, 178000, 240000])

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
model = LinearRegression(0.01,5000)
model.fit(X_train_scaled,Y_train)

new_house = np.array([[1000,15,2]])
new_house_scaled = scaler.transform(new_house)
prediction = model.prediction(new_house_scaled)
print(prediction)