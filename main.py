#From https://www.makeuseof.com/stock-price-data-using-python/ and https://www.w3schools.com/python/python_ml_getting_started.asp
import yfinance as yf
import datetime
import matplotlib.pyplot as plt
from time import sleep
from scipy import stats
import numpy
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
import pandas

print("Starting...")

start = "2010-01-01"
end = datetime.datetime.today().strftime("%Y-%m-%d")
ticker = "AAPL"

print("Downloading...")
data = yf.download(ticker, start, end, interval="1d")
print("Download complete!")

data = data.reset_index() #Date was the index, now it's a column
data["Date"] = data["Date"].apply(lambda x: x.value)

testingSize = data.index.size / 5
testingSize = int(testingSize)

training = data.head(len(data) - testingSize)
testing = data.tail(testingSize)

x = training[['Date', 'Open', 'High', 'Low', 'Volume']]
testingCleaned = testing[['Date', 'Open', 'High', 'Low', 'Volume']]
scale = StandardScaler()
x = scale.fit_transform(x)
y = training["Close"]

print("Generating model...")
model = linear_model.LinearRegression()
model.fit(x, y)
print("Model generated!")

def predict(x, shouldScale):
    if(shouldScale):
        x = scale.transform(x)
    return model.predict(x)

print(model.coef_)

predicted = predict(testingCleaned, True)
residuals = testing["Close"] - predicted

plt.figure(figsize=(15, 10))
plt.xlabel("Date")
plt.ylabel("Price")

# plt.plot(training["Date"], training["Close"], color="green")
# plt.plot(training["Date"], predict(x, False), color="blue")

plt.plot(testing["Date"], testing["Close"], color="green")
plt.plot(testing["Date"], predicted, color="red")
plt.plot(testing["Date"], residuals, color="blue")

plt.show()