#From https://www.makeuseof.com/stock-price-data-using-python/ and https://www.w3schools.com/python/python_ml_getting_started.asp
import yfinance as yf
import datetime
import matplotlib.pyplot as plt
from time import sleep
from scipy import stats
import numpy
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import pandas

print("Starting...")

start = "2010-01-01"
end = datetime.datetime.today().strftime("%Y-%m-%d")
tickers = [
    "AAPL", "MSFT", "TSLA", "AYI", "JPM", "XOM", "CHV", "FXAIX", "FSPTX", "FZROX", "FNCMX", "GOOG", "AMZN", "NVDA", "META", "UNH"
]

print("Downloading...")

dataArray = []
for ticker in tickers:
    newData = yf.download(ticker, start, end, interval="1d")[["Close"]].to_numpy()
    for i in range(0, newData.size):
        dataArray.append(newData[i][0])

print("Download complete!")

data = pandas.DataFrame({ "Close": dataArray })
print(data)

# data = data.reset_index() #Date was the index, now it's a column
# data["Date"] = data["Date"].apply(lambda x: x.value)

historyLength = 30
trainingCols = []
for i in range(historyLength):
    data.insert(i+1, i+1, data.index, True)
    trainingCols.append(i+1)

print("Formatting data...")
for index, row in data.iterrows():
    if(index <= historyLength):
        continue
    else:
        for i in range(historyLength):
            data.iat[index, i+1] = data.iat[index - i, 0]

data = data.tail(len(data) - historyLength - 1)

print("Formatted data")
print(data)


testingSize = data.index.size / 5
testingSize = int(testingSize)

training = data.head(data.size - testingSize)
testing = data.tail(testingSize)

training.set_index(1, inplace=True)
training.sort_index(inplace=True)

testing.set_index(1, inplace=True)
testing.sort_index(inplace=True)

trainingCols.pop(0)

x = training[trainingCols]
y = training["Close"]

print("Generating model...")
model = linear_model.LinearRegression()
model.fit(x, y)
print("Model generated!")

def predict(x):
    return model.predict(x)

print(model.coef_)

predicted = predict(testing[trainingCols])
residuals = testing["Close"] - predicted

print("R^2: " + str(r2_score(testing["Close"], predicted)))

plt.figure(figsize=(15, 10))

plt.xlabel("Price")
plt.ylabel("$")

plt.plot(x.index, predict(x), color="red")
plt.plot(x.index, predict(x)-training["Close"], color="blue")

plt.plot(testing.index, predicted, color="orange")
plt.plot(testing.index, residuals, color="purple")

plt.plot(x.index, y, color="green")
plt.plot(testing.index, testing["Close"], color="aqua")

plt.show()