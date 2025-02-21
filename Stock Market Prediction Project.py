#!/usr/bin/env python
# coding: utf-8

# In[263]:


import pandas as pd


# In[264]:


data = pd.read_csv('stocks.csv')


# In[265]:


print(data.head())


# In[266]:


data=data.dropna()  #Handling missing values


# In[436]:


#Correlation Heatmap – Feature Relationships
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.heatmap(data[["Close", "Volume", "Open", "High", "Low"]].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()


# In[267]:


data.plot.line(y="Close", use_index=True)


# In[268]:


data.info() #checking if the delete action is performed, as well as the other column's information


# In[435]:


#Volume vs. Price Movement
fig, ax1 = plt.subplots(figsize=(12,6))

ax1.set_xlabel("Date")
ax1.set_ylabel("Closing Price", color="blue")
ax1.plot(data["Close"], label="Closing Price", color="blue")
ax1.tick_params(axis="y", labelcolor="blue")

ax2 = ax1.twinx()
ax2.set_ylabel("Volume", color="green")
ax2.bar(data.index, data["Volume"], color="green", alpha=0.3)
ax2.tick_params(axis="y", labelcolor="green")

plt.title("Stock Price vs. Trading Volume")
plt.show()
#High volume with price movement = strong trend confirmation.


# In[269]:


data["Tomorrow"]= data["Close"].shift(-1) #Creating a new columnn "Tomorrow" which holds the next day's closing stock price


# In[270]:


data


# In[271]:


data["Target"]=(data["Tomorrow"]>data["Close"]).astype(int)


# In[272]:


data


# In[273]:


data.shape


# In[344]:


#Training an initial ML model

from sklearn.ensemble import RandomForestClassifier #choosing RandomForestClassifier due to it's accuracy, avoids overfitting better than others, and can pick non-linear tendencies in the data.
model= RandomForestClassifier(n_estimators=185, min_samples_split=100, random_state=1, class_weight="balanced") #creating the model
#"n_estimators" are the no' of individual decision trees we want to train - higher they are, higher the accuracy is. "min_samples_split" this will help us to protect from overfitting. "random_state=1" will help us to get same results all the time we run the model, or model's results will be predictible.

train= data.iloc[ :-100]
test= data.iloc[-100: ]
predictors= ["Open","High","Low","Close","Volume"]
model.fit(train[predictors], train["Target"])


# In[345]:


#Measuring the aaccuracy of the model
from sklearn.metrics import precision_score
preds= model.predict(test[predictors]) #preds is the prediction score


# In[346]:


preds= pd.Series(preds, index=test.index)


# In[347]:


print(set(preds))


# In[348]:


precision_score(test["Target"], preds)


# In[349]:


#plotting the predictions
combined= pd.concat([test["Target"], preds], axis=1)


# In[350]:


combined.plot()


# In[351]:


def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds= model.predict(test[predictors])
    preds= pd.Series(preds, index=test.index, name="Predictions")
    combined=pd.concat([test["Target"], preds], axis=1)
    return combined


# In[438]:


#Rolling Volatility – Risk Measurement
data["Volatility"] = data["Close"].pct_change().rolling(window=30).std()
plt.figure(figsize=(12, 6))
plt.plot(data["Volatility"], label="30-Day Rolling Volatility", color="purple")
plt.xlabel("Date")
plt.ylabel("Volatility")
plt.title("Stock Rolling Volatility Over Time")
plt.legend()
plt.show()


# In[372]:


#Building a backtesting system

def backtest(data1, model, predictors, start=50, step=10): #Testing per year, by taking 10 years of data
    all_predictions= [] #Creating a list in which each data frame stores the predicted value, for a year
    for i in range(start, data1.shape[0], step):
        train= data1.iloc[:i].copy() #Contains all years prior to the current year
        test= data1.iloc[i:(i+step)].copy() #Contains the current year
        predictions= predict(train, test, predictors, model)
        if predictions is not None and not predictions.empty:
            all_predictions.append(predictions)
    if len(all_predictions) == 0:
        raise ValueError("No predictions generated. Check model training and prediction functions.")
    return pd.concat(all_predictions,ignore_index=True) #Combines all data frames into a single data frame


# In[373]:


predictions= backtest(data, model, predictors)


# In[374]:


predictions["Predictions"].value_counts() #counts the no' of predicted times, for each type of value


# In[375]:


data["Target"].value_counts()


# In[376]:


precision_score(predictions["Target"], predictions["Predictions"])


# In[377]:


predictions["Target"].value_counts() / predictions.shape[0]


# In[441]:


import plotly.express as px

fig = px.line(data, x=data.index, y="Close", title="Interactive Stock Price Chart")
fig.show()


# In[398]:


#Adding additional predictors to our model, to improve accuracy 
horizons= [2,5,30,60] #horizons are the rolling means, i.e we will calculate mean of close price in the last 2 days, 5, 30, &60 days, and will find the ratio between today's closing price and closing price in those periods. Performing this to improve predictions
new_predictors= []
data = data.select_dtypes(include=["number"])
for horizon in horizons:
    rolling_averages= data.rolling(horizon).mean()
    ratio_column= f"Close_Ratio_{horizon}" #creating new column
    data[ratio_column]= data["Close"] / rolling_averages["Close"]
    trend_column= f"Trend_{horizon}" #This column holds the no' of days in the past X days(i.e days in horizon) that the stock price went up
    data[trend_column]= data.shift(1).rolling(horizon).sum()["Target"] #This will calculate the sum of the 1's in target, i.e the trend when stock went up and has predicted corectly
    new_predictors+= [ratio_column, trend_column]


# In[399]:


data


# In[400]:


#Improving the model

model= RandomForestClassifier(n_estimators=100, min_samples_split=50, random_state=1, )


# In[401]:


data


# In[402]:


import numpy as np
def predict(train, test, predictors, model):
    train = train.copy()
    test = test.copy()
    # Fill missing values with column means
    train[predictors] = train[predictors].apply(lambda x: x.fillna(x.mean()))
    test[predictors] = test[predictors].apply(lambda x: x.fillna(x.mean()))

    model.fit(train[predictors], train["Target"])
    preds= model.predict_proba(test[predictors])[:,1] #This returns the probability of the stock price that goes up or down, and selecting the 2nd column which will give the probability that stock price goes up
    # Apply custom threshold (0.6)
    preds = np.where(preds >= 0.6, 1, 0)  # 1 if >= 0.6, else 0
    #preds[preds >= .6] = 1  #Setting our custom threshold, by default it is "0.5".so if it greater than .6 there is a chance that price will go up.
    #pred[ preds < .6] = 0
    return pd.DataFrame({"Predictions": preds}, index=test.index)
    #preds= pd.Series(preds, index=tesst.index, name="Predictions")
    #combined=pd.concat([test["Target"], preds], axis=1)
    #return combined


# In[403]:


print("Missing values in train set:\n", train[predictors].isna().sum())
print("Missing values in test set:\n", test[predictors].isna().sum())


# In[404]:


train.shape


# In[406]:


test.shape


# In[412]:


pd.Series(new_predictors).value_counts()


# In[405]:


data


# In[428]:


predictions["Predictions"].value_counts()


# In[430]:


import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(data["Close"], label="Closing Price", color="blue")
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.title("Stock Price Trend Over Time")
plt.legend()
plt.show()


# In[431]:


data["50_MA"] = data["Close"].rolling(window=50).mean()  # 50-day moving average
data["200_MA"] = data["Close"].rolling(window=200).mean()

plt.figure(figsize=(12, 6))
plt.plot(data["Close"], label="Closing Price", color="blue")
plt.plot(data["50_MA"], label="50-Day MA", color="orange")
plt.plot(data["200_MA"], label="200-Day MA", color="red")
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.title("Stock Price with Moving Averages")
plt.legend()
plt.show()
#Identifies trends, bullish/bearish crossovers, and long-term patterns.


# In[ ]:




