import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as dates
import yfinance as yf
import os
from datetime import datetime
from dateutil.relativedelta import relativedelta
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn import metrics


def download_stock_data(ticker, years):
    """
    downloads the stock_data of the company ${ticker} and saves as a csv-file in a "cleaned" version
    :param ticker: ticker of the searched company name
    :return: filename of the downloaded stock data
    """

    # sanity check
    if ticker is None or years is None or years < 0:
        print("pass reasonable arguments: not None and not negative years!")
        return

    # calculate start- and enddate
    dateformat = '%Y-%m-%d'
    endDate = datetime.today().strftime(dateformat)
    startDate = (datetime.now() - relativedelta(years=years)).strftime(dateformat)

    # download stockdata using yfinance
    resultData = yf.download(ticker, startDate, endDate)
    print(resultData.head())
    print(resultData.info())

    # preprocess data by adding columns
    resultData["Open-Close"] = resultData["Open"] - resultData["Close"]
    resultData["Day"] = resultData.index.day
    resultData["Month"] = resultData.index.month
    resultData["Year"] = resultData.index.year
    resultData["Is_quarter_end"] = np.where((resultData["Month"]) % 3 == 0, 1, 0)
    resultData["Low-High"] = resultData["Low"] - resultData["High"]
    # "Comparison_1_day" describing whether the stock_price of nth day was lower than the next day
    resultData["Comparison_1_day"] = np.where((resultData["Close"].shift(-1) > resultData["Close"])
                                              | (resultData["Open"].shift(-1) > resultData["Open"]), 1, 0)

    # add average column as average of 'Low' and 'High'
    resultData["Average"] = (resultData["Low"] + resultData["High"]) / 2.0

    # drop unnecessary columns
    columns_to_drop = ["Adj Close", "Volume"]
    resultData.drop(columns_to_drop, axis="columns", inplace=True)

    # save the stockdata as a csv
    pwd = os.getcwd()
    fileName = '_'.join(["stock_data", ticker, startDate, endDate + ".csv"])
    path = os.path.join(pwd, "stock_data_csv", fileName)
    print(f"path: {path}")
    resultData.to_csv(path)

    return fileName


def plot_stock_data(fileName):
    """
    plots the stock data using matplotlib and save the plot as png file into the directory "stock_data_plots"
    :param fileName: fileName of the csv file of the wanted stock data
    :return: None
    """
    pwd = os.getcwd()
    path = os.path.join(pwd, "stock_data_csv", fileName)
    stock_data = pd.read_csv(path)
    fileName = os.path.split(path)[1]
    Ticker = fileName.split('_')[2]
    startDate = fileName.split('_')[3]
    endDate = fileName.split('_')[4]

    # sanity check
    if not isinstance(stock_data, pd.DataFrame):
        print(f"the given argument {stock_data} is not of type pd.DataFrame!")

    # plot
    dates = pd.to_datetime(stock_data["Date"])
    plt.plot(dates, stock_data["Average"], label="Average")
    plt.plot(dates, stock_data["Open"], label="Low")
    plt.plot(dates, stock_data["Close"], label="High")
    plt.gcf().autofmt_xdate()
    plt.title(f"Stock_data of {Ticker} \n from {startDate} to {endDate}")
    plt.legend()

    # Save the plot as png
    format = ".png"
    save_path = os.path.join(pwd, "stock_data_plots", fileName + format)
    plt.savefig(save_path)
    print(f"saved file: {save_path}")


def split_and_normalize_data(stock_data, valid_size):
    """
    noralize data and split data into train- and valid-data
    target as whether one could have made profit by buying the stock and selling it on the next day
    :param stock_data:
    :return: X_train, X_valid, Y_train, Y_valid
    """

    features = stock_data[["Day", "Month", "Year", "Open-Close", "Is_quarter_end", "Low-High"]]
    target = stock_data["Comparison_1_day"]

    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    X_train, X_valid, Y_train, Y_valid = train_test_split(
        features, target, test_size=valid_size, random_state=2023)
    print(f"splitted data using valid_size={valid_size}")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_valid shape: {X_valid.shape}")

    return X_train, X_valid, Y_train, Y_valid


def develop_models_and_evaluate(X_train, X_valid, Y_train, Y_valid):
    """
    develop different models (Logistic Regression, Support Vector Machine, XGBClassifier) and evaluate them using the valid_data
    :return: models
    """
    models = [LogisticRegression(), SVC(kernel='poly', probability=True), XGBClassifier()]

    print()
    print("Calcuating accuracies of trained models")
    for model in models:
        model.fit(X_train, Y_train)
        print(f'{model} : ')
        print('Training Accuracy : ', metrics.roc_auc_score(
            Y_train, model.predict_proba(X_train)[:, 1]))
        print('Validation Accuracy : ', metrics.roc_auc_score(
            Y_valid, model.predict_proba(X_valid)[:, 1]))
        print()

    return models


if __name__ == '__main__':
    # test
    fileName = download_stock_data("GOOGL", 10)
    plot_stock_data(fileName)
    stock_data = pd.read_csv(f"stock_data_csv/{fileName}")
    X_train, X_valid, Y_train, Y_valid = split_and_normalize_data(stock_data, 0.1)
    logistic_regression, SVC, XGBClassifier = develop_models_and_evaluate(X_train, X_valid, Y_train, Y_valid)
