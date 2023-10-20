import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import os
from datetime import datetime
from dateutil.relativedelta import relativedelta

def download_stock_data(ticker, years):
    """
    downloads the stock_data of the company ${ticker} and saves as a csv-file in a "cleaned" version
    :param ticker: ticker of the searched company name
    :return: filename of the downloaded stock data
    """

    # sanity check
    if ticker is None or years is None or years < 0 :
        print("pass reasonable arguments: not None and not negative years!")
        return

    # calculate start- and enddate
    dateformat = '%Y-%m-%d'
    endDate = datetime.today().strftime(dateformat)
    startDate = (datetime.now() - relativedelta(years=years)).strftime(dateformat)

    # download stockdata using yfinance
    resultData = yf.download(ticker, startDate, endDate)

    # add average column as average of 'Low' and 'High'
    resultData["Average"] = (resultData["Low"] + resultData["High"]) / 2.0

    # drop unnecessary columns
    columns_to_drop = ["Open",  "Close", "Adj Close", "Volume"]
    resultData.drop(columns_to_drop, axis="columns", inplace=True)

    # save the stockdata as a csv
    pwd = os.getcwd()
    fileName = '_'.join(["stock_data", ticker, startDate, endDate + ".csv"])
    path = os.path.join(pwd, "stock_data_csv", fileName)
    print(f"path: {path}")
    resultData.to_csv(path)

    print(resultData.head())

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
    print(f"fileName: {fileName}")

    # sanity check
    if not isinstance(stock_data, pd.DataFrame):
        print(f"the given argument {stock_data} is not of type pd.DataFrame!")

    X = stock_data["Date"]
    plt.plot(X, stock_data["Average"], label = "Average")
    plt.plot(X, stock_data["Low"], label = "Low")
    plt.plot(X, stock_data["High"], label = "High")
    plt.title(f"Stock_data of ")
    plt.legend()

    # Save the plot as png
    format = ".png"
    save_path = os.path.join(pwd, "stock_data_plots", fileName + format)
    plt.savefig(save_path)

    return


if __name__ == '__main__':
    # test
    fileName = download_stock_data("GOOGL", 1)
    plot_stock_data(fileName)
