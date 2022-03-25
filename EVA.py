# Import Libraries
import os

import pandas as pd
import torch as pytorch
import numpy as np
import requests as req
import datetime
import time
import json
import xlsxwriter as xls
import math as m


# The API Request Handler
class ApiRequest:
    def __init__(self):
        self.username = 'cameronjohnstone4@gmail.com'
        self.password = 'PG@cMr19L@rT'
        self.apikey = '2btYLJqHWCNF2pu0O8J6AfOovDB_NWSA'
        self.apiurl = 'https://api.polygon.io/'

# The Algorithm
class EvaAI:
    # Current Algorithm Goal: Use free historical data from Polygon with api call result objects to train a neural network to automate trades
    # Goal: create for loop using free api (limit 5 api calls per minute) to gather market information about S&P 500 stocks over last 2 years in 5 minutes interval windows.
    # Populate in .csv file
    # Steps:    1) create arrays of current market data from the ticker
    #           2) create arrays of results the regression will validate itself against in training
    #           3) put all data in a csv and all csvs in a folder
    #           4) give neural network ability to execute a buy or sell command
    #           5) create data set that tracks ficticious money based on when ai executes commands
    #           6) set reward for neural net to maximize return on investment over time
    #           7) alternative information for training: performance of similar companies using 'tickerdetails: similar', keywords from news articles from 'tickernews: keywords'
    #                   company market cap, company Price-to-earnings ratio, other ratios, information from SEC filings

    def __init__(self):
        self.api_object = ApiRequest()
        print("Eva Algorithm Activated")

    def create_training_data(self):
        print("Method Called: Create Training Data")
        # Description: Create training and validation dataset for the AI
        # Create a new directory for the latest training data
        timenow = datetime.datetime.now().strftime('%H-%M-%S')
        print(f"Current Time: {timenow}")
        datenow = datetime.date.today().strftime('%d-%m-%Y')
        print(f"Current Date: {datenow}")
        newdirectory = f'{datenow} {timenow} Training Data'
        os.mkdir(f'C:/Users/Cameron Johnstone/Desktop/Python Projects/InvestAI/Training Data/{newdirectory}')
        print(f"New Directory Created: C:/Users/Cameron Johnstone/Desktop/Python Projects/InvestAI/Training Data/{newdirectory}")

        # Read in stock tickers from S&P 500 and convert to pandas object
        sp_tickers = pd.read_csv('C:/Users/Cameron Johnstone/Desktop/Python Projects/InvestAI/sp_500_stocks.csv')

        # Set input parameters to get_ticker_exchange_info method
        timespan = 'minute'  # 'day', 'month'
        timemultiplier = '5'
        starttime = '2020-01-08'
        endtime = '2021-01-08'

        # iterate through all S&P500 stocks to gather historical data and store in excel
        print("Status: Iterating through list of tickers")
        for i in sp_tickers['Ticker']:
            print(f"Status: Gather financial data for: {i}")
            ticker_exchange_info = self.api_object.get_ticker_exchangedata(i, timemultiplier, timespan, starttime, endtime)
            ticker_financial_info = self.api_object.get_ticker_financialdata(i, starttime, endtime, timemultiplier)
            time.sleep(30)  # unnecessary when I get a paid profile, if I ever recreate this

            # add the new information to the existing data pandasDataFrame and store in a .csv file
            print(f"Status: Joining Exchange Data with SEC Financial Data")
            ticker_exchange_info[1].join(ticker_financial_info)
            ticker_exchange_info[1].to_csv(f'C:/Users/Cameron Johnstone/Desktop/Python Projects/InvestAI/Training Data/{newdirectory}/{ticker_exchange_info[0]}_trainingdata.csv')
            print(f'Status: CSV file created, {ticker_exchange_info[0]}_trainingdata.csv')

# The Main
def main():
    # instantiate objects of the necessary classes
    eva = EvaAI()
    eva.create_training_data()


if __name__ == '__main__':
    main()
