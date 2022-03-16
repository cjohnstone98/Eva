# Import Libraries
import os

import pandas as pd
import torch as pytorch
import numpy as np
import requests as req
import polygon as poly
import datetime
import time
import json
import xlsxwriter as xls
import math as m


# Code to setup the Polygon.io API
# Polygon.io login information (if it's ever needed)

# The API Request Handler
class ApiRequest:
    def __init__(self):
        self.username = 'cameronjohnstone4@gmail.com'
        self.password = 'PG@cMr19L@rT'
        self.apikey = '2btYLJqHWCNF2pu0O8J6AfOovDB_NWSA'
        self.apiurl = 'https://api.polygon.io/'

    def create_stockcluster_client(self):
        # creates a websocket client for the Polygon.io Stocks Cluster
        # Usage: get real time data from any stock by subscribing to it
        client = poly.WebSocketClient(poly.STOCKS_CLUSTER, self.apikey)
        # example code:
        # client.run_async()
        # client.subscribe("T.MSFT", "T.AAPL", "T.AMD", "T.NVDA")
        # client.close_connection()
        return client

    def create_cryptocluster_client(self):
        # creates a websocket client for the Polygon.io Crpyto Cluster
        # Usage: get real time data from any coin by subscribing to it
        client = poly.WebSocketClient(poly.CRYPTO_CLUSTER, self.apikey)
        return client

    def get_all_stock_tickers(self):
        # create a list of stock tickers
        api_callkey = "v3/reference/tickers?"
        api_marketparam = "market=stocks"
        api_activeparam = "active=true"
        api_sortparam = "sort=ticker"
        api_orderparam = "order=asc"
        api_limitparam = "limit=10000"
        api_keyparam = "apiKey=" + self.apikey
        api_get_request = self.apiurl + api_callkey + api_marketparam + "&" + api_activeparam + "&" + api_sortparam + "&" + api_orderparam + "&" + api_limitparam + "&" + api_keyparam
        get_all_stock_ticker_api_request = req.get(api_get_request)
        list_of_stock_tickers = []
        list_of_stock_names = []
        k = 0
        while k == 0:
            for i in list(get_all_stock_ticker_api_request.json().values())[0]:
                list_of_stock_tickers.append(i['ticker'])
                list_of_stock_names.append(i['name'])
                if k == 999:
                    get_all_stock_ticker_api_request = req.get(list(get_all_stock_ticker_api_request.json().values())[4] + "&" + api_keyparam)
                    k = -1
                k += 1
        else:
            pd.DataFrame({'Name': [list_of_stock_names], 'Ticker': [list_of_stock_tickers]}).to_csv('stocktickerdatabase.csv')

    def get_ticker_exchangedata(self, ticker, timespanmultiplier, timespan, starttime, endtime):
        print("Method Called: Get Ticker Exchange Info")
        # get information about a current stock
        api_callkey = "v2/aggs/"
        api_tickerparam = f"ticker/{ticker}/range/"
        api_multiplierparam = f"{timespanmultiplier}/"  # 30
        api_timespanparam = f"{timespan}/"  # minute
        api_startdateparam = f"{starttime}/"  # 2021-01-08
        api_enddateparam = f"{endtime}?"  # 2022-01-08
        api_resultslimitparam = "limit=50000&"
        api_keyparam = "apiKey=" + self.apikey
        api_get_request = self.apiurl + api_callkey + api_tickerparam + api_multiplierparam + api_timespanparam + api_startdateparam + api_enddateparam + api_resultslimitparam + api_keyparam

        # create blank arrays for pandas dataframe - based on available Polygon.io information
        list_of_stock_closeprice = []
        list_of_stock_lowestprice = []
        list_of_stock_numberoftransactions = []
        list_of_stock_volumes = []
        list_of_stock_weightedvolume = []
        list_of_stock_openprice = []
        list_of_stock_highestprice = []

        try:
            get_live_stock_info_results = req.get(api_get_request).json()

            # Pull information out of json object and put in nice horizontal arrays for .csv
            for i in list(get_live_stock_info_results['results']):
                list_of_stock_volumes.append(i['v'])
                list_of_stock_openprice.append(i['o'])
                list_of_stock_highestprice.append(i['h'])
                if i['v'] == 0:
                    list_of_stock_weightedvolume.append(0)
                    list_of_stock_numberoftransactions.append(0)
                else:
                    list_of_stock_weightedvolume.append(i['vw'])
                    list_of_stock_numberoftransactions.append(i['n'])
                list_of_stock_lowestprice.append(i['l'])
                list_of_stock_closeprice.append(i['c'])

            # put relevant results in a Pandas DataFrame for easy access
            ticker_dataframe = pd.DataFrame({'Open Price': list_of_stock_openprice, 'Close Price': list_of_stock_closeprice, 'Highest Price': list_of_stock_highestprice,
                                             'Lowest Price': list_of_stock_lowestprice, 'Volume': list_of_stock_volumes, 'Weighted Volume': list_of_stock_weightedvolume,
                                             'Num. Transactions': list_of_stock_numberoftransactions})
            ticker_info_array = [ticker, ticker_dataframe]
            return ticker_info_array
        except KeyError == 'results':
            print(f"It appears the ticker symbol: {ticker}, does not exist in the database. No results could be fetched from Polygon.io")

    def check_market_status(self):
        print("Method Called: Check Market Status")
        # Check if the market is open/closed
        get_marketstatus_api_request = req.get("https://api.polygon.io/v1/marketstatus/now?apiKey=2btYLJqHWCNF2pu0O8J6AfOovDB_NWSA").json()
        in_earlyhours = get_marketstatus_api_request["afterHours"]

        # or just set by the time of day to avoid so many api calls

    def get_ticker_financialdata(self, ticker, trainingdata_startdate, trainingdata_enddate, timebetweendata):
        print("Method Called: Get Ticker Financial Data")
        # API information
        api_callkey = "vX/reference/financials?"
        api_tickerparam = f"ticker={ticker}&"
        filingstartdate = datetime.datetime.fromisoformat(trainingdata_startdate) - datetime.timedelta(days=91)  # should yield five quarters of information
        filingstartdatestring = f"{filingstartdate.year}-{filingstartdate.month}-{filingstartdate.day}"
        api_filingstartdateparam = f"filing_date.gte={filingstartdatestring}&"
        filingenddate = datetime.datetime.fromisoformat(trainingdata_enddate)
        filingenddatestring = f"{filingenddate.year}-{filingenddate.month}-{filingenddate.day}"
        api_filingenddateparam = f"filing_date.lt{filingenddatestring}&"
        api_limitparam = "limit=5&"
        api_sortparam = "sort=filing_date&"
        api_keyparam = "apiKey=" + self.apikey
        api_get_request_quarterly = self.apiurl + api_callkey + api_tickerparam + api_filingstartdateparam + api_filingenddateparam + "timeframe=quarterly&" \
                                    + api_limitparam + api_sortparam + api_keyparam
        api_get_request_annual = self.apiurl + api_callkey + api_tickerparam + api_filingstartdateparam + api_filingenddateparam + "timeframe=annual&" \
                                    + api_limitparam + api_sortparam + api_keyparam

        gettickerfinancialdata_quarterly_results = req.get(api_get_request_quarterly).json()
        gettickerfinancialdata_annual_results = req.get(api_get_request_annual).json()

        # Get all results and sort them chronologically
        total_financialdata_array = []
        # i = 0
        for result in gettickerfinancialdata_quarterly_results['results']:
            total_financialdata_array.append(result)
            # i += 1
        # i = 0
        for result in gettickerfinancialdata_annual_results['results']:
            total_financialdata_array.append(result)
            # i += 1
        filingstartdatearray = []
        for result in total_financialdata_array:
            filingstartdatearray.append(result['end_date'])
        filingstartdatearray.sort()
        sortedfinancialdata = []
        for date in filingstartdatearray:
            k = 0
            for item in total_financialdata_array:
                if date == item['end_date']:
                    sortedfinancialdata.append(item)
                k += 1

        # determine how many times the information should be repeated in the Pandas DataFrames so that the values line up with the relevant times
        # this is imperfect because not all data exists always, so the AI will be trained to match data that doesn't quite fit reality
        qa_timemultiplier = (datetime.datetime.fromisoformat(sortedfinancialdata[0]['end_date']) - filingstartdate).days
                            # this assumes 24 hours of data is available, double check
        qb_timemultiplier = (datetime.datetime.fromisoformat(sortedfinancialdata[1]['end_date']) - datetime.datetime.fromisoformat(sortedfinancialdata[0]['end_date'])).days
        qc_timemultiplier = (datetime.datetime.fromisoformat(sortedfinancialdata[2]['end_date']) - datetime.datetime.fromisoformat(sortedfinancialdata[1]['end_date'])).days
        qd_timemultiplier = (datetime.datetime.fromisoformat(sortedfinancialdata[3]['end_date']) - datetime.datetime.fromisoformat(sortedfinancialdata[2]['end_date'])).days
        qe_timemultiplier = (filingenddate - datetime.datetime.fromisoformat(sortedfinancialdata[3]['end_date'])).days
        multipliers_array = np.array(np.multiply([qa_timemultiplier, qb_timemultiplier, qc_timemultiplier, qd_timemultiplier, qe_timemultiplier], (1440*(9/24)) / int(timebetweendata)), dtype=np.int64)

        # Create blank arrays to store data for the most recent quarter
        asset_values = ['Asset Value', []]
        equity_values = ['Equity Value', []]
        liability_values = ['Liabilities Value', []]
        exchange_gains_losses = ['Exchange G/L Value', []]
        net_cash_flows = ['NCF Value', []]
        net_cash_flows_from_financing_activities = ['NCFfFA Value', []]
        comprehensive_incomes_loss_to_parent = ['CILtP', []]
        comprehensive_incomes_loss = ['CIL Value', []]
        other_comprehensive_incomes_loss = ['OCIL Value', []]
        basic_earnings_per_share = ['EPS Value', []]
        cost_of_revenues = ['Cost of Revenues Value', []]
        gross_profits = ['GP Value', []]
        operating_expenses = ['Operating Expenses Value', []]
        revenues = ['Revenues Value', []]
        listofdata = [asset_values, equity_values, liability_values, exchange_gains_losses, net_cash_flows, net_cash_flows_from_financing_activities, comprehensive_incomes_loss_to_parent,
                      comprehensive_incomes_loss, other_comprehensive_incomes_loss, basic_earnings_per_share, cost_of_revenues, gross_profits, operating_expenses, revenues]

        # Create blank arrays to store the difference in data between the previous quarter and the most current
        asset_delta = ['Assets Delta', []]
        equity_delta = ['Equities Delta', []]
        liabilities_delta = ['Liabilities Delta', []]
        exchange_gains_losses_delta = ['Exchange G/L Delta', []]
        net_cash_flows_delta = ['NCF Delta', []]
        net_cash_flows_from_financing_activities_delta = ['NCFfFA Delta', []]
        comprehensive_incomes_loss_delta = ['CIL Delta', []]
        comprehensive_incomes_loss_to_parent_delta = ['CILtP Delta', []]
        other_comprehensive_incomes_loss_delta = ['OCIL Delta', []]
        basic_earnings_per_share_delta = ['EPS Delta', []]
        cost_of_revenues_delta = ['Cost of Revenues Delta', []]
        gross_profits_delta = ['GP Delta', []]
        operating_expenses_delta = ['Operating Expenses Delta', []]
        revenues_delta = ['Revenues Delta', []]
        listofdatadeltas = [asset_delta, equity_delta, liabilities_delta, exchange_gains_losses_delta, net_cash_flows_delta, net_cash_flows_from_financing_activities_delta,
                            comprehensive_incomes_loss_delta, comprehensive_incomes_loss_to_parent_delta, other_comprehensive_incomes_loss_delta, basic_earnings_per_share_delta,
                            cost_of_revenues_delta, gross_profits_delta, operating_expenses_delta, revenues_delta]

        # Financial Data
        i = 0
        # Pre-setting error flags to "False" to assign them at a higher level
        balance_sheet_error = False
        cash_flow_statement_error = False
        for quarter in sortedfinancialdata:
            k = 0
            if quarter == []:
                print(f"It appears that Polygon.io does not support financial information for: {ticker}. No data was accessed.")
            else:
                # assigning values to their respective arrays and repeating it based on the multiplier_array
                while k < multipliers_array[i]:
                    if len(quarter["financials"]["balance_sheet"]) != 0:
                        asset_values[1].append(quarter["financials"]["balance_sheet"]["assets"]["value"])
                        equity_values[1].append(quarter["financials"]["balance_sheet"]["equity"]["value"])
                        liability_values[1].append(quarter["financials"]["balance_sheet"]["liabilities"]["value"])
                    else:
                        balance_sheet_error = True
                    if len(quarter["financials"]["cash_flow_statement"]) != 0:
                        # (CAMERON) Error here, need to figure out how to remove a column from the pandas dataframe if it does not show up here
                        exchange_gains_losses[1].append(quarter["financials"]["cash_flow_statement"]["exchange_gains_losses"]["value"])
                        net_cash_flows[1].append(quarter["financials"]["cash_flow_statement"]["net_cash_flow"]["value"])
                        net_cash_flows_from_financing_activities[1].append(quarter["financials"]["cash_flow_statement"]["net_cash_flow_from_financing_activities"]["value"])
                    else:
                        cash_flow_statement_error = True
                    comprehensive_incomes_loss[1].append(quarter["financials"]["comprehensive_income"]["comprehensive_income_loss"]["value"])
                    comprehensive_incomes_loss_to_parent[1].append(quarter["financials"]["comprehensive_income"]["comprehensive_income_loss_attributable_to_parent"]["value"])
                    other_comprehensive_incomes_loss[1].append(quarter["financials"]["comprehensive_income"]["other_comprehensive_income_loss"]["value"])
                    basic_earnings_per_share[1].append(quarter["financials"]["income_statement"]["basic_earnings_per_share"]["value"])
                    cost_of_revenues[1].append(quarter["financials"]["income_statement"]["cost_of_revenue"]["value"])
                    gross_profits[1].append(quarter["financials"]["income_statement"]["gross_profit"]["value"])
                    operating_expenses[1].append(quarter["financials"]["income_statement"]["operating_expenses"]["value"])
                    revenues[1].append(quarter["financials"]["income_statement"]["revenues"]["value"])
                    k += 1
                # Print warnings for any missing training data columns
                if balance_sheet_error:
                    print(f'WARNING: Training data columns: {asset_values[0]}, {equity_values[0]}, {liability_values[0]}; were not found.')
                if cash_flow_statement_error:
                    print(f'WARNING: Training data columns: {exchange_gains_losses[0]}, {net_cash_flows[0]}, {net_cash_flows_from_financing_activities[0]}; were not found.')
        formatted_limited_financialdata = np.array([asset_values, equity_values, liability_values, exchange_gains_losses, net_cash_flows, net_cash_flows_from_financing_activities,
                                           comprehensive_incomes_loss, comprehensive_incomes_loss_to_parent, other_comprehensive_incomes_loss, basic_earnings_per_share,
                                           cost_of_revenues, gross_profits, operating_expenses, revenues])
        formatted_delta_financialdata = np.array([asset_delta, equity_delta, liabilities_delta, exchange_gains_losses_delta, net_cash_flows_delta, net_cash_flows_from_financing_activities_delta,
                                     comprehensive_incomes_loss_delta, comprehensive_incomes_loss_to_parent_delta, other_comprehensive_incomes_loss_delta, basic_earnings_per_share_delta,
                                     cost_of_revenues_delta, gross_profits_delta, operating_expenses_delta, revenues_delta])

        # a counter for each individual data point in formatted_limited_financialdata
        i = 0
        # a blank array to hold the data indices for subtracting numbers from formatted_limited_financialdata as i increases through the list of financial data
        index = [0, 0]
        while i < len(asset_values):
            # Set the indices for the two values to be subtracted from formatted_limited_financialdata based on the multiplier_array values that separate the quarters
            if i < multipliers_array[0]:
                index[0] = 0
                index[1] = multipliers_array[0]
            elif multipliers_array[0] <= i < multipliers_array[1]:
                index[0] = multipliers_array[0]
                index[1] = multipliers_array[1]
            elif multipliers_array[1] <= i < multipliers_array[2]:
                index[0] = multipliers_array[1]
                index[1] = multipliers_array[2]
            else:
                index[0] = multipliers_array[2]
                index[1] = multipliers_array[3]

            k = 0   # a counter to move through each item in formatted_x_financialdata arrays
            for item in formatted_delta_financialdata:
                # add the difference between the two data points, separated in index by the quarter multiplier_array value, from the formatted_limited_financialdata array to the
                # array of deltas
                item[1].append(formatted_limited_financialdata[k][1][index[0]]-formatted_limited_financialdata[k][1][index[1]])
                k += 1
            i += 1

        # delete all the QE data from formatted_limited_financialdata and join the two datasets
        formatted_financialdata = formatted_limited_financialdata[1:(multipliers_array[0] + multipliers_array[1] + multipliers_array[2])]

        # create the pandas data array to return
        dataframe = pd.DataFrame(data=formatted_financialdata)
        return dataframe

    # def get_ticker_news(self, stockticker, publishdate = datetime.date.today(), apikey = "https://api.polygon.io/"):
    #     api_callkey = "v2/reference/news?"
    #     api_tickerparam = "ticker=" + stockticker
    #     api_publisheddateparam = 'published_utc.gte=' + publishdate
    #     api_get_request = apikey + api_callkey + api_tickerparam + '&' + api_publisheddateparam + '&'
    #     results = req.get(api_get_request).json
    #     # convert JSON object from API call to python dictionary
    #     # article_keywords = results[]
    #     # parse and convert python dictionary information to useful format for pytorch
    #     return


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
