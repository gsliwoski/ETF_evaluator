import autogen
import sys
import os
import pandas as pd
import yfinance as yf
#from pandas_datareader import data
import wbgapi as wb
from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers import Html2TextTransformer
import numpy as np
from typing import Annotated
from datetime import datetime
import requests
from langchain_openai import ChatOpenAI
import argparse
import json

yf.pdr_override()

all_keys = json.load(open("api_keys.json"))
FRED_API = all_keys['FRED']
OPENAI_API_KEY = all_keys['OPENAI']
GOOGLE_API_KEY = all_keys['GOOGLE']
SEI = all_keys['SEI']
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
os.environ['OAI_CONFIG_LIST'] = f"""[{{"model": "gpt-4-turbo", "api_key": "{OPENAI_API_KEY}"}}]"""
config_list = autogen.config_list_from_json("OAI_CONFIG_LIST")

gpt4_config = {
    "cache_seed": 100,
    "temperature": 0.2,
    "config_list": config_list,
    "timeout": 120
}

financial_expert = autogen.AssistantAgent(
    name="Expert_Financial_Advisor",
    system_message="Expert_Finanacial_Advisor. You are a highly successful expert financial advisor with decades of experience. You keep up with all "
                   "market trends and are capable of successfully selecting the most attractive ETFs to buy at a given "
                   "moment in time considering a wide range of financial and market aspects and trends. "
                   "When evaluating an ETF, you consider all of the following current information in detail:\n"
                   "- historical performance\n"
                   "- expense ratio\n"
                   "- dividend yield\n"
                   "- fund holdings and allocation\n"
                   "- market capitalization\n"
                   "- volatility and risk\n"
                   "- liquidity\n"
                   "- performance relative to peers\n"
                   "- economic indicators\n"
                   "- analyst ratings and reports\n"
                   "- tax efficiency\n"
                   "- GDP growth rates\n"
                   "- interest rates\n"
                   "- inflation\n"
                   "- unemployment rates\n"
                   "- currency strength\n"
                   "- government policies\n"
                   "- sector performance\n"
                   "- market sentiment\n"
                   "- technological advancements\n"
                   "- demographic trends\n"
                   "- supply chain issues\n"
                   "- commodity prices\n"
                   "You may request web searching, stock price retrieval, html scraping from the executor. "
                   "All information you consider must be current and up-to-date. Do not base the analysis on generic information. "
                   "It is critical that you base your analysis on the information retrieved with tools and the Engineer. "
                   "For additional information, and more sophisticated look up tools, especially if web scraping did not deliver all of the necessary information "
                   "it is critical to ask the engineer to write code to look up specific information using api's and other tools in python. "
                   "Once you have every piece of information regarding all the different aspects and indicators, generate "
                   "a comprehensive report that outlines the information, with strong critiques considering the risk tolerance, "
                   "All conclusions and critiques should be accompanied with real data that informed them. "
                   "and a detailed conclusion about whether or not the ETF is currently an attractive investment or not. "
                   "Lastly, when the summary has been shared, print TERMINATE.",
    llm_config=gpt4_config)

engineer = autogen.AssistantAgent(
    name="Engineer",
    system_message="Engineer. You are responsible for generating python code for answering a question or fulfilling a request from the Expert_Financial_Advisor. "
                   "If your code produces an error, carefully review the code and the error and attempt to fix the code and rerun it. "
                   "If you can't perform the request after several tries, then simply return that the request is impossible. "
                   "You are particularly proficient with API's including the world bank api (and python library wbgapi), "
                   "the IMF (imfpy python library), the FRED API, and investor.com api (investpy python library).",
    llm_config=gpt4_config
)

user_proxy = autogen.UserProxyAgent(
    name="User_Proxy",
    system_message="A human admin that can execute code and useful tools such as web search, html scraping, and stock information lookup.",
    human_input_mode="NEVER",
    is_termination_msg=lambda x: "TERMINATE" in x.get("content", "").rstrip(),
    code_execution_config={
        "work_dir": "coding",
        "last_n_messages": 3,
        "use_docker": False
    }
)

search = GoogleSearchAPIWrapper(google_api_key=GOOGLE_API_KEY, google_cse_id=SEI, k=10)

@user_proxy.register_for_execution()
@financial_expert.register_for_llm(description="Searches Google and returns the top 10 result urls.")
def googleit(query: Annotated[str, "Google query"]) -> list:
    '''Good for searching google and getting the top 10 hits url. Takes a query to google and returns a list of links.'''
    return [x['link'] for x in search.results(query, num_results=10)]

@user_proxy.register_for_execution()
@financial_expert.register_for_llm(description="Scrapes a given url html and returns as text.")
def scraper(url: Annotated[str, "URL to retrieve HTML from to scrape"]) -> dict:
    '''Good for scraping the contents from a specific url and converting the html to text.
       it takes a url string and returns the html converted to text.'''
    loader = AsyncHtmlLoader([url], verify_ssl=False)
    docs = loader.load()
    html2text = Html2TextTransformer()
    docs_transformed = html2text.transform_documents(docs)
    docdict = {"page_content": docs_transformed[0].page_content,
               "metadata": docs_transformed[0].metadata}
    return docdict

@user_proxy.register_for_execution()
@financial_expert.register_for_llm(description="Get stock information for a given symbol including high, low, close, open, and volume.")
def fetch_prices_for_symbol(symbol: Annotated[str, "Stock symbol of interest"], days: Annotated[int, "How many days back to look, maximum possible days is 1095"]) -> pd.DataFrame:
    end_date = pd.Timestamp.now()
    start_date = end_date - pd.DateOffset(days=days)
    data = yf.download(symbol, start=start_date, end=end_date)
    data.reset_index(inplace=True)
    data = data[["Date", "Open", "High", "Low", "Close", "Volume"]]
    data.rename(
        columns={"Date": "date", "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"},
        inplace=True)
    return data.to_json(orient="records")

summarizer = ChatOpenAI(model="gpt-4-turbo", temperature=0.2, max_tokens=4096)

@user_proxy.register_for_execution()
@financial_expert.register_for_llm(description="Get economic and macroeconomic data for the USA.")
def fred_lookup(start_date: Annotated[str, "Start date of information as string in format 'YYYY-MM-DD' Must be '2023-01-01' or more recent"],
                requested_information: Annotated[list, "All types of requested information can include one or more of the following: "
                                                       "'employment and labor', 'market indicators', interest rates', "
                                                       "'economic activity', 'inflation and prices', 'housing market', "
                                                       "'consumer sentiment and spending', 'business and industry' "
                                                       "'banking and finance', 'government and policy', 'international indicators', "
                                                       "commodity prices"]) -> str:
    sids = {
        "employment and labor": ["UNRATE", "JTSJOL", "CIVPART", "CES0500000003"],
        "market indicators": ["SP500", "DJIA", "NASDAQCOM", "WILL5000INDFC", "VIXCLS"],
        "interest rates": ["FEDFUNDS", "GS10", "GS2", "DFF", "USD12MD156N"],
        "economic activity": ["GDPC1", "INDPRO", "DGORDER", "PAYEMS", "RSXFS"],
        "inflation and prices": ["CPIAUCSL", "PPIACO", "PCE", "CPILFESL"],
        "housing market": ["HOUST", "HSN1F", "CSUSHPISA"],
        "consumer sentiment and spending": ["UMCSENT", "PSAVERT", "DRSFRMACBS"],
        "business and industry": ["BUSINV", "ISM_MFG", "TOTBUSSMSA"],
        "banking and finance": ["TOTALSL", "TOTBKCR", "USGSEC"],
        "government and policy": ["MTSDS133FMS", "SLGTXREV", "TIC"],
        "international indicators": ["IMPUS", "EXPGS", "EXUSEU"],
        "commodity prices": ["GOLDAMGBD228NLBM", "DCOILWTICO"]
    }
    descriptions = {
        "SP500": "S&P 500 stock market index",
        "DJIA": "Dow Jones Industrial Average stock market index",
        "NASDAQCOM": "NASDAQ Composite Index stock market index",
        "WILL5000INDFC": "Wilshire 5000 Total Market Index - measures market capitalization",
        "VIXCLS": "CBOE Volatility Index - market volatility indicator",
        "FEDFUNDS": "Federal Funds Rate - influences all other interest rates",
        "GS10": "10-Year Treasury Constant Maturity Rate - long-term interest rates",
        "GS2": "2-Year Treasury Constant Maturity Rate - short-term interest rates",
        "DFF": "Effective Federal Funds Rate - daily federal funds rate",
        "USD12MD156N": "LIBOR - international lending rate",
        "GDPC1": "Real Gross Domestic Product - overall economic output",
        "INDPRO": "Industrial Production Index - measures real output for industries",
        "DGORDER": "Durable Goods Orders - indicator of manufacturing health",
        "PAYEMS": "Nonfarm Payroll - total number of paid U.S. workers, excluding farm and a few other job classifications",
        "RSXFS": "Retail Sales - measures consumer spending",
        "CPIAUCSL": "Consumer Price Index - measures consumer price inflation",
        "PPIACO": "Producer Price Index - measures wholesale price inflation",
        "PCE": "Personal Consumption Expenditures - reflects consumer spending",
        "CPILFESL": "Core Inflation Rate - inflation rate excluding food and energy prices",
        "HOUST": "Housing Starts - new residential construction projects",
        "HSN1F": "New Home Sales - sales of newly constructed homes",
        "CSUSHPISA": "Case-Shiller Home Price Index - measures home prices",
        "UNRATE": "Unemployment Rate - percentage of unemployed workforce",
        "JTSJOL": "Job Openings and Labor Turnover - number of job openings each month",
        "CIVPART": "Labor Force Participation Rate - active workforce percentage",
        "CES0500000003": "Average Hourly Earnings - average earnings, indicative of wage inflation",
        "UMCSENT": "Consumer Sentiment Index - measures consumer confidence",
        "PSAVERT": "Personal Saving Rate - percentage of income saved by households",
        "DRSFRMACBS": "Credit Card Delinquency Rates - financial stress indicator",
        "BUSINV": "Business Inventories - total inventories held by manufacturers, wholesalers, and retailers",
        "ISM_MFG": "ISM Manufacturing Index - health of the manufacturing sector",
        "TOTBUSSMSA": "Total Business Sales - combined sales of all U.S. businesses",
        "TOTALSL": "Total Consumer Credit - amount of consumer credit outstanding",
        "TOTBKCR": "Commercial Bank Credit - loans issued by commercial banks",
        "USGSEC": "Assets and Liabilities of Commercial Banks - financial health of banks",
        "MTSDS133FMS": "Federal Surplus or Deficit - federal government budget balance",
        "SLGTXREV": "State and Local Tax Revenue - financial health of state and local governments",
        "TIC": "Treasury International Capital - international capital flows",
        "IMPUS": "U.S. Imports - total value of imports, a sign of economic activity",
        "EXPGS": "U.S. Exports - total value of exports, reflects economic strength",
        "EXUSEU": "Exchange Rates - USD to Euro, affects multinational investments",
        "GOLDAMGBD228NLBM": "Gold Prices - often a safe haven during market turmoil",
        "DCOILWTICO": "Crude Oil Prices - major impact on economic conditions"
    }
    errors = []
    summaries = {}
    for req in requested_information:
        if req not in sids.keys():
            errors.append(f"{req} is not a valid category")
            continue
        else:
            requested_ids = sids[req]
        df = pd.DataFrame()
        base = "https://api.stlouisfed.org/fred/series/observations?"
        dates = f'&observation_start={start_date}'
        ftype = '&file_type=json'
        apikey = f"&api_key={FRED_API}"
        for code in requested_ids:
            cdf = pd.DataFrame()
            series_id = f"series_id={code}"
            url = f"{base}{series_id}{dates}{apikey}{ftype}"
            r = requests.get(url)
            if r.status_code == 200:
                results = r.json()['observations']
            else:
                results = []
            if len(results) > 0:
                cdf[descriptions[code]] = [i['value'] for i in results]
                cdf.index = pd.to_datetime([i['date'] for i in results])
                df = df.join(cdf, how="outer")
            else:
                errors.append(f"Failed to gather data for {code}")
        if df.empty:
            return f"No data retrieved for {req}"
        else:
            summary_prompt = f"The following is a dataframe tracking financial information for {req} starting on date {start_date}. Summarize this in detail pointing out specific values and other indicators and conclusions that would be interesting for someone evaluating a potential ETF investment:\n{df.to_string()}"
            try:
                print(f"Summarizing {req} ({len(requested_ids)} ids)")
                summary = summarizer.invoke(summary_prompt).content
                summaries[req] = summary
            except:
                errors.append(f"Failed to summarize data for {req}")
    summaries = str(summaries) if len(summaries.keys())>0 else "No data summaries available."
    if len(errors) > 0:
        summaries += f"\n\nSome requested information is missing due to the following errors: {', '.join(errors)}"
    return summaries

def save_chat(groupchat, filename):
    with open(filename, "w", encoding="utf-8") as file:
        for message in groupchat.messages:
            file.write("-"*20 + "\n")
            file.write(f'###\n{message["name"]}\n###\n')
            file.write(message["content"]+"\n")
            file.write("-"*20 + "\n")

def evaluate_etf(etf_symbol, risk = "moderate to high", save_file = "etf_evaluation_chat.txt"):
    groupchat = autogen.GroupChat(agents=[user_proxy, engineer, financial_expert], messages=[], max_round=50)
    manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=gpt4_config)
    message = f"Retrieve all information that informs an analysis of the ETF {etf_symbol}. "
    message += "Once you have all information that you can possibly get, produce a summary. "
    message += "The summary should contain all relevant information as well as a conclusion as to whether the ETF is an attractive "
    message += "investment at the current time given all current information and indicators. "
    message += f"Evaluate for the context of an investor with risk tolerance: {risk}. "
    message += f"The date of this analysis is {datetime.now().strftime('%m-%d-%Y')}"
    user_proxy.initiate_chat(manager, message=message)
    if save_file:
        save_chat(groupchat, save_file)
    return groupchat

if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description='Autogen bot that evaluates ETF given risk tolerance')
    parser.add_argument('ETF', type=str, help="The ticker symbol of the ETF to evaluate.")
    parser.add_argument('risk', type=str, help="Risk tolerance (ex low, moderate to high, high, etc)", default="moderate to high")
    parser.add_argument('save_file', type=str, help="Filename to save autogen chat to.", default=None)
    args = parser.parse_args()
    evaluate_etf(args.ETF, args.risk, args.save_file)