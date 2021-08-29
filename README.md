# Air-Quality-Index
Air Quality Index prediction using Random forest model and Streamlit


* Created a tool that predicts air quality index (Particulate Matter 2.5 (PM2.5))
* Scraped historical data from https://en.tutiempo.net/ using Requests and BeautifulSoup
*  Optimized Linear, LASSO, Ridge, Random forest regressors using GirdSearchCV and RandomizedSearchCV to reach the best model
*  Built a client app using Streamlit


# Code and Resources used
Pythoon Version: 3.8 \
Packages: pandas, numpy, requests, beautifulsoup, sklearn, matplotlib, seaborn,  pickle

# Web Scraping
Used Requests package to scrape historical records from 2013 to 2018 form https://en.tutiempo.net/ and used BeautifulSoup to clean the data to get the following: \

* T - Average Temperature (°C)
* TM	- Maximum temperature (°C)
* Tm	- Minimum temperature (°C)
* SLP - Atmospheric pressure at sea level (hPa)
* H - Average relative humidity (%)
* VV	- Average visibility (Km)
* V - Average wind speed (Km/h)
* VM	- Maximum sustained wind speed (Km/h)
* PM 2.5 - Particulate matter 2.5 (PM2.5)

