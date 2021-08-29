# Air-Quality-Index
Air Quality Index prediction using Random forest model and Streamlit


* Created a tool that predicts air quality index (Particulate Matter 2.5 (PM2.5))
* Scraped historical data from https://en.tutiempo.net/ and https://openweathermap.org/api/air-pollution using Requests and BeautifulSoup
*  Optimized Linear, LASSO, Ridge, Random forest regressors using GirdSearchCV and RandomizedSearchCV to reach the best model
*  Built a client app using Streamlit


# Code and Resources used
Pythoon Version: 3.8 \
Packages: pandas, numpy, requests, beautifulsoup, sklearn, matplotlib, seaborn,  pickle

# Web Scraping
Used Requests package to scrape historical records form https://en.tutiempo.net/ and https://openweathermap.org/api/air-pollution to get the following variables: \

* T - Average Temperature (°C)
* TM	- Maximum temperature (°C)
* Tm	- Minimum temperature (°C)
* SLP - Atmospheric pressure at sea level (hPa)
* H - Average relative humidity (%)
* VV	- Average visibility (Km)
* V - Average wind speed (Km/h)
* VM	- Maximum sustained wind speed (Km/h)
* PM 2.5 - Particulate matter 2.5 (PM2.5)

# Data Cleaning
After scraping the data, I needed to clean it up so that it was usable for our model following changes were made:

* Converted data from HTML table into datafrom
* Removed null values
* Parsed numeric data for all columns

# EDA
I looked at the distribution of data and below are the few highlights: \
![Alt Text](https://github.com/Harishkumar215/Air-Quality-Index/blob/main/Images/Figure%202021-08-28%20182406.png)
![Alt Text](https://github.com/Harishkumar215/Air-Quality-Index/blob/main/Images/Figure%202021-08-28%20182425.png)
![Alt Text](https://github.com/Harishkumar215/Air-Quality-Index/blob/main/Images/Figure%202021-08-28%20183150.png)

# Model Building
I split the data into train and tests sets with test size of 30%

I tired different models and evaluated them using Mean Absolute Error (MAE)

I tried following models:

* Linear Regression - Baseline for the model
* LASSO and Ridge regression - I thought normalized regression would be effective
* Random Forest - Because of sparsity associated with data, I thought that this would be good fit

# Model Performance


