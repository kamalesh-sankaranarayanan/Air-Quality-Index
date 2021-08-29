# -*- coding: utf-8 -*-
"""
Created on Sat Aug 28 16:05:49 2021

@author: Harish
"""

import os
import time # to calculate time it takes to download from the website
import requests # download web page in html
import sys

def retrieve_html():
    for year in range(2013, 2021):
        for month in range(1,13):
            if (month < 10):
                url = 'https://en.tutiempo.net/climate/0{}-{}/ws-432950.html'.format(month, year)
            else:
                url = 'https://en.tutiempo.net/climate/{}-{}/ws-432950.html'.format(month, year)
            raw = requests.get(url)
            raw_utf = raw.text.encode('utf-8')
            
            if not os.path.exists("C:/Users\Harish/Documents/Projects/Air Quality Index/Data/Html_data/{}".format(year)):
                os.makedirs("C:/Users\Harish/Documents/Projects/Air Quality Index/Data/Html_data/{}".format(year))
            with open("C:/Users\Harish/Documents/Projects/Air Quality Index/Data/Html_data/{}/{}.html".format(year, month), "wb") as output:
                output.write(raw_utf)
            
            sys.stdout.flush()
        
        

if __name__ == '__main__':
    start_time = time.time()
    retrieve_html()
    stop_time = time.time()
    print("Time taken {}".format(stop_time - start_time))