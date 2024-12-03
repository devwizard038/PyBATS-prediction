import csv
import numpy as np
import pandas as pd
import math

from pandas.tseries.holiday import USFederalHolidayCalendar

from pybats.shared import load_sales_example
from pybats.analysis import *
from pybats.point_forecast import *
from pybats.plot import *

# Specify the path to your CSV file
csv_file_path = 'PMI.csv'
month_abbreviations = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

rounds = 30

# Open the CSV file
with open(csv_file_path, mode='r', newline='') as file:
    csv_reader = csv.reader(file)
    rows = list(csv_reader)

    dlup = [int(row[2]) for row in rows]
    passengers = [int(row[3]) for row in rows]
    
    print(passengers)
    
    passengers = np.array(passengers)
    dlup = np.array(dlup)
    
    dlup = np.append(dlup, np.arange(len(dlup) + 1, 602))
    passengers = np.append(passengers, np.zeros(601 - len(passengers)))

    k = 1
    forecast_start = len(rows)
    forecast_end = 600
    seasPeriods=[12]
    seasHarmComponents = [[1, 2, 5, 6, 11, 12, 7, 10, 8, 9, 3, 4]]
    
    start_year = (len(rows)) // 12 + 1991
    start_month = (len(rows)) % 12
    
    sum_forecast = np.zeros(forecast_end - forecast_start + 1)

    for i in range(rounds):
        mod, samples = analysis(Y=passengers, X=dlup, family="poisson",
            forecast_start=forecast_start,      # First time step to forecast on
            forecast_end=forecast_end,          # Final time step to forecast on
            seasPeriods=seasPeriods,
            seasHarmComponents=seasHarmComponents,
            k=k,                                # Forecast horizon. If k>1, default is to forecast 1:k steps ahead, marginally
            prior_length=len(rows),                     # How many data point to use in defining prior
            rho=.0,                             # Random effect extension, which increases the forecast variance (see Berry and West, 2019)
            deltrend=1,                      # Discount factor on the trend component (the intercept)
            delregn=1,                        # Discount factor on the regression component
            delseas=1,
        )
    
        forecast = median(samples)
        
        sum_forecast = np.add(sum_forecast, forecast)
        
    jul_forecast = sum_forecast[0][0] / rounds
    ratio = jul_forecast / passengers[len(rows) - 1]
    
    forecast_result = []
        
    file_path = 'result.csv'
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        # Write header if the file does not exist
    
        for i in np.arange(1, len(sum_forecast)):
            # Calculate the current year and month
            year = start_year + (start_month + i - 1) // 12
            month_index = (start_month + i - 1) % 12  # Month index for the abbreviation
            
            csv_data = []
            csv_data.append(month_abbreviations[month_index] + ' ' + str(year))
            csv_data.append(str(math.floor(sum_forecast[i][0] / rounds / ratio)))
            
            # Print the result in MMM/YYYY -> value format
            writer.writerow(csv_data)