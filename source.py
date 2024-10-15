import csv
import numpy as np
import pandas as pd

from pybats.shared import load_sales_example
from pybats.analysis import *
from pybats.point_forecast import *
from pybats.plot import *

# Specify the path to your CSV file
csv_file_path = 'm_data._345.csv'

# Open the CSV file
with open(csv_file_path, mode='r', newline='') as file:
    csv_reader = csv.reader(file)
    rows = list(csv_reader)

    dlup = [int(row[2]) for row in rows]
    passengers = [int(row[3]) for row in rows]
    
    passengers = np.array(passengers)
    dlup = np.array(dlup)
    
    dlup = np.append(dlup, np.arange(len(dlup) + 1, 601))
    passengers = np.append(passengers, np.zeros(600 - len(passengers)))

    k = 1
    forecast_start = len(rows) + 1
    forecast_end = 599
    seasPeriods=[12]

    mod, samples = analysis(Y=passengers, X=dlup, family="poisson",
        forecast_start=forecast_start,      # First time step to forecast on
        forecast_end=forecast_end,          # Final time step to forecast on
        seasPeriods=seasPeriods,
        k=k,                                # Forecast horizon. If k>1, default is to forecast 1:k steps ahead, marginally
        prior_length=len(rows),                     # How many data point to use in defining prior
        rho=0,                             # Random effect extension, which increases the forecast variance (see Berry and West, 2019)
        deltrend=0.5,                      # Discount factor on the trend component (the intercept)
        delregn=0.5,                        # Discount factor on the regression component
        delseas=0.5,
    )
    
    forecast = median(samples)
    
    print(forecast)