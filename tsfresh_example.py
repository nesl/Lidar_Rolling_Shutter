from tsfresh import extract_relevant_features
import pandas as pd
import os, os.path
import pdb

freq_dir = "/home/kiototeko/tareas/vibrometry_laser"

final_time_series = pd.DataFrame(columns = ["id", "time", "ch1", "ch2", "ch3", "ch4", "ch5", "ch6", "ch7", "ch8", "ch9", "ch10", "ch11", "ch12", "ch13", "ch14"])

y = []

for idx,i in enumerate(range(100,550,50)):
    freq_path = os.path.join(freq_dir, str(i) + ".csv")
    time_series = pd.read_csv(freq_path, header=None)#, usecols=list(range(0,700)))
    time_series = time_series.T
    time_series.columns = ["ch1", "ch2", "ch3", "ch4", "ch5", "ch6", "ch7", "ch8", "ch9", "ch10", "ch11", "ch12", "ch13", "ch14"]
    time_series.insert(loc=0, column="time", value=[n % 70 for n in range(0,time_series.shape[0])])
    time_series.insert(loc=0, column="id", value=[idx for n in range(0,time_series.shape[0])])
    final_time_series = final_time_series.append(time_series)
    y.append(i)
   
y = pd.Series(y)
print(final_time_series.head())
features_filtered_direct = extract_relevant_features(final_time_series, y, column_id='id', column_sort='time')
print(features_filtered_direct.shape)
print(features_filtered_direct)
