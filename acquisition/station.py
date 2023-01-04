import pandas as pd
import numpy as np
import util

#Only work for 서울시 공공자전거 대여소 정보 data
def read_station_data(source, encoding='cp949'):
    station_df = pd.read_csv(source, encoding=encoding)
    station_df = station_df.iloc[:,:-3] #drop useless column
    station_df = station_df.iloc[4:] #drop useless row
    station_df = station_df.reset_index(drop=True)
    station_df.columns = ["station_ID", "station_name", "location", "address", "latitude", "longitude", "installation_date"]
    station_df = station_df.astype({"station_ID":np.uint32, "latitude":np.float64, "longitude":np.float64})
    return station_df

def process_station_data(station_df):
    #change coordinate system
    changed_coordinate = np.array([util.change_coordinate_system(coordinate) for coordinate in zip(station_df['latitude'], station_df['longitude'])]).T
    station_df['x'] = changed_coordinate[0]
    station_df['y'] = changed_coordinate[1]

    return station_df

    
