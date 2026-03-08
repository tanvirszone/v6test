import pandas as pd
import numpy as np

def load_and_process_data(file_path):
    df = pd.read_csv(file_path)
    df['CRASH DATE'] = pd.to_datetime(df['CRASH DATE'])
    
    # Top factors & vehicles
    top_factors = df['CONTRIBUTING FACTOR VEHICLE 1'].value_counts().nlargest(5).index
    top_vehicles = df['VEHICLE TYPE CODE 1'].value_counts().nlargest(10).index
    
    df['factor_grouped'] = df['CONTRIBUTING FACTOR VEHICLE 1'].where(
        df['CONTRIBUTING FACTOR VEHICLE 1'].isin(top_factors), 'RARE FACTORS'
    )
    df['vehicle_grouped'] = df['VEHICLE TYPE CODE 1'].where(
        df['VEHICLE TYPE CODE 1'].isin(top_vehicles), 'RARE TYPES'
    )
    
    # Aggregate daily
    factor_counts = pd.crosstab(df['CRASH DATE'], df['factor_grouped'])
    vehicle_counts = pd.crosstab(df['CRASH DATE'], df['vehicle_grouped'])
    daily_injuries = df.groupby('CRASH DATE')['NUMBER OF PERSONS INJURED'].sum()
    daily_weather = df.groupby('CRASH DATE')['WEATHER TEMP'].mean()
    daily_holiday = df.groupby('CRASH DATE')['FEDERAL HOLIDAY'].first()
    
    daily_df = pd.concat([daily_injuries, factor_counts, vehicle_counts, daily_weather, daily_holiday], axis=1)
    daily_df = daily_df.reset_index()
    daily_df = daily_df.rename(columns={
        'NUMBER OF PERSONS INJURED':'DAILY_INJURIES',
        'WEATHER TEMP':'DAILY_TEMP'
    }).sort_values('CRASH DATE')
    
    # Time-based features
    daily_df['day_of_week'] = daily_df['CRASH DATE'].dt.dayofweek
    daily_df['month'] = daily_df['CRASH DATE'].dt.month
    daily_df['is_weekend'] = daily_df['day_of_week'].isin([5,6]).astype(int)
    
    daily_df['dow_sin'] = np.sin(2 * np.pi * daily_df['day_of_week']/7)
    daily_df['dow_cos'] = np.cos(2 * np.pi * daily_df['day_of_week']/7)
    daily_df['month_sin'] = np.sin(2 * np.pi * daily_df['month']/12)
    daily_df['month_cos'] = np.cos(2 * np.pi * daily_df['month']/12)
    
    # Lag features
    for lag in [1,2,3,7,14]:
        daily_df[f'lag_{lag}'] = daily_df['DAILY_INJURIES'].shift(lag)
    for window in [7,14]:
        daily_df[f'roll_mean_{window}'] = daily_df['DAILY_INJURIES'].rolling(window).mean().shift(1)
    
    daily_df = daily_df.dropna().reset_index(drop=True)
    return daily_df