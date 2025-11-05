import pandas as pd
import numpy as np
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from typing import Tuple, List, Union, Any
import warnings
warnings.filterwarnings('ignore')

class EnergyForecaster:
    def __init__(self, forecast_method='prophet'):
        self.forecast_method = forecast_method
        self.demand_model = None
        self.solar_model = None
        self.wind_model = None
        
    def prepare_data_for_prophet(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        prophet_df = pd.DataFrame({
            'ds': df['timestamp'],
            'y': df[column]
        })
        return prophet_df
    
    def train_demand_forecaster(self, historical_data: pd.DataFrame):
        if self.forecast_method == 'prophet':
            self.demand_model = Prophet(
                daily_seasonality=True,
                weekly_seasonality=True,
                yearly_seasonality=False,
                changepoint_prior_scale=0.05
            )
            prophet_data = self.prepare_data_for_prophet(historical_data, 'total_demand')
            self.demand_model.fit(prophet_data)
        elif self.forecast_method == 'arima':
            self.demand_model = ARIMA(historical_data['total_demand'], order=(2, 1, 2))
            self.demand_model = self.demand_model.fit()
    
    def train_renewable_forecasters(self, historical_data: pd.DataFrame):
        if 'renewable_generation' in historical_data.columns:
            if self.forecast_method == 'prophet':
                self.solar_model = Prophet(
                    daily_seasonality=True,
                    weekly_seasonality=False,
                    yearly_seasonality=False
                )
                prophet_data = self.prepare_data_for_prophet(historical_data, 'renewable_generation')
                self.solar_model.fit(prophet_data)
    
    def forecast_demand(self, periods: int = 24) -> pd.DataFrame:
        if self.demand_model is None:
            return pd.DataFrame()
        
        if self.forecast_method == 'prophet':
            future = self.demand_model.make_future_dataframe(periods=periods, freq='h')
            forecast = self.demand_model.predict(future)
            forecast['yhat'] = forecast['yhat'].clip(lower=0)
            return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)
        elif self.forecast_method == 'arima':
            forecast_values = self.demand_model.forecast(steps=periods)
            forecast_values = np.maximum(forecast_values, 0)
            return pd.DataFrame({
                'yhat': forecast_values
            })
    
    def forecast_renewable(self, periods: int = 24) -> pd.DataFrame:
        if self.solar_model is None:
            return pd.DataFrame()
        
        if self.forecast_method == 'prophet':
            future = self.solar_model.make_future_dataframe(periods=periods, freq='h')
            forecast = self.solar_model.predict(future)
            forecast['yhat'] = forecast['yhat'].clip(lower=0)
            return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)
        
        return pd.DataFrame()
    
    def calculate_forecast_accuracy(self, actual: Union[pd.Series, Any], predicted: Union[pd.Series, Any]) -> dict:
        if len(actual) == 0 or len(predicted) == 0:
            return {'mae': 0, 'rmse': 0, 'mape': 0}
        
        actual_vals = actual.values if isinstance(actual, pd.Series) else actual
        predicted_vals = predicted.values if isinstance(predicted, pd.Series) else predicted
        
        min_len = min(len(actual_vals), len(predicted_vals))
        actual_vals = actual_vals[:min_len]
        predicted_vals = predicted_vals[:min_len]
        
        mae = np.mean(np.abs(actual_vals - predicted_vals))
        rmse = np.sqrt(np.mean((actual_vals - predicted_vals) ** 2))
        
        non_zero_mask = actual_vals != 0
        if np.any(non_zero_mask):
            mape = np.mean(np.abs((actual_vals[non_zero_mask] - predicted_vals[non_zero_mask]) / actual_vals[non_zero_mask])) * 100
        else:
            mape = 0
        
        return {
            'mae': mae,
            'rmse': rmse,
            'mape': mape
        }

class SimpleForecaster:
    def __init__(self):
        self.demand_pattern = None
        self.renewable_pattern = None
        
    def train_from_patterns(self, historical_data: pd.DataFrame):
        if 'total_demand' in historical_data.columns:
            historical_data['hour'] = pd.to_datetime(historical_data['timestamp']).dt.hour
            self.demand_pattern = historical_data.groupby('hour')['total_demand'].mean()
        
        if 'renewable_generation' in historical_data.columns:
            self.renewable_pattern = historical_data.groupby('hour')['renewable_generation'].mean()
    
    def forecast_demand(self, current_hour: int, periods: int = 24) -> List[float]:
        if self.demand_pattern is None:
            return [0] * periods
        
        forecasts = []
        for i in range(periods):
            hour = (current_hour + i) % 24
            forecasts.append(self.demand_pattern.get(hour, 0))
        
        return forecasts
    
    def forecast_renewable(self, current_hour: int, periods: int = 24) -> List[float]:
        if self.renewable_pattern is None:
            return [0] * periods
        
        forecasts = []
        for i in range(periods):
            hour = (current_hour + i) % 24
            forecasts.append(self.renewable_pattern.get(hour, 0))
        
        return forecasts
