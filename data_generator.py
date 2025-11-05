import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional

class WeatherSimulator:
    def __init__(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.current_conditions = {
            'cloud_cover': 0.3,
            'wind_speed': 8.0,
            'temperature': 20.0
        }
    
    def update_weather(self, hour: int) -> Dict:
        hour_factor = np.sin((hour - 6) * np.pi / 12)
        
        cloud_base = 0.3 + 0.2 * np.sin(hour * np.pi / 24)
        cloud_noise = np.random.normal(0, 0.1)
        self.current_conditions['cloud_cover'] = np.clip(cloud_base + cloud_noise, 0, 1)
        
        wind_base = 8 + 3 * np.sin(hour * np.pi / 12)
        wind_noise = np.random.normal(0, 2)
        self.current_conditions['wind_speed'] = np.clip(wind_base + wind_noise, 0, 30)
        
        temp_base = 20 + 10 * hour_factor
        temp_noise = np.random.normal(0, 2)
        self.current_conditions['temperature'] = temp_base + temp_noise
        
        return self.current_conditions.copy()

class HistoricalDataGenerator:
    def __init__(self, grid):
        self.grid = grid
        self.weather_sim = WeatherSimulator(seed=42)
    
    def generate_historical_data(self, 
                                days: int = 30, 
                                start_date: Optional[datetime] = None) -> pd.DataFrame:
        if start_date is None:
            start_date = datetime.now() - timedelta(days=days)
        
        self.grid.reset_simulation(start_date)
        
        hours = days * 24
        
        for _ in range(hours):
            hour = self.grid.current_time.hour
            conditions = self.weather_sim.update_weather(hour)
            self.grid.simulate_step(conditions, load_multiplier=1.0)
        
        return self.grid.get_history_dataframe()
    
    def generate_baseline_data(self,
                              days: int = 30,
                              start_date: Optional[datetime] = None) -> pd.DataFrame:
        if start_date is None:
            start_date = datetime.now() - timedelta(days=days)
        
        self.grid.reset_simulation(start_date)
        
        hours = days * 24
        baseline_data = []
        
        for _ in range(hours):
            hour = self.grid.current_time.hour
            day_of_week = self.grid.current_time.weekday()
            
            total_demand = sum(
                consumer.calculate_demand(hour, day_of_week, 1.0) 
                for consumer in self.grid.consumers
            )
            
            grid_cost_per_kwh = next(
                (source.cost_per_kwh for source in self.grid.energy_sources 
                 if source.source_type == 'grid'),
                0.15
            )
            
            baseline_cost = total_demand * grid_cost_per_kwh
            
            baseline_data.append({
                'timestamp': self.grid.current_time,
                'demand': total_demand,
                'cost': baseline_cost
            })
            
            self.grid.current_time += timedelta(hours=1)
        
        return pd.DataFrame(baseline_data)

def create_sample_grid():
    from grid_simulation import SmartGrid, EnergySource, Consumer
    
    grid = SmartGrid()
    
    grid.add_energy_source(EnergySource(
        name='Main Grid',
        source_type='grid',
        capacity=500.0,
        cost_per_kwh=0.15
    ))
    
    grid.add_energy_source(EnergySource(
        name='Solar Farm',
        source_type='solar',
        capacity=200.0,
        cost_per_kwh=0.05
    ))
    
    grid.add_energy_source(EnergySource(
        name='Wind Turbines',
        source_type='wind',
        capacity=150.0,
        cost_per_kwh=0.07
    ))
    
    grid.add_consumer(Consumer(
        name='Residential Area',
        consumer_type='residential',
        base_load=50.0,
        peak_load=150.0
    ))
    
    grid.add_consumer(Consumer(
        name='Commercial District',
        consumer_type='commercial',
        base_load=80.0,
        peak_load=200.0
    ))
    
    grid.add_consumer(Consumer(
        name='Industrial Zone',
        consumer_type='industrial',
        base_load=100.0,
        peak_load=150.0
    ))
    
    return grid
