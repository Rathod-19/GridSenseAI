import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

class EnergySource:
    def __init__(self, name: str, source_type: str, capacity: float, cost_per_kwh: float):
        self.name = name
        self.source_type = source_type
        self.capacity = capacity
        self.cost_per_kwh = cost_per_kwh
        self.current_generation = 0.0
        
    def generate(self, conditions: Dict) -> float:
        if self.source_type == 'grid':
            self.current_generation = min(conditions.get('demand', 0), self.capacity)
        elif self.source_type == 'solar':
            hour = conditions.get('hour', 12)
            cloud_cover = conditions.get('cloud_cover', 0.2)
            if 6 <= hour <= 18:
                solar_factor = np.sin((hour - 6) * np.pi / 12)
                self.current_generation = self.capacity * solar_factor * (1 - cloud_cover * 0.7)
            else:
                self.current_generation = 0.0
        elif self.source_type == 'wind':
            wind_speed = conditions.get('wind_speed', 5)
            if wind_speed < 3:
                self.current_generation = 0.0
            elif wind_speed > 25:
                self.current_generation = 0.0
            else:
                wind_factor = min((wind_speed - 3) / 12, 1.0)
                self.current_generation = self.capacity * wind_factor
        
        return self.current_generation

class Consumer:
    def __init__(self, name: str, consumer_type: str, base_load: float, peak_load: float):
        self.name = name
        self.consumer_type = consumer_type
        self.base_load = base_load
        self.peak_load = peak_load
        self.current_demand = 0.0
        
    def calculate_demand(self, hour: int, day_of_week: int, load_multiplier: float = 1.0) -> float:
        if self.consumer_type == 'residential':
            if hour < 6:
                demand_factor = 0.3
            elif 6 <= hour < 9:
                demand_factor = 0.8
            elif 9 <= hour < 17:
                demand_factor = 0.5
            elif 17 <= hour < 22:
                demand_factor = 1.0
            else:
                demand_factor = 0.6
            
            if day_of_week >= 5:
                demand_factor *= 1.2
                
        elif self.consumer_type == 'commercial':
            if 8 <= hour < 18:
                demand_factor = 1.0
            elif 6 <= hour < 8 or 18 <= hour < 20:
                demand_factor = 0.6
            else:
                demand_factor = 0.2
            
            if day_of_week >= 5:
                demand_factor *= 0.4
                
        elif self.consumer_type == 'industrial':
            if 6 <= hour < 22:
                demand_factor = 1.0
            else:
                demand_factor = 0.7
        else:
            demand_factor = 0.5
        
        noise = np.random.normal(0, 0.05)
        self.current_demand = (self.base_load + (self.peak_load - self.base_load) * demand_factor) * load_multiplier * (1 + noise)
        self.current_demand = max(0, self.current_demand)
        
        return self.current_demand

class SmartGrid:
    def __init__(self):
        self.energy_sources: List[EnergySource] = []
        self.consumers: List[Consumer] = []
        self.current_time = datetime.now()
        self.history = []
        
    def add_energy_source(self, source: EnergySource):
        self.energy_sources.append(source)
        
    def add_consumer(self, consumer: Consumer):
        self.consumers.append(consumer)
        
    def simulate_step(self, conditions: Dict, load_multiplier: float = 1.0) -> Dict:
        hour = self.current_time.hour
        day_of_week = self.current_time.weekday()
        
        total_demand = sum(consumer.calculate_demand(hour, day_of_week, load_multiplier) 
                          for consumer in self.consumers)
        
        conditions['hour'] = hour
        conditions['demand'] = total_demand
        
        generations = {}
        for source in self.energy_sources:
            gen = source.generate(conditions)
            generations[source.name] = gen
        
        total_generation = sum(generations.values())
        
        renewable_generation = sum(
            gen for source, gen in zip(self.energy_sources, generations.values())
            if source.source_type in ['solar', 'wind']
        )
        
        grid_generation = sum(
            gen for source, gen in zip(self.energy_sources, generations.values())
            if source.source_type == 'grid'
        )
        
        energy_deficit = max(0, total_demand - total_generation)
        energy_surplus = max(0, total_generation - total_demand)
        
        total_cost = sum(
            source.cost_per_kwh * gen 
            for source, gen in zip(self.energy_sources, generations.values())
        )
        
        renewable_percentage = (renewable_generation / total_generation * 100) if total_generation > 0 else 0
        
        step_data = {
            'timestamp': self.current_time,
            'total_demand': total_demand,
            'total_generation': total_generation,
            'renewable_generation': renewable_generation,
            'grid_generation': grid_generation,
            'energy_deficit': energy_deficit,
            'energy_surplus': energy_surplus,
            'total_cost': total_cost,
            'renewable_percentage': renewable_percentage,
            'generations': generations.copy(),
            'demands': {consumer.name: consumer.current_demand for consumer in self.consumers}
        }
        
        self.history.append(step_data)
        self.current_time += timedelta(hours=1)
        
        return step_data
    
    def get_history_dataframe(self) -> pd.DataFrame:
        if not self.history:
            return pd.DataFrame()
        
        df_data = []
        for record in self.history:
            row = {
                'timestamp': record['timestamp'],
                'total_demand': record['total_demand'],
                'total_generation': record['total_generation'],
                'renewable_generation': record['renewable_generation'],
                'grid_generation': record['grid_generation'],
                'energy_deficit': record['energy_deficit'],
                'energy_surplus': record['energy_surplus'],
                'total_cost': record['total_cost'],
                'renewable_percentage': record['renewable_percentage']
            }
            for source_name, gen in record['generations'].items():
                row[f'gen_{source_name}'] = gen
            for consumer_name, demand in record['demands'].items():
                row[f'demand_{consumer_name}'] = demand
            df_data.append(row)
        
        return pd.DataFrame(df_data)
    
    def reset_simulation(self, start_time: Optional[datetime] = None):
        self.history = []
        self.current_time = start_time if start_time else datetime.now()
