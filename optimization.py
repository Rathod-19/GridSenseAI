import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.optimize import linprog

class EnergyOptimizer:
    def __init__(self, prioritize_renewable=True):
        self.prioritize_renewable = prioritize_renewable
        
    def optimize_dispatch_rule_based(self, 
                                     demand: float, 
                                     available_sources: Dict[str, Dict],
                                     forecasted_demand: Optional[List[float]] = None) -> Dict[str, float]:
        dispatch = {}
        remaining_demand = demand
        
        renewable_sources = {k: v for k, v in available_sources.items() 
                           if v['type'] in ['solar', 'wind']}
        grid_sources = {k: v for k, v in available_sources.items() 
                       if v['type'] == 'grid'}
        
        if self.prioritize_renewable:
            for source_name, source_info in sorted(renewable_sources.items(), 
                                                   key=lambda x: x[1]['cost']):
                available = source_info['capacity']
                dispatch_amount = min(available, remaining_demand)
                dispatch[source_name] = dispatch_amount
                remaining_demand -= dispatch_amount
                
                if remaining_demand <= 0:
                    break
        
        if remaining_demand > 0:
            for source_name, source_info in sorted(grid_sources.items(), 
                                                   key=lambda x: x[1]['cost']):
                available = source_info['capacity']
                dispatch_amount = min(available, remaining_demand)
                dispatch[source_name] = dispatch_amount
                remaining_demand -= dispatch_amount
                
                if remaining_demand <= 0:
                    break
        
        for source_name in available_sources.keys():
            if source_name not in dispatch:
                dispatch[source_name] = 0.0
        
        return dispatch
    
    def optimize_dispatch_linear_programming(self,
                                             demand: float,
                                             available_sources: Dict[str, Dict]) -> Dict[str, float]:
        source_names = list(available_sources.keys())
        n_sources = len(source_names)
        
        if n_sources == 0:
            return {}
        
        costs = [available_sources[name]['cost'] for name in source_names]
        capacities = [available_sources[name]['capacity'] for name in source_names]
        
        renewable_bonus = []
        for name in source_names:
            if available_sources[name]['type'] in ['solar', 'wind']:
                renewable_bonus.append(-0.01)
            else:
                renewable_bonus.append(0)
        
        c = [cost + bonus for cost, bonus in zip(costs, renewable_bonus)]
        
        A_ub = [[1 if i == j else 0 for j in range(n_sources)] for i in range(n_sources)]
        b_ub = capacities
        
        A_eq = [[1] * n_sources]
        b_eq = [demand]
        
        bounds = [(0, cap) for cap in capacities]
        
        try:
            result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, 
                           bounds=bounds, method='highs')
            
            if result.success:
                dispatch = {name: result.x[i] for i, name in enumerate(source_names)}
            else:
                dispatch = self.optimize_dispatch_rule_based(demand, available_sources)
        except:
            dispatch = self.optimize_dispatch_rule_based(demand, available_sources)
        
        return dispatch
    
    def calculate_optimization_metrics(self, 
                                       optimized_dispatch: Dict[str, float],
                                       available_sources: Dict[str, Dict],
                                       demand: float) -> Dict:
        total_generation = sum(optimized_dispatch.values())
        
        renewable_generation = sum(
            amount for source_name, amount in optimized_dispatch.items()
            if available_sources[source_name]['type'] in ['solar', 'wind']
        )
        
        grid_generation = sum(
            amount for source_name, amount in optimized_dispatch.items()
            if available_sources[source_name]['type'] == 'grid'
        )
        
        total_cost = sum(
            amount * available_sources[source_name]['cost']
            for source_name, amount in optimized_dispatch.items()
        )
        
        renewable_percentage = (renewable_generation / total_generation * 100) if total_generation > 0 else 0
        
        unmet_demand = max(0, demand - total_generation)
        surplus = max(0, total_generation - demand)
        
        return {
            'total_generation': total_generation,
            'renewable_generation': renewable_generation,
            'grid_generation': grid_generation,
            'total_cost': total_cost,
            'renewable_percentage': renewable_percentage,
            'unmet_demand': unmet_demand,
            'surplus': surplus
        }
    
    def suggest_load_shifting(self,
                             current_demand: float,
                             forecasted_demand: List[float],
                             forecasted_renewable: List[float]) -> List[Dict]:
        suggestions = []
        
        for i, (future_demand, future_renewable) in enumerate(zip(forecasted_demand[:24], 
                                                                   forecasted_renewable[:24])):
            if future_renewable > future_demand and current_demand > forecasted_renewable[0]:
                renewable_surplus_ratio = (future_renewable - future_demand) / future_demand if future_demand > 0 else 0
                
                if renewable_surplus_ratio > 0.2:
                    suggestions.append({
                        'hours_ahead': i + 1,
                        'recommended_action': 'shift_load',
                        'reason': f'High renewable availability expected ({renewable_surplus_ratio*100:.1f}% surplus)',
                        'potential_savings': renewable_surplus_ratio * 0.5
                    })
        
        return suggestions[:3]

class GridController:
    def __init__(self, optimizer: EnergyOptimizer):
        self.optimizer = optimizer
        self.dispatch_history = []
        
    def execute_dispatch(self, 
                        demand: float,
                        available_sources: Dict[str, Dict],
                        use_linear_programming: bool = False) -> Dict:
        if use_linear_programming:
            dispatch = self.optimizer.optimize_dispatch_linear_programming(demand, available_sources)
        else:
            dispatch = self.optimizer.optimize_dispatch_rule_based(demand, available_sources)
        
        metrics = self.optimizer.calculate_optimization_metrics(dispatch, available_sources, demand)
        
        result = {
            'dispatch': dispatch,
            'metrics': metrics
        }
        
        self.dispatch_history.append(result)
        
        return result
    
    def get_efficiency_comparison(self, 
                                  optimized_cost: float, 
                                  baseline_cost: float) -> Dict:
        cost_savings = baseline_cost - optimized_cost
        savings_percentage = (cost_savings / baseline_cost * 100) if baseline_cost > 0 else 0
        
        return {
            'cost_savings': cost_savings,
            'savings_percentage': savings_percentage,
            'optimized_cost': optimized_cost,
            'baseline_cost': baseline_cost
        }
