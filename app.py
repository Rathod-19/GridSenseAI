import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from grid_simulation import SmartGrid, EnergySource, Consumer
from forecasting import EnergyForecaster, SimpleForecaster
from optimization import EnergyOptimizer, GridController
from data_generator import WeatherSimulator, HistoricalDataGenerator, create_sample_grid

st.set_page_config(
    page_title="GridSense - AI Smart Grid Management",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    body,
    div[data-testid="stAppViewContainer"], /* Target the main app container */
    div[data-testid="stVerticalBlock"] {
        background-color: #f0f7ff !important; /* Light sky blue background */
    } 

    /* Hide Streamlit's default menu and deploy button */
    #MainMenu {
        visibility: hidden;
    }

    .main .block-container {
        background-color: #ffffff !important; /* White content block for clean contrast */
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    .main-header {
        font-size: 2rem !important; /* Made header significantly larger */
        font-weight: 900; /* Bolder font weight */
        background: linear-gradient(135deg, #0ea5e9 0%, #2563eb 100%); /* Sky blue to Indigo gradient */
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        padding-bottom: 0.5rem;
        text-align: center;
        line-height: 1.1; /* Adjusted for new size */
        margin-bottom: 2rem;
        letter-spacing: -0.04em; /* Slightly tighter letter spacing */
    }
    
    .section-header {
        font-size: 2rem;
        font-weight: 700;
        color: #1e40af; /* Dark Blue for headers */
        margin: 2rem 0 1rem 0;
        border-left: 4px solid #0ea5e9; /* Sky Blue accent */
        padding-left: 1rem;
    }
    
    .subsection-header {
        font-size: 1.25rem;
        font-weight: 700;
        color: #1d4ed8; /* Slightly lighter Dark Blue */
        margin: 1rem 0 0.75rem 0;
    }
    
    .stMetric {
        background-color: #ffffff; /* White cards for contrast */
        padding: 1.25rem;
        border-radius: 12px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
        border: 1px solid #e2e8f0;
        transition: transform 0.2s;
    }
    
    .stMetric:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.08);
    }
    
    .stMetric label {
        font-size: 0.875rem !important;
        font-weight: 600 !important;
        color: #4a5568 !important; /* Gray text */
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .stMetric [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 700 !important;
        color: #1a202c !important; /* Dark text */
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: #e0f2fe; /* Light cyan for tab background */
        padding: 0.5rem;
        border-radius: 12px;
    }
    
    .stTabs [data-baseweb="tab"] {
        font-size: 1rem;
        font-weight: 600;
        color: #0284c7; /* Medium blue text for inactive tabs */
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
    }
    
    .stTabs [aria-selected="true"] {
        background: #0ea5e9; /* Sky blue for active tab */
        color: #ffffff !important;
    }
    
    div[data-testid="stSidebar"] {
        background-color: #ffffff;
    }
    
    div[data-testid="stSidebar"] h1, 
    div[data-testid="stSidebar"] h2,
    div[data-testid="stSidebar"] h3 {
        color: #1a202c;
        font-weight: 700;
    }
    
    .stButton button {
        font-weight: 600;
        border-radius: 8px;
        border: 1px solid #2563eb;
        transition: all 0.2s;
    }
    
    .stButton button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 6px -1px rgba(0, 0, a, 0.1);
    }
    
    [data-testid="stDataFrame"] {
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid #e2e8f0;
    }
    
    .info-card {
        background: #ffffff;
        padding: 1rem;
        border-radius: 12px;
        border-left: 4px solid #0ea5e9;
        font-size: 1rem;
        font-weight: 500;
        color: #2d3748;
}
    
    h1 {
        font-size: 2.25rem !important;
        font-weight: 700 !important;
        color: #1a202c !important;
    }
    
    h2 {
        font-size: 2rem !important;
        font-weight: 700 !important;
        color: #1a202c !important;
    }
    
    h3 {
        font-size: 1.5rem !important;
        font-weight: 600 !important;
        color: #2d3748 !important;
    }
    
    p, div, span {
        font-size: 1rem;
        color: #4a5568;
        line-height: 1.6;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_grid():
    return create_sample_grid()

@st.cache_data
def generate_initial_data(_grid):
    data_gen = HistoricalDataGenerator(_grid)
    historical_df = data_gen.generate_historical_data(days=7)
    baseline_df = data_gen.generate_baseline_data(days=7)
    return historical_df, baseline_df

def create_energy_flow_chart(current_data):
    sources = list(current_data['generations'].keys())
    values = list(current_data['generations'].values())
    
    colors = []
    for source in sources:
        if 'Solar' in source:
            colors.append('#fbbf24')
        elif 'Wind' in source:
            colors.append('#60a5fa')
        else:
            colors.append('#ef4444')
    
    fig = go.Figure(data=[go.Bar(
        x=sources,
        y=values,
        marker_color=colors,
        text=[f'<b>{v:.1f} kW</b>' for v in values],
        textposition='auto',
        textfont=dict(size=14, color='white')
    )])
    
    fig.add_hline(y=current_data['total_demand'], 
                  line_dash="dash", 
                  line_color="#7c3aed",
                  line_width=3,
                  annotation_text=f"<b>Demand: {current_data['total_demand']:.1f} kW</b>",
                  annotation_font=dict(size=14, color='#7c3aed'))
    
    fig.update_layout(
        title=dict(
            text="<b>Current Energy Generation by Source</b>",
            font=dict(size=20, color='#1a202c')
        ),
        xaxis_title=dict(text="<b>Energy Source</b>", font=dict(size=14)),
        yaxis_title=dict(text="<b>Generation (kW)</b>", font=dict(size=14)),
        height=400,
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12, color='#4a5568')
    )
    
    return fig

def create_time_series_chart(df, columns, title, ylabel):
    fig = go.Figure()
    
    colors = {'total_demand': '#ef4444', 
              'total_generation': '#3b82f6',
              'renewable_generation': '#10b981',
              'grid_generation': '#f59e0b'}
    
    for col in columns:
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df[col],
                mode='lines',
                name=col.replace('_', ' ').title(),
                line=dict(color=colors.get(col, None), width=3)
            ))
    
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(size=20, color='#1a202c')
        ),
        xaxis_title=dict(text="<b>Time</b>", font=dict(size=14)),
        yaxis_title=dict(text=f"<b>{ylabel}</b>", font=dict(size=14)),
        height=400,
        hovermode='x unified',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(
            font=dict(size=13),
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='#e2e8f0',
            borderwidth=1
        )
    )
    
    return fig

def create_renewable_pie_chart(renewable_gen, grid_gen):
    fig = go.Figure(data=[go.Pie(
        labels=['<b>Renewable Energy</b>', '<b>Grid Energy</b>'],
        values=[renewable_gen, grid_gen],
        marker_colors=['#10b981', '#ef4444'],
        hole=0.45,
        textfont=dict(size=15, color='white'),
        textposition='inside'
    )])
    
    fig.update_layout(
        title=dict(
            text="<b>Energy Mix (Generation)</b>", font=dict(size=20, color='#1a202c')
        ),
        height=350,
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=13, color='#4a5568'),
        showlegend=True,
        legend=dict(
            font=dict(size=14),
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='#e2e8f0',
            borderwidth=1
        )
    )
    
    return fig

def create_forecast_chart(historical_df, forecast_df, metric_name):
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=historical_df['timestamp'].tail(48),
        y=historical_df[metric_name].tail(48),
        mode='lines',
        name='<b>Historical</b>',
        line=dict(color='#3b82f6', width=3)
    ))
    
    if 'ds' in forecast_df.columns:
        fig.add_trace(go.Scatter(
            x=forecast_df['ds'],
            y=forecast_df['yhat'],
            mode='lines',
            name='<b>Forecast</b>',
            line=dict(color='#7c3aed', width=3, dash='dash')
        ))
        
        if 'yhat_lower' in forecast_df.columns and 'yhat_upper' in forecast_df.columns:
            fig.add_trace(go.Scatter(
                x=forecast_df['ds'].tolist() + forecast_df['ds'].tolist()[::-1],
                y=forecast_df['yhat_upper'].tolist() + forecast_df['yhat_lower'].tolist()[::-1],
                fill='toself',
                fillcolor='rgba(124, 58, 237, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='<b>Confidence Interval</b>',
                showlegend=True
            ))
    
    fig.update_layout(
        title=dict(
            text=f"<b>{metric_name.replace('_', ' ').title()} - AI Forecast</b>",
            font=dict(size=20, color='#1a202c')
        ),
        xaxis_title=dict(text="<b>Time</b>", font=dict(size=14)),
        yaxis_title=dict(text="<b>kW</b>", font=dict(size=14)),
        height=400,
        hovermode='x unified',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12, color='#4a5568'),
        legend=dict(
            font=dict(size=13),
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='#e2e8f0',
            borderwidth=1
        )
    )
    
    return fig

st.markdown('<p class="main-header">⚡ GridSense: AI-Powered Smart Grid Energy Management</p>', 
            unsafe_allow_html=True)

if 'grid' not in st.session_state:
    st.session_state.grid = initialize_grid()
    st.session_state.weather_sim = WeatherSimulator()
    st.session_state.optimizer = EnergyOptimizer(prioritize_renewable=True)
    st.session_state.controller = GridController(st.session_state.optimizer)
    st.session_state.forecaster = EnergyForecaster(forecast_method='prophet')
    st.session_state.simple_forecaster = SimpleForecaster()
    
    historical_df, baseline_df = generate_initial_data(st.session_state.grid)
    st.session_state.historical_df = historical_df
    st.session_state.baseline_df = baseline_df
    
    if len(historical_df) > 48:
        try:
            st.session_state.forecaster.train_demand_forecaster(historical_df)
            st.session_state.forecaster.train_renewable_forecasters(historical_df)
            st.session_state.forecast_trained = True
        except:
            st.session_state.forecast_trained = False
            st.session_state.simple_forecaster.train_from_patterns(historical_df)
    else:
        st.session_state.forecast_trained = False
    
    st.session_state.simulation_running = False
    st.session_state.current_step = 0

with st.sidebar:
    st.header("⚙️ Grid Controls")
    
    st.subheader("Grid Configuration")
    
    with st.expander("Energy Sources", expanded=True):
        st.write("**Main Grid**")
        grid_capacity = st.slider("Capacity (kW)", 100, 1000, 500, key="grid_cap")
        
        st.write("**Solar Farm**")
        solar_capacity = st.slider("Capacity (kW)", 50, 500, 200, key="solar_cap")
        
        st.write("**Wind Turbines**")
        wind_capacity = st.slider("Capacity (kW)", 50, 300, 150, key="wind_cap")
    
    with st.expander("Consumer Load", expanded=True):
        load_multiplier = st.slider(
            "Load Multiplier", 
            0.5, 2.0, 1.0, 0.1,
            help="Adjust overall consumer demand"
        )
    
    st.subheader("Optimization Settings")
    prioritize_renewable = st.checkbox("Prioritize Renewable Energy", value=True)
    use_linear_prog = st.checkbox("Use Linear Programming", value=False, 
                                   help="Advanced optimization using linear programming")
    
    st.subheader("Simulation Control")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶️ Start", use_container_width=True):
            st.session_state.simulation_running = True
    with col2:
        if st.button("⏸️ Pause", use_container_width=True):
            st.session_state.simulation_running = False
    
    if st.button("🔄 Reset Simulation", use_container_width=True):
        st.session_state.grid.reset_simulation()
        st.session_state.current_step = 0
        st.session_state.simulation_running = False
        st.rerun()
    
    st.divider()
    st.subheader("Forecasting")
    forecast_enabled = st.checkbox("Enable AI Forecasting", value=True)
    forecast_periods = st.slider("Forecast Hours", 6, 48, 24)

st.session_state.grid.energy_sources[0].capacity = grid_capacity
st.session_state.grid.energy_sources[1].capacity = solar_capacity
st.session_state.grid.energy_sources[2].capacity = wind_capacity
st.session_state.optimizer.prioritize_renewable = prioritize_renewable

if st.session_state.simulation_running:
    conditions = st.session_state.weather_sim.update_weather(
        st.session_state.grid.current_time.hour
    )
    current_data = st.session_state.grid.simulate_step(conditions, load_multiplier)
    st.session_state.current_step += 1
    
    historical_df = st.session_state.grid.get_history_dataframe()
    if len(historical_df) > 0:
        st.session_state.historical_df = historical_df
else:
    if len(st.session_state.grid.history) > 0:
        current_data = st.session_state.grid.history[-1]
    else:
        conditions = st.session_state.weather_sim.update_weather(12)
        current_data = st.session_state.grid.simulate_step(conditions, load_multiplier)

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Live Dashboard", 
    "🔮 AI Forecasting", 
    "⚡ Optimization",
    "📈 Analytics"
])

with tab1:
    st.header("Real-Time Grid Status")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Demand",
            f"{current_data['total_demand']:.1f} kW",
            delta=None
        )
    
    with col2:
        st.metric(
            "Total Generation",
            f"{current_data['total_generation']:.1f} kW",
            delta=None
        )
    
    with col3:
        st.metric(
            "Renewable %",
            f"{current_data['renewable_percentage']:.1f}%",
            delta=None
        )
    
    with col4:
        st.metric(
            "Current Cost (₹)",
            f"₹{current_data['total_cost']:.2f}/hr",
            delta=None
        )
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(create_energy_flow_chart(current_data), use_container_width=True)
    
    with col2:
        st.plotly_chart(
            create_renewable_pie_chart(
                current_data['renewable_generation'],
                current_data['grid_generation']
            ),
            use_container_width=True
        )
    
    if len(st.session_state.historical_df) > 0:
        st.subheader("Historical Trends")
        st.plotly_chart(
            create_time_series_chart(
                st.session_state.historical_df.tail(48),
                ['total_demand', 'total_generation', 'renewable_generation'],
                "Energy Flow Over Time",
                "Energy (kW)"
            ),
            use_container_width=True
        )

with tab2:
    st.markdown('<p class="section-header">🔮 AI-Powered Energy Forecasting</p>', unsafe_allow_html=True)
    
    if forecast_enabled and st.session_state.forecast_trained:
        try:
            demand_forecast = st.session_state.forecaster.forecast_demand(forecast_periods)
            renewable_forecast = st.session_state.forecaster.forecast_renewable(forecast_periods)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.plotly_chart(
                    create_forecast_chart(
                        st.session_state.historical_df,
                        demand_forecast,
                        'total_demand'
                    ),
                    use_container_width=True
                )
            
            with col2:
                st.plotly_chart(
                    create_forecast_chart(
                        st.session_state.historical_df,
                        renewable_forecast,
                        'renewable_generation'
                    ),
                    use_container_width=True
                )
            
            st.subheader("Forecast Accuracy Metrics")
            
            if len(st.session_state.historical_df) > forecast_periods:
                recent_actual = st.session_state.historical_df['total_demand'].tail(forecast_periods)
                recent_predicted = demand_forecast['yhat'].head(len(recent_actual))
                
                accuracy = st.session_state.forecaster.calculate_forecast_accuracy(
                    recent_actual, recent_predicted
                )
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("MAE (Mean Absolute Error)", f"{accuracy['mae']:.2f} kW")
                with col2:
                    st.metric("RMSE (Root Mean Square Error)", f"{accuracy['rmse']:.2f} kW")
                with col3:
                    st.metric("MAPE (Mean Absolute % Error)", f"{accuracy['mape']:.2f}%")
        
        except Exception as e:
            st.warning("Forecasting model is training. Using pattern-based predictions.")
            st.info("AI forecasting will be available once sufficient data is collected.")
    else:
        st.info("📊 AI Forecasting is initializing. Enable 'Start' simulation to collect data for training the forecasting model.")
        st.write("The system uses Prophet (Facebook's time series forecasting algorithm) to predict:")
        st.write("- Energy demand patterns")
        st.write("- Renewable energy generation")
        st.write("- Load shifting opportunities")

with tab3:
    st.markdown('<p class="section-header">⚡ Energy Optimization</p>', unsafe_allow_html=True)
    
    available_sources = {
        source.name: {
            'type': source.source_type,
            'capacity': source.current_generation,
            'cost': source.cost_per_kwh
        }
        for source in st.session_state.grid.energy_sources
    }
    
    dispatch_result = st.session_state.controller.execute_dispatch(
        current_data['total_demand'],
        available_sources,
        use_linear_prog
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Optimized Dispatch")
        dispatch_df = pd.DataFrame([
            {
                'Source': name,
                'Type': available_sources[name]['type'],
                'Dispatched (kW)': amount,
                'Cost (₹/kWh)': available_sources[name]['cost']
            }
            for name, amount in dispatch_result['dispatch'].items()
        ])
        st.dataframe(dispatch_df, use_container_width=True, hide_index=True)
    
    with col2:
        st.subheader("Optimization Metrics")
        metrics = dispatch_result['metrics']
        
        st.metric("Renewable Utilization", f"{metrics['renewable_percentage']:.1f}%")
        st.metric("Total Cost", f"₹{metrics['total_cost']:.2f}/hr")
        st.metric("Unmet Demand", f"{metrics['unmet_demand']:.1f} kW")
    
    if len(st.session_state.baseline_df) > 0:
        st.subheader("Cost Comparison: Optimized vs Baseline")
        
        baseline_cost = st.session_state.baseline_df['cost'].mean()
        optimized_cost = current_data['total_cost']
        
        comparison = st.session_state.controller.get_efficiency_comparison(
            optimized_cost, baseline_cost
        )
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Baseline Cost (₹)", f"₹{comparison['baseline_cost']:.2f}/hr")
        with col2:
            st.metric("Optimized Cost (₹)", f"₹{comparison['optimized_cost']:.2f}/hr")
        with col3:
            st.metric(
                "Savings", 
                f"₹{comparison['cost_savings']:.2f}/hr",
                delta=f"{comparison['savings_percentage']:.1f}%"
            )

with tab4:
    st.markdown('<p class="section-header">📈 Performance Analytics</p>', unsafe_allow_html=True)
    
    if len(st.session_state.historical_df) > 0:
        df = st.session_state.historical_df
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Energy Efficiency Over Time")
            fig = px.line(
                df.tail(168),
                x='timestamp',
                y='renewable_percentage',
                title='<b>Renewable Energy Percentage</b>',
                labels={'renewable_percentage': 'Renewable %', 'timestamp': 'Time'}
            )
            fig.update_traces(line_color='#10b981', line_width=3)
            fig.update_layout(
                title=dict(font=dict(size=20, color='#1a202c')),
                xaxis_title=dict(text="<b>Time</b>", font=dict(size=14)),
                yaxis_title=dict(text="<b>Renewable %</b>", font=dict(size=14)),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(size=13, color='#4a5568')
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Cost Analysis")
            fig = px.line(
                df.tail(168),
                x='timestamp',
                y='total_cost',
                title='<b>Operational Cost Over Time</b>',
                labels={'total_cost': 'Cost (₹/hr)', 'timestamp': 'Time'}
            )
            fig.update_traces(line_color='#ef4444', line_width=3)
            fig.update_layout(
                title=dict(font=dict(size=20, color='#1a202c')),
                xaxis_title=dict(text="<b>Time</b>", font=dict(size=14)),
                yaxis_title=dict(text="<b>Cost (₹/hr)</b>", font=dict(size=14)),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(size=13, color='#4a5568')
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Summary Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_demand = df['total_demand'].mean()
            st.metric("Avg Demand", f"{avg_demand:.1f} kW")
        
        with col2:
            avg_renewable = df['renewable_percentage'].mean()
            st.metric("Avg Renewable %", f"{avg_renewable:.1f}%")
        
        with col3:
            total_energy = df['total_generation'].sum()
            st.metric("Total Energy Generated", f"{total_energy:.0f} kWh")
        
        with col4:
            total_cost = df['total_cost'].sum()
            st.metric("Total Cost", f"₹{total_cost:.2f}")
        
        st.subheader("Energy Balance Analysis")
        balance_df = pd.DataFrame({
            'Metric': ['Total Demand', 'Total Generation', 'Renewable', 'Grid'],
            'Value (kWh)': [
                df['total_demand'].sum(),
                df['total_generation'].sum(),
                df['renewable_generation'].sum(),
                df['grid_generation'].sum()
            ]
        })
        
        fig = px.bar(
            balance_df,
            x='Metric',
            y='Value (kWh)',
            title='<b>Cumulative Energy Balance</b>',
            color='Metric',
            color_discrete_map={
                'Total Demand': '#ef4444',
                'Total Generation': '#3b82f6',
                'Renewable': '#10b981',
                'Grid': '#f59e0b'
            }
        )
        fig.update_layout(
            title=dict(font=dict(size=20, color='#1a202c')),
            xaxis_title=dict(text="<b>Metric</b>", font=dict(size=14)),
            yaxis_title=dict(text="<b>Value (kWh)</b>", font=dict(size=14)),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(size=13, color='#4a5568')
        )
        st.plotly_chart(fig, use_container_width=True)

st.divider()

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"""
        <div class="info-card">
            <span style="font-size: 1.5rem;">🕐</span><br>
            <span style="font-size: 0.875rem; font-weight: 600; color: #4a5568; text-transform: uppercase;">Simulation Time</span><br>
            <span style="font-size: 1.25rem; font-weight: 700; color: #1a202c;">{st.session_state.grid.current_time.strftime('%Y-%m-%d %H:%M')}</span>
        </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown(f"""
        <div class="info-card">
            <span style="font-size: 1.5rem;">📊</span><br>
            <span style="font-size: 0.875rem; font-weight: 600; color: #4a5568; text-transform: uppercase;">Data Points Collected</span><br>
            <span style="font-size: 1.25rem; font-weight: 700; color: #1a202c;">{len(st.session_state.historical_df):,}</span>
        </div>
    """, unsafe_allow_html=True)
with col3:
    status_icon = "🟢" if st.session_state.simulation_running else "🔴"
    status_text = "Running" if st.session_state.simulation_running else "Paused"
    status_color = "#10b981" if st.session_state.simulation_running else "#ef4444" # Green for running, Red for paused
    st.markdown(f"""
        <div class="info-card">
            <span style="font-size: 1.5rem;">{status_icon}</span><br>
            <span style="font-size: 0.875rem; font-weight: 600; color: #4a5568; text-transform: uppercase;">System Status</span><br>
            <span style="font-size: 1.25rem; font-weight: 700; color: {status_color};">{status_text}</span>
        </div>
    """, unsafe_allow_html=True)

if st.session_state.simulation_running:
    st.rerun()
