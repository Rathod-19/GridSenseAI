# GridSense - AI-powered Smart Grid Energy Management

## Overview

GridSense is an AI-powered smart grid energy management system that simulates and optimizes energy distribution across a micro-grid network. The application uses machine learning for demand forecasting and optimization algorithms to balance energy supply and demand while prioritizing renewable energy sources. Built with Python and Streamlit, it provides real-time visualization of energy flow, generation, and consumption patterns across multiple energy sources (grid, solar, wind) and consumer types (residential, commercial, industrial).

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
**Technology**: Streamlit web framework with professional UI/UX design
- **Decision**: Streamlit chosen for rapid development and built-in data visualization capabilities
- **Rationale**: Provides interactive dashboard without requiring separate frontend/backend development
- **Components**: 
  - Real-time metric displays for grid status with elevated card design
  - Interactive visualizations using Plotly for energy flow and forecasting
  - Configuration sidebar for simulation parameters
  - Professional typography system with hierarchical font sizes (Inter font family)
  - Consistent color scheme across all UI elements
  - Mobile-friendly responsive design
- **UI/UX Improvements (Nov 2024)**:
  - **Typography Hierarchy**: 
    - Main header: 3rem gradient text (purple gradient #667eea to #764ba2)
    - Section headers: 1.875rem bold
    - Subsection headers: 1.5rem semi-bold
    - Chart titles: 20px bold
    - Metric labels: 0.875rem uppercase with letter spacing
    - Metric values: 2rem bold
  - **Color System**:
    - Renewable energy: #10b981 (emerald green)
    - Grid/Demand: #ef4444 (red)
    - Total generation: #3b82f6 (blue)
    - Grid source: #f59e0b (amber)
    - Solar: #fbbf24 (yellow-amber)
    - Wind: #60a5fa (sky blue)
    - Primary accent: #667eea to #764ba2 gradient (purple)
  - **Component Styling**:
    - Metric cards with subtle shadows, borders, and hover effects
    - Tab navigation with gradient background for active states
    - Chart enhancements: bold titles, larger fonts, consistent colors, transparent backgrounds
    - Custom info cards with gradient backgrounds and icon-label-value hierarchy
    - Professional sidebar with gradient background
  - **Visual Polish**:
    - Improved contrast for better readability
    - Consistent spacing and padding throughout
    - Modern glassmorphism-inspired card designs
    - Smooth transitions and hover states

### Simulation Engine
**Core Module**: Grid simulation with object-oriented design
- **Decision**: Separate classes for EnergySource and Consumer entities
- **Rationale**: Enables flexible grid configuration and realistic modeling of different energy sources and consumption patterns
- **Key Features**:
  - Time-based energy generation modeling (solar follows day/night cycles, wind depends on speed)
  - Consumer demand modeling with hourly and day-of-week patterns
  - Weather simulation affecting renewable generation (cloud cover for solar, wind speed for wind)
  - Historical data generation for training ML models

### AI/ML Forecasting System
**Technology**: Prophet and ARIMA for time series forecasting
- **Decision**: Dual forecasting approach with pluggable models
- **Rationale**: 
  - Prophet handles seasonality and trends well for energy demand patterns
  - ARIMA provides alternative statistical approach
  - Flexibility to switch between methods based on data characteristics
- **Architecture**: 
  - Separate forecasters for demand and renewable generation
  - Training on historical simulation data
  - Multi-period ahead forecasting (default 24 hours)

### Optimization Engine
**Approaches**: Rule-based and linear programming optimization
- **Decision**: Hybrid optimization strategy
- **Rationale**:
  - Rule-based: Simple, fast, prioritizes renewable sources by default
  - Linear programming: Optimal solution considering costs and constraints
  - Allows comparison between heuristic and optimal approaches
- **Objective**: Minimize energy costs while meeting demand and prioritizing renewable sources
- **Constraints**: Source capacity limits, demand fulfillment requirements

### Data Management
**Pattern**: In-memory data generation and caching
- **Decision**: Streamlit's caching decorators for grid state and historical data
- **Rationale**: 
  - No persistent storage needed for simulation-based application
  - Caching prevents regenerating expensive historical data on UI interactions
  - Grid state maintained in session for consistency
- **Data Flow**:
  - HistoricalDataGenerator creates training datasets
  - WeatherSimulator provides conditions for generation calculations
  - Pandas DataFrames for time-series data manipulation

### Component Integration
**Pattern**: Modular architecture with clear separation of concerns
- **Modules**:
  - `grid_simulation.py`: Core simulation entities and physics
  - `forecasting.py`: ML models for prediction
  - `optimization.py`: Energy dispatch optimization
  - `data_generator.py`: Historical data and weather simulation
  - `app.py`: Streamlit UI and orchestration
- **Benefits**: 
  - Easy testing of individual components
  - Flexibility to swap implementations (e.g., different forecasting models)
  - Clear dependencies between layers

## External Dependencies

### Machine Learning Libraries
- **Prophet**: Facebook's time series forecasting library for demand prediction
- **statsmodels**: ARIMA implementation for alternative forecasting approach
- **NumPy**: Numerical computations for simulation physics and data generation
- **Pandas**: Time-series data manipulation and analysis

### Optimization Libraries
- **SciPy**: Linear programming solver (`linprog`) for optimal energy dispatch

### Visualization and UI
- **Streamlit**: Web application framework and hosting platform
- **Plotly**: Interactive charts for energy flow visualization (graph_objects and express)
- **Matplotlib/Seaborn**: Additional plotting capabilities (referenced in architecture doc)

### Python Standard Library
- **datetime/timedelta**: Time-based simulation and data generation
- **warnings**: Suppressing model training warnings for cleaner output

### Development Environment
- **Python 3.x**: Runtime environment

**Note**: No external databases, authentication services, or real-time grid APIs are used. The system is self-contained with simulated data generation.