import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
from datetime import timedelta
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(
    page_title="Smart Building Energy Efficiency System",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .status-good {
        background: linear-gradient(135deg, #4caf50 0%, #45a049 100%);
        padding: 15px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 10px;
    }
    .status-warning {
        background: linear-gradient(135deg, #ff9800 0%, #f57c00 100%);
        padding: 15px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 10px;
    }
    .status-critical {
        background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
        padding: 15px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 10px;
    }
    .info-box {
        background: #f8f9fa;
        border-left: 5px solid #3498db;
        padding: 20px;
        margin: 20px 0;
        border-radius: 5px;
    }
    .recommendation-box {
        background: linear-gradient(135deg, #9c27b0 0%, #7b1fa2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'building_data' not in st.session_state:
    st.session_state.building_data = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None

# Generate realistic building data
@st.cache_data
def generate_building_data():
    np.random.seed(42)
    start_date = datetime.datetime(2023, 1, 1)
    end_date = datetime.datetime(2024, 12, 31)
    date_range = pd.date_range(start=start_date, end=end_date, freq='H')
    data = []
    for dt in date_range:
        hour = dt.hour
        base_consumption = 200 + 300 * np.sin(np.pi * (hour - 6) / 12) if 6 <= hour <= 18 else 150
        weekday_factor = 1.0 if dt.weekday() < 5 else 0.6
        seasonal_factor = 1.3 if dt.month in [6, 7, 8, 12, 1, 2] else 1.0
        if dt.weekday() < 5 and 8 <= hour <= 18:
            occupancy = np.random.uniform(0.3, 0.9)
        elif dt.weekday() < 5 and (6 <= hour < 8 or 18 < hour <= 20):
            occupancy = np.random.uniform(0.1, 0.4)
        else:
            occupancy = np.random.uniform(0.05, 0.2)
        temperature = 25 + 15 * np.sin(2 * np.pi * dt.dayofyear / 365) + np.random.normal(0, 3)
        humidity = 60 + 20 * np.sin(2 * np.pi * dt.dayofyear / 365 + np.pi/4) + np.random.normal(0, 5)
        occupancy_factor = 0.5 + 0.5 * occupancy
        temp_factor = 1.0 + 0.3 * abs(temperature - 22) / 10
        energy_consumption = (base_consumption * weekday_factor * seasonal_factor *
                              occupancy_factor * temp_factor + np.random.normal(0, 20))
        hvac_power = energy_consumption * 0.6
        lighting_power = energy_consumption * 0.2
        equipment_power = energy_consumption * 0.2
        data.append({
            'datetime': dt,
            'hour': hour,
            'day_of_week': dt.weekday(),
            'month': dt.month,
            'is_weekend': dt.weekday() >= 5,
            'occupancy_rate': occupancy,
            'temperature': temperature,
            'humidity': humidity,
            'energy_consumption': max(0, energy_consumption),
            'hvac_power': max(0, hvac_power),
            'lighting_power': max(0, lighting_power),
            'equipment_power': max(0, equipment_power)
        })
    return pd.DataFrame(data)

# Train energy prediction model
@st.cache_data
def train_energy_model(df):
    features = ['hour', 'day_of_week', 'month', 'occupancy_rate',
                'temperature', 'humidity', 'is_weekend']
    X = df[features]
    y = df['energy_consumption']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    return model, {'MAE': mae, 'RMSE': rmse, 'R²': r2}, X_test, y_test, y_pred

# Generate recommendations
def generate_recommendations(current_consumption, predicted_consumption, occupancy, temperature):
    recommendations = []
    if current_consumption > predicted_consumption * 1.1:
        recommendations.append("🔴 Current consumption is 10% above predicted. Check HVAC settings.")
    if occupancy < 0.3 and current_consumption > 300:
        recommendations.append("⚠️ Low occupancy but high consumption. Consider reducing HVAC in unused zones.")
    if abs(temperature - 22) > 3:
        recommendations.append(f"🌡️ Optimize temperature settings. Current: {temperature:.1f}°C")
    if len(recommendations) == 0:
        recommendations.append("✅ Energy consumption is within optimal range.")
    return recommendations

# Sidebar
st.sidebar.title("🏢 Smart Building Energy System")
page = st.sidebar.selectbox(
    "Select Page",
    ["🏠 Dashboard", "📊 Analytics", "🤖 AI Predictions", "⚙️ System Control", "📋 Reports"]
)

# Load data
if st.session_state.building_data is None:
    with st.spinner("Loading building data..."):
        st.session_state.building_data = generate_building_data()
        st.session_state.model, metrics, X_test, y_test, y_pred = train_energy_model(st.session_state.building_data)
        st.session_state.predictions = {'X_test': X_test, 'y_test': y_test, 'y_pred': y_pred, 'metrics': metrics}

df = st.session_state.building_data

# Main content based on selected page
if page == "🏠 Dashboard":
    st.markdown('<h1 class="main-header">🏢 Smart Building Energy Dashboard</h1>', unsafe_allow_html=True)
    current_time = datetime.datetime.now()
    current_hour = current_time.hour
    current_occupancy = np.random.uniform(0.2, 0.8) if 8 <= current_hour <= 18 else np.random.uniform(0.05, 0.3)
    current_temp = 22 + np.random.normal(0, 2)
    current_consumption = 250 + 200 * current_occupancy + np.random.normal(0, 30)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Current Consumption</h3>
            <h2>{current_consumption:.1f} kWh</h2>
            <p>Real-time usage</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Occupancy Rate</h3>
            <h2>{current_occupancy:.1%}</h2>
            <p>Current building occupancy</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Temperature</h3>
            <h2>{current_temp:.1f}°C</h2>
            <p>Average indoor temp</p>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        savings = 20
        st.markdown(f"""
        <div class="metric-card">
            <h3>Energy Savings</h3>
            <h2>{savings}%</h2>
            <p>Monthly reduction</p>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("### 🔧 System Status")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="status-good">
            <h4>HVAC System</h4>
            <p>✅ Operating Normally</p>
            <p>Efficiency: 92%</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="status-warning">
            <h4>Lighting System</h4>
            <p>⚠️ Minor Issues</p>
            <p>Zone 3 needs attention</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="status-good">
            <h4>AI Optimizer</h4>
            <p>✅ Active</p>
            <p>Last updated: 2 min ago</p>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("### 📈 Real-Time Energy Consumption")
    recent_data = df.tail(168)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=recent_data['datetime'],
        y=recent_data['energy_consumption'],
        mode='lines',
        name='Energy Consumption',
        line=dict(color='#3498db', width=2)
    ))
    fig.update_layout(
        title="Energy Consumption (Last 7 Days)",
        xaxis_title="Time",
        yaxis_title="Energy Consumption (kWh)",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("### 💡 AI Recommendations")
    recommendations = generate_recommendations(
        current_consumption,
        current_consumption * 0.9,
        current_occupancy,
        current_temp
    )
    for rec in recommendations:
        st.markdown(f"""
        <div class="recommendation-box">
            <p>{rec}</p>
        </div>
        """, unsafe_allow_html=True)

elif page == "📊 Analytics":
    st.markdown('<h1 class="main-header">📊 Energy Analytics</h1>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=datetime.date(2024, 1, 1))
    with col2:
        end_date = st.date_input("End Date", value=datetime.date(2024, 12, 31))
    mask = (df['datetime'].dt.date >= start_date) & (df['datetime'].dt.date <= end_date)
    filtered_df = df[mask]
    col1, col2 = st.columns(2)
    with col1:
        daily_pattern = filtered_df.groupby('hour')['energy_consumption'].mean().reset_index()
        fig = px.line(daily_pattern, x='hour', y='energy_consumption',
                     title='Average Energy Consumption by Hour of Day')
        fig.update_traces(line_color='#e74c3c')
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        avg_hvac = filtered_df['hvac_power'].mean()
        avg_lighting = filtered_df['lighting_power'].mean()
        avg_equipment = filtered_df['equipment_power'].mean()
        fig = px.pie(
            values=[avg_hvac, avg_lighting, avg_equipment],
            names=['HVAC', 'Lighting', 'Equipment'],
            title='Energy Consumption Breakdown'
        )
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("### 📈 Occupancy vs Energy Consumption")
    fig = px.scatter(
        filtered_df.sample(1000),
        x='occupancy_rate',
        y='energy_consumption',
        color='temperature',
        title='Energy Consumption vs Occupancy Rate',
        labels={'occupancy_rate': 'Occupancy Rate', 'energy_consumption': 'Energy Consumption (kWh)'}
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("### 📅 Monthly Energy Trends")
    monthly_data = filtered_df.groupby('month').agg({
        'energy_consumption': 'mean',
        'occupancy_rate': 'mean',
        'temperature': 'mean'
    }).reset_index()
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Monthly Energy Consumption', 'Monthly Occupancy Rate',
                       'Monthly Temperature', 'Energy Efficiency'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    fig.add_trace(go.Bar(x=monthly_data['month'], y=monthly_data['energy_consumption'],
                        name='Energy Consumption', marker_color='#3498db'), row=1, col=1)
    fig.add_trace(go.Scatter(x=monthly_data['month'], y=monthly_data['occupancy_rate'],
                           mode='lines+markers', name='Occupancy Rate', line=dict(color='#e74c3c')), row=1, col=2)
    fig.add_trace(go.Scatter(x=monthly_data['month'], y=monthly_data['temperature'],
                           mode='lines+markers', name='Temperature', line=dict(color='#f39c12')), row=2, col=1)
    efficiency = monthly_data['energy_consumption'] / monthly_data['occupancy_rate']
    fig.add_trace(go.Bar(x=monthly_data['month'], y=efficiency,
                        name='Energy per Occupancy', marker_color='#9b59b6'), row=2, col=2)
    fig.update_layout(height=800, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

elif page == "🤖 AI Predictions":
    st.markdown('<h1 class="main-header">🤖 AI Energy Predictions</h1>', unsafe_allow_html=True)
    st.markdown("### 📊 Model Performance")
    col1, col2, col3 = st.columns(3)
    metrics = st.session_state.predictions['metrics']
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>R² Score</h3>
            <h2>{metrics['R²']:.3f}</h2>
            <p>Model accuracy</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>MAE</h3>
            <h2>{metrics['MAE']:.1f}</h2>
            <p>Mean Absolute Error</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>RMSE</h3>
            <h2>{metrics['RMSE']:.1f}</h2>
            <p>Root Mean Square Error</p>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("### 📈 Predictions vs Actual")
    y_test = st.session_state.predictions['y_test']
    y_pred = st.session_state.predictions['y_pred']
    fig = go.Figure()
    sample_indices = np.random.choice(len(y_test), 500, replace=False)
    fig.add_trace(go.Scatter(
        x=y_test.iloc[sample_indices],
        y=y_pred[sample_indices],
        mode='markers',
        name='Predictions',
        marker=dict(color='#3498db', size=8, opacity=0.6)
    ))
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        name='Perfect Prediction',
        line=dict(color='#e74c3c', dash='dash')
    ))
    fig.update_layout(
        title="Predicted vs Actual Energy Consumption",
        xaxis_title="Actual Energy Consumption (kWh)",
        yaxis_title="Predicted Energy Consumption (kWh)",
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("### 🎯 Feature Importance")
    feature_importance = st.session_state.model.feature_importances_
    features = ['Hour', 'Day of Week', 'Month', 'Occupancy Rate',
                'Temperature', 'Humidity', 'Is Weekend']
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': feature_importance
    }).sort_values('Importance', ascending=True)
    fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                 title='Feature Importance for Energy Prediction')
    fig.update_traces(marker_color='#9b59b6')
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("### 🔮 Future Energy Predictions")
    col1, col2, col3 = st.columns(3)
    with col1:
        pred_hour = st.slider("Hour of Day", 0, 23, 12)
        pred_day = st.selectbox("Day of Week",
                               ['Monday', 'Tuesday', 'Wednesday', 'Thursday',
                                'Friday', 'Saturday', 'Sunday'])
        pred_month = st.selectbox("Month",
                                 ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    with col2:
        pred_occupancy = st.slider("Occupancy Rate", 0.0, 1.0, 0.5)
        pred_temperature = st.slider("Temperature (°C)", 10.0, 40.0, 22.0)
        pred_humidity = st.slider("Humidity (%)", 20.0, 80.0, 50.0)
    with col3:
        pred_weekend = st.checkbox("Weekend", value=False)
        if st.button("🔮 Make Prediction"):
            pred_data = np.array([[
                pred_hour,
                ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'].index(pred_day),
                ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'].index(pred_month) + 1,
                pred_occupancy,
                pred_temperature,
                pred_humidity,
                pred_weekend
            ]])
            prediction = st.session_state.model.predict(pred_data)[0]
            st.success(f"Predicted Energy Consumption: {prediction:.1f} kWh")
            if prediction > 400:
                st.warning("⚠️ High energy consumption predicted. Consider:")
                st.markdown("- Reducing HVAC setpoint by 1-2°C")
                st.markdown("- Implementing occupancy-based lighting")
                st.markdown("- Pre-cooling during off-peak hours")

elif page == "⚙️ System Control":
    st.markdown('<h1 class="main-header">⚙️ System Control Center</h1>', unsafe_allow_html=True)
    st.markdown("### 🌡️ HVAC System Controls")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Temperature Settings")
        temp_setpoint = st.slider("Temperature Setpoint (°C)", 18.0, 26.0, 22.0)
        hvac_mode = st.selectbox("HVAC Mode", ["Auto", "Heating", "Cooling", "Fan Only"])
        hvac_schedule = st.checkbox("Enable Smart Scheduling", value=True)
        if st.button("Apply HVAC Settings"):
            st.success(f"✅ HVAC settings updated: {temp_setpoint}°C, {hvac_mode} mode")
    with col2:
        st.markdown("#### Zone Controls")
        zones = ["Zone 1 (Lobby)", "Zone 2 (Offices)", "Zone 3 (Conference)", "Zone 4 (Cafeteria)"]
        for zone in zones:
            col_a, col_b = st.columns([2, 1])
            with col_a:
                st.write(zone)
            with col_b:
                st.selectbox(f"Status", ["On", "Off", "Auto"], key=f"zone_{zone}")
    st.markdown("### 💡 Lighting System Controls")
    col1, col2 = st.columns(2)
    with col1:
        brightness = st.slider("Overall Brightness (%)", 0, 100, 75)
        auto_dimming = st.checkbox("Enable Auto-Dimming", value=True)
        occupancy_lighting = st.checkbox("Occupancy-Based Lighting", value=True)
        if st.button("Apply Lighting Settings"):
            st.success(f"✅ Lighting updated: {brightness}% brightness")
    with col2:
        st.markdown("#### Schedule Override")
        override_zones = st.multiselect("Override Zones", zones)
        override_duration = st.selectbox("Override Duration",
                                       ["1 hour", "2 hours", "4 hours", "Until next schedule"])
        if st.button("Apply Override"):
            if override_zones:
                st.success(f"✅ Override applied to {len(override_zones)} zones")
    st.markdown("### 🔧 Energy Optimization")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Optimization Settings")
        optimization_mode = st.selectbox("Optimization Mode",
                                       ["Maximum Savings", "Balanced", "Comfort Priority"])
        auto_optimization = st.checkbox("Enable Auto-Optimization", value=True)
        if st.button("Run Optimization"):
            with st.spinner("Optimizing energy settings..."):
                time.sleep(2)
                st.success("✅ Optimization complete! Estimated savings: 15%")
    with col2:
        st.markdown("#### System Status")
        systems = [
            ("HVAC System", "✅ Normal", "good"),
            ("Lighting System", "⚠️ Zone 3 Issue", "warning"),
            ("Energy Meter", "✅ Normal", "good"),
            ("AI Controller", "✅ Active", "good")
        ]
        for system, status, level in systems:
            if level == "good":
                class_name = "status-good"
            elif level == "warning":
                class_name = "status-warning"
            else:
                class_name = "status-critical"
            st.markdown(f"""
            <div class="{class_name}">
                <strong>{system}</strong><br>
                {status}
            </div>
            """, unsafe_allow_html=True)

elif page == "📋 Reports":
    st.markdown('<h1 class="main-header">📋 Energy Reports</h1>', unsafe_allow_html=True)
    report_type = st.selectbox("Select Report Type",
                              ["Daily Summary", "Weekly Analysis", "Monthly Report", "Annual Overview"])
    if report_type == "Daily Summary":
        st.markdown("### 📅 Daily Energy Summary")
        selected_date = st.date_input("Select Date", value=datetime.date.today())
        daily_data = df[df['datetime'].dt.date == selected_date]
        if not daily_data.empty:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                total_consumption = daily_data['energy_consumption'].sum()
                st.metric("Total Consumption", f"{total_consumption:.1f} kWh")
            with col2:
                avg_occupancy = daily_data['occupancy_rate'].mean()
                st.metric("Average Occupancy", f"{avg_occupancy:.1%}")
            with col3:
                peak_consumption = daily_data['energy_consumption'].max()
                st.metric("Peak Consumption", f"{peak_consumption:.1f} kWh")
            with col4:
                avg_temp = daily_data['temperature'].mean()
                st.metric("Average Temperature", f"{avg_temp:.1f}°C")
            fig = px.line(daily_data, x='datetime', y='energy_consumption',
                         title=f'Energy Consumption on {selected_date}')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No data available for selected date")
    elif report_type == "Weekly Analysis":
        st.markdown("### 📊 Weekly Energy Analysis")
        end_date = datetime.date.today()
        start_date = end_date - timedelta(days=7)
        weekly_data = df[(df['datetime'].dt.date >= start_date) &
                        (df['datetime'].dt.date <= end_date)]
        daily_summary = weekly_data.groupby(weekly_data['datetime'].dt.date).agg({
            'energy_consumption': ['sum', 'mean', 'max'],
            'occupancy_rate': 'mean',
            'temperature': 'mean'
        }).round(2)
        daily_summary.columns = ['Total kWh', 'Average kWh', 'Peak kWh', 'Avg Occupancy', 'Avg Temp']
        st.dataframe(daily_summary)
        st.markdown("#### Weekly Consumption Trend")
        fig = px.line(
            daily_summary.reset_index(),
            x='datetime',
            y='Total kWh',
            title='Total Daily Energy Consumption (Last 7 Days)'
        )
        st.plotly_chart(fig, use_container_width=True)
    elif report_type == "Monthly Report":
        st.markdown("### 📆 Monthly Energy Report")
        months = list(range(1, 13))
        years = df['datetime'].dt.year.unique()
        col1, col2 = st.columns(2)
        with col1:
            selected_month = st.selectbox("Month", months, format_func=lambda x: datetime.date(1900, x, 1).strftime('%B'))
        with col2:
            selected_year = st.selectbox("Year", sorted(years, reverse=True))
        monthly_data = df[(df['datetime'].dt.month == selected_month) & (df['datetime'].dt.year == selected_year)]
        if not monthly_data.empty:
            total_consumption = monthly_data['energy_consumption'].sum()
            avg_daily = monthly_data.groupby(monthly_data['datetime'].dt.date)['energy_consumption'].sum().mean()
            peak_day = monthly_data.groupby(monthly_data['datetime'].dt.date)['energy_consumption'].sum().idxmax()
            avg_occupancy = monthly_data['occupancy_rate'].mean()
            avg_temp = monthly_data['temperature'].mean()
            st.markdown(f"""
                <div class="info-box">
                    <b>Total Consumption:</b> {total_consumption:.1f} kWh<br>
                    <b>Average Daily Consumption:</b> {avg_daily:.1f} kWh<br>
                    <b>Peak Day:</b> {peak_day}<br>
                    <b>Average Occupancy:</b> {avg_occupancy:.1%}<br>
                    <b>Average Temperature:</b> {avg_temp:.1f}°C
                </div>
            """, unsafe_allow_html=True)
            st.markdown("#### Daily Consumption Trend")
            daily_trend = monthly_data.groupby(monthly_data['datetime'].dt.date)['energy_consumption'].sum().reset_index()
            fig = px.bar(daily_trend, x='datetime', y='energy_consumption', title='Daily Energy Consumption')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No data available for selected month.")
    elif report_type == "Annual Overview":
        st.markdown("### 📅 Annual Energy Overview")
        years = df['datetime'].dt.year.unique()
        selected_year = st.selectbox("Year", sorted(years, reverse=True))
        annual_data = df[df['datetime'].dt.year == selected_year]
        if not annual_data.empty:
            monthly_summary = annual_data.groupby(annual_data['datetime'].dt.month).agg({
                'energy_consumption': 'sum',
                'occupancy_rate': 'mean',
                'temperature': 'mean'
            }).reset_index()
            monthly_summary['Month'] = monthly_summary['datetime'].apply(lambda x: datetime.date(1900, x, 1).strftime('%B'))
            st.markdown("#### Annual Summary Table")
            st.dataframe(monthly_summary[['Month', 'energy_consumption', 'occupancy_rate', 'temperature']].rename(
                columns={
                    'energy_consumption': 'Total kWh',
                    'occupancy_rate': 'Avg Occupancy',
                    'temperature': 'Avg Temp (°C)'
                }
            ))
            st.markdown("#### Monthly Consumption Trend")
            fig = px.line(monthly_summary, x='Month', y='energy_consumption', markers=True,
                          title='Monthly Energy Consumption')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No data available for selected year.")
