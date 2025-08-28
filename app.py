import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# Add src directory to path
sys.path.append('src')

from segmentation import CustomerSegmentation
from forecasting import DemandForecasting

# Page configuration
st.set_page_config(
    page_title="Consumer Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 5px solid #1f77b4;
}
.insight-box {
    background-color: #e8f4fd;
    padding: 1rem;
    border-radius: 0.5rem;
    border: 1px solid #1f77b4;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and cache the datasets"""
    try:
        raw_data = pd.read_csv('data/synthetic_data.csv')
        processed_data = pd.read_csv('data/processed_data.csv')
        return raw_data, processed_data
    except FileNotFoundError:
        st.error("Data files not found! Please run the data generator first.")
        st.code("python src/data_generator.py")
        st.stop()

def main():
    # Header
    st.markdown('<h1 class="main-header">🛍️ Consumer Analytics Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("**Data Science Project for P&G Internship Application**")
    
    # Load data
    raw_data, processed_data = load_data()
    
    # Sidebar
    st.sidebar.title("📋 Navigation")
    page = st.sidebar.selectbox(
        "Choose Analysis",
        ["📊 Overview", "👥 Customer Segmentation", "📈 Demand Forecasting", "💡 Business Insights"]
    )
    
    if page == "📊 Overview":
        show_overview(raw_data, processed_data)
    elif page == "👥 Customer Segmentation":
        show_segmentation(processed_data)
    elif page == "📈 Demand Forecasting":
        show_forecasting(raw_data)
    elif page == "💡 Business Insights":
        show_insights(raw_data, processed_data)

def show_overview(raw_data, processed_data):
    """Display overview dashboard"""
    st.header("📊 Business Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_revenue = raw_data['total_amount'].sum()
        st.metric("Total Revenue", f"${total_revenue:,.2f}")
    
    with col2:
        total_customers = raw_data['customer_id'].nunique()
        st.metric("Total Customers", f"{total_customers:,}")
    
    with col3:
        total_transactions = len(raw_data)
        st.metric("Total Transactions", f"{total_transactions:,}")
    
    with col4:
        avg_order_value = raw_data['total_amount'].mean()
        st.metric("Avg Order Value", f"${avg_order_value:.2f}")
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Revenue by Category")
        category_revenue = raw_data.groupby('category')['total_amount'].sum().reset_index()
        fig = px.pie(category_revenue, values='total_amount', names='category', 
                    title="Revenue Distribution by Product Category")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Sales Channel Performance")
        channel_data = raw_data.groupby('channel').agg({
            'total_amount': 'sum',
            'transaction_id': 'count'
        }).reset_index()
        fig = px.bar(channel_data, x='channel', y='total_amount',
                    title="Revenue by Sales Channel")
        st.plotly_chart(fig, use_container_width=True)
    
    # Time series
    st.subheader("Daily Sales Trend")
    raw_data['transaction_date'] = pd.to_datetime(raw_data['transaction_date'])
    daily_sales = raw_data.groupby('transaction_date')['total_amount'].sum().reset_index()
    fig = px.line(daily_sales, x='transaction_date', y='total_amount',
                 title="Daily Revenue Trend")
    st.plotly_chart(fig, use_container_width=True)
    
    # Demographics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Customer Age Distribution")
        fig = px.histogram(processed_data, x='age', nbins=20, 
                          title="Customer Age Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Gender Distribution")
        gender_counts = processed_data['gender'].value_counts()
        fig = px.pie(values=gender_counts.values, names=gender_counts.index,
                    title="Customer Gender Distribution")
        st.plotly_chart(fig, use_container_width=True)

def show_segmentation(processed_data):
    """Display customer segmentation analysis"""
    st.header("👥 Customer Segmentation Analysis")
    
    # Initialize segmentation
    with st.spinner("Performing customer segmentation..."):
        segmenter = CustomerSegmentation(processed_data.copy())
        segmenter.prepare_features()
        
        # Clustering
        n_clusters = st.sidebar.slider("Number of Segments", 2, 8, 4)
        segmented_data = segmenter.perform_clustering(n_clusters=n_clusters)
        segments = segmenter.analyze_segments()
    
    # Segment overview
    st.subheader("📋 Segment Overview")
    
    # Display segment summary
    summary_df = segments['summary']
    st.dataframe(summary_df.round(2), use_container_width=True)
    
    # Segment interpretations
    st.subheader("🎯 Segment Profiles")
    
    cols = st.columns(min(n_clusters, 3))
    for i, (cluster, info) in enumerate(segments['interpretations'].items()):
        with cols[i % 3]:
            st.markdown(f"""
            <div class="metric-card">
                <h4>Segment {cluster}: {info['type']}</h4>
                <p>{info['description']}</p>
                <ul>
                    <li><strong>Size:</strong> {info['size']} customers</li>
                    <li><strong>Avg Recency:</strong> {info['avg_recency']:.1f} days</li>
                    <li><strong>Avg Frequency:</strong> {info['avg_frequency']:.1f} purchases</li>
                    <li><strong>Avg Monetary:</strong> ${info['avg_monetary']:.2f}</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    # Visualizations
    st.subheader("📊 Segment Visualizations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 3D scatter plot
        fig = px.scatter_3d(
            segmented_data, 
            x='recency', 
            y='frequency', 
            z='monetary',
            color='cluster',
            title='Customer Segments - RFM Analysis',
            labels={'cluster': 'Segment'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Segment distribution
        segment_counts = segmented_data['cluster'].value_counts().sort_index()
        fig = px.pie(
            values=segment_counts.values,
            names=[f'Segment {i}' for i in segment_counts.index],
            title='Customer Segment Distribution'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Marketing recommendations
    st.subheader("💡 Marketing Recommendations")
    recommendations = segmenter.get_segment_recommendations()
    
    for cluster, rec in recommendations.items():
        with st.expander(f"Segment {cluster} - {rec['strategy']}"):
            st.write(f"**Strategy:** {rec['strategy']}")
            st.write("**Recommended Actions:**")
            for action in rec['actions']:
                st.write(f"• {action}")

def show_forecasting(raw_data):
    """Display demand forecasting analysis"""
    st.header("📈 Demand Forecasting")
    
    # Initialize forecasting
    with st.spinner("Preparing forecasting models..."):
        forecaster = DemandForecasting(raw_data.copy())
        daily_sales = forecaster.prepare_time_series()
    
    # Forecast parameters
    st.sidebar.subheader("Forecast Settings")
    days_ahead = st.sidebar.slider("Days to Forecast", 7, 90, 30)
    
    # Run forecasting
    with st.spinner("Running forecasting models..."):
        comparison = forecaster.compare_models(days_ahead=days_ahead)
    
    # Model comparison
    st.subheader("🔮 Model Performance Comparison")
    st.dataframe(comparison.round(4), use_container_width=True)
    
    # Best model highlight
    best_model = comparison.iloc[0]['Model']
    st.success(f"🏆 Best performing model: **{best_model}** with {comparison.iloc[0]['Accuracy']:.1%} accuracy")
    
    # Forecast visualization
    st.subheader("📊 Forecast Results")
    
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=daily_sales['date'],
        y=daily_sales['revenue'],
        mode='lines',
        name='Historical Revenue',
        line=dict(color='blue', width=2)
    ))
    
    # Add forecasts
    colors = ['red', 'green', 'orange']
    for i, (model_name, results) in enumerate(forecaster.forecasts.items()):
        forecast_data = results['forecast']
        
        if 'ds' in forecast_data.columns:  # Prophet format
            future_data = forecast_data[forecast_data['ds'] > daily_sales['date'].max()]
            fig.add_trace(go.Scatter(
                x=future_data['ds'],
                y=future_data['yhat'],
                mode='lines',
                name=f'{model_name.title()} Forecast',
                line=dict(color=colors[i % len(colors)], width=2, dash='dash')
            ))
        else:  # Other models
            fig.add_trace(go.Scatter(
                x=forecast_data['date'],
                y=forecast_data['predicted_revenue'],
                mode='lines',
                name=f'{model_name.title()} Forecast',
                line=dict(color=colors[i % len(colors)], width=2, dash='dash')
            ))
    
    fig.update_layout(
        title=f'Revenue Forecasting - Next {days_ahead} Days',
        xaxis_title='Date',
        yaxis_title='Revenue ($)',
        hovermode='x unified',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Seasonal patterns
    st.subheader("📅 Seasonal Patterns")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Weekly pattern
        weekly_pattern = daily_sales.groupby('day_of_week')['revenue'].mean().reset_index()
        weekly_pattern['day_name'] = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        fig = px.bar(weekly_pattern, x='day_name', y='revenue',
                    title="Average Revenue by Day of Week")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Monthly pattern
        monthly_pattern = daily_sales.groupby('month')['revenue'].mean().reset_index()
        fig = px.bar(monthly_pattern, x='month', y='revenue',
                    title="Average Revenue by Month")
        st.plotly_chart(fig, use_container_width=True)

def show_insights(raw_data, processed_data):
    """Display business insights and recommendations"""
    st.header("💡 Business Insights & Recommendations")
    
    # Key insights
    st.subheader("🔍 Key Findings")
    
    # Revenue insights
    total_revenue = raw_data['total_amount'].sum()
    avg_customer_value = processed_data['monetary'].mean()
    top_category = raw_data.groupby('category')['total_amount'].sum().idxmax()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="insight-box">
            <h4>💰 Revenue Performance</h4>
            <p>Total revenue of <strong>${total_revenue:,.2f}</strong> generated from {len(raw_data):,} transactions</p>
            <p>Average customer lifetime value: <strong>${avg_customer_value:.2f}</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="insight-box">
            <h4>🏆 Top Performing Category</h4>
            <p><strong>{top_category}</strong> is the highest revenue generating category</p>
            <p>Focus marketing efforts on this category for maximum ROI</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        high_value_customers = len(processed_data[processed_data['monetary'] > processed_data['monetary'].quantile(0.8)])
        st.markdown(f"""
        <div class="insight-box">
            <h4>⭐ High-Value Customers</h4>
            <p><strong>{high_value_customers}</strong> customers in top 20% by value</p>
            <p>These customers drive significant revenue and should be prioritized</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Strategic recommendations
    st.subheader("🎯 Strategic Recommendations")
    
    recommendations = [
        {
            "title": "Customer Retention Program",
            "description": "Implement loyalty program targeting high-value customer segments",
            "impact": "Potential 15-20% increase in customer lifetime value",
            "priority": "High"
        },
        {
            "title": "Seasonal Marketing Campaigns",
            "description": "Leverage seasonal patterns for targeted promotional campaigns",
            "impact": "Expected 10-15% boost in seasonal sales",
            "priority": "Medium"
        },
        {
            "title": "Cross-Category Promotion",
            "description": "Promote complementary products across high-performing categories",
            "impact": "Increase average order value by 8-12%",
            "priority": "Medium"
        },
        {
            "title": "Digital Channel Optimization",
            "description": "Enhance online customer experience and conversion rates",
            "impact": "Potential 20-25% growth in online sales",
            "priority": "High"
        }
    ]
    
    for i, rec in enumerate(recommendations):
        with st.expander(f"{i+1}. {rec['title']} - Priority: {rec['priority']}"):
            st.write(f"**Description:** {rec['description']}")
            st.write(f"**Expected Impact:** {rec['impact']}")
            
            if rec['priority'] == 'High':
                st.success("🚀 Recommended for immediate implementation")
            else:
                st.info("📅 Consider for next quarter planning")
    
    # ROI Calculator
    st.subheader("💹 ROI Calculator")
    
    col1, col2 = st.columns(2)
    
    with col1:
        investment = st.number_input("Marketing Investment ($)", min_value=1000, max_value=100000, value=10000, step=1000)
        expected_lift = st.slider("Expected Revenue Lift (%)", 1, 50, 15)
    
    with col2:
        current_revenue = total_revenue
        projected_revenue = current_revenue * (1 + expected_lift/100)
        additional_revenue = projected_revenue - current_revenue
        roi = (additional_revenue - investment) / investment * 100
        
        st.metric("Current Revenue", f"${current_revenue:,.2f}")
        st.metric("Projected Revenue", f"${projected_revenue:,.2f}")
        st.metric("Additional Revenue", f"${additional_revenue:,.2f}")
        st.metric("ROI", f"{roi:.1f}%")
        
        if roi > 100:
            st.success("🎉 Excellent ROI potential!")
        elif roi > 50:
            st.info("👍 Good ROI potential")
        else:
            st.warning("⚠️ Consider adjusting investment or expectations")

if __name__ == "__main__":
    main()