import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("Prophet not available. Using alternative forecasting methods.")

class DemandForecasting:
    def __init__(self, data):
        self.data = data
        self.daily_sales = None
        self.forecasts = {}
        
    def prepare_time_series(self):
        """
        Prepare daily sales time series data
        """
        # Convert transaction_date to datetime
        self.data['transaction_date'] = pd.to_datetime(self.data['transaction_date'])
        
        # Aggregate daily sales
        self.daily_sales = self.data.groupby('transaction_date').agg({
            'total_amount': 'sum',
            'quantity': 'sum',
            'transaction_id': 'count'
        }).reset_index()
        
        self.daily_sales.columns = ['date', 'revenue', 'units_sold', 'transactions']
        
        # Fill missing dates
        date_range = pd.date_range(
            start=self.daily_sales['date'].min(),
            end=self.daily_sales['date'].max(),
            freq='D'
        )
        
        full_dates = pd.DataFrame({'date': date_range})
        self.daily_sales = full_dates.merge(self.daily_sales, on='date', how='left')
        self.daily_sales = self.daily_sales.fillna(0)
        
        # Add time features
        self.daily_sales['day_of_week'] = self.daily_sales['date'].dt.dayofweek
        self.daily_sales['month'] = self.daily_sales['date'].dt.month
        self.daily_sales['quarter'] = self.daily_sales['date'].dt.quarter
        self.daily_sales['is_weekend'] = self.daily_sales['day_of_week'].isin([5, 6]).astype(int)
        
        # Calculate moving averages
        self.daily_sales['revenue_ma7'] = self.daily_sales['revenue'].rolling(window=7).mean()
        self.daily_sales['revenue_ma30'] = self.daily_sales['revenue'].rolling(window=30).mean()
        
        return self.daily_sales
    
    def analyze_trends(self):
        """
        Analyze sales trends and patterns
        """
        if self.daily_sales is None:
            self.prepare_time_series()
            
        # Monthly aggregation
        monthly_sales = self.daily_sales.groupby(self.daily_sales['date'].dt.to_period('M')).agg({
            'revenue': 'sum',
            'units_sold': 'sum',
            'transactions': 'sum'
        }).reset_index()
        monthly_sales['date'] = monthly_sales['date'].dt.to_timestamp()
        
        # Weekly patterns
        weekly_pattern = self.daily_sales.groupby('day_of_week').agg({
            'revenue': 'mean',
            'units_sold': 'mean',
            'transactions': 'mean'
        }).reset_index()
        weekly_pattern['day_name'] = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        
        # Monthly patterns
        monthly_pattern = self.daily_sales.groupby('month').agg({
            'revenue': 'mean',
            'units_sold': 'mean',
            'transactions': 'mean'
        }).reset_index()
        
        return {
            'monthly_sales': monthly_sales,
            'weekly_pattern': weekly_pattern,
            'monthly_pattern': monthly_pattern
        }
    
    def visualize_trends(self):
        """
        Create comprehensive trend visualizations
        """
        trends = self.analyze_trends()
        
        # 1. Daily Revenue Trend
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=self.daily_sales['date'],
            y=self.daily_sales['revenue'],
            mode='lines',
            name='Daily Revenue',
            line=dict(color='lightblue', width=1)
        ))
        fig1.add_trace(go.Scatter(
            x=self.daily_sales['date'],
            y=self.daily_sales['revenue_ma7'],
            mode='lines',
            name='7-Day MA',
            line=dict(color='orange', width=2)
        ))
        fig1.add_trace(go.Scatter(
            x=self.daily_sales['date'],
            y=self.daily_sales['revenue_ma30'],
            mode='lines',
            name='30-Day MA',
            line=dict(color='red', width=2)
        ))
        fig1.update_layout(
            title='Daily Revenue Trends',
            xaxis_title='Date',
            yaxis_title='Revenue ($)',
            hovermode='x unified'
        )
        fig1.show()
        
        # 2. Seasonal Patterns
        fig2 = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Monthly Sales Trend', 'Weekly Pattern', 'Monthly Pattern', 'Revenue Distribution')
        )
        
        # Monthly trend
        fig2.add_trace(
            go.Scatter(
                x=trends['monthly_sales']['date'],
                y=trends['monthly_sales']['revenue'],
                mode='lines+markers',
                name='Monthly Revenue'
            ),
            row=1, col=1
        )
        
        # Weekly pattern
        fig2.add_trace(
            go.Bar(
                x=trends['weekly_pattern']['day_name'],
                y=trends['weekly_pattern']['revenue'],
                name='Avg Daily Revenue'
            ),
            row=1, col=2
        )
        
        # Monthly pattern
        fig2.add_trace(
            go.Bar(
                x=trends['monthly_pattern']['month'],
                y=trends['monthly_pattern']['revenue'],
                name='Avg Monthly Revenue'
            ),
            row=2, col=1
        )
        
        # Revenue distribution
        fig2.add_trace(
            go.Histogram(
                x=self.daily_sales['revenue'],
                nbinsx=30,
                name='Revenue Distribution'
            ),
            row=2, col=2
        )
        
        fig2.update_layout(height=600, title_text="Sales Pattern Analysis")
        fig2.show()
        
        return fig1, fig2
    
    def forecast_prophet(self, days_ahead=30):
        """
        Forecast using Prophet (if available)
        """
        if not PROPHET_AVAILABLE:
            print("Prophet not available. Please install: pip install prophet")
            return None
            
        # Prepare data for Prophet
        prophet_data = self.daily_sales[['date', 'revenue']].copy()
        prophet_data.columns = ['ds', 'y']
        
        # Initialize and fit Prophet model
        model = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=True,
            changepoint_prior_scale=0.05
        )
        
        model.fit(prophet_data)
        
        # Create future dataframe
        future = model.make_future_dataframe(periods=days_ahead)
        forecast = model.predict(future)
        
        # Calculate accuracy on historical data
        historical_forecast = forecast[forecast['ds'] <= prophet_data['ds'].max()]
        mae = mean_absolute_error(
            prophet_data['y'], 
            historical_forecast['yhat']
        )
        rmse = np.sqrt(mean_squared_error(
            prophet_data['y'], 
            historical_forecast['yhat']
        ))
        
        self.forecasts['prophet'] = {
            'model': model,
            'forecast': forecast,
            'mae': mae,
            'rmse': rmse,
            'accuracy': 1 - (mae / prophet_data['y'].mean())
        }
        
        return forecast
    
    def forecast_linear_trend(self, days_ahead=30):
        """
        Simple linear trend forecasting
        """
        # Prepare features
        X = np.arange(len(self.daily_sales)).reshape(-1, 1)
        y = self.daily_sales['revenue'].values
        
        # Fit linear regression
        model = LinearRegression()
        model.fit(X, y)
        
        # Predict future
        future_X = np.arange(len(self.daily_sales), len(self.daily_sales) + days_ahead).reshape(-1, 1)
        future_predictions = model.predict(future_X)
        
        # Historical predictions for accuracy
        historical_predictions = model.predict(X)
        mae = mean_absolute_error(y, historical_predictions)
        rmse = np.sqrt(mean_squared_error(y, historical_predictions))
        
        # Create forecast dataframe
        future_dates = pd.date_range(
            start=self.daily_sales['date'].max() + pd.Timedelta(days=1),
            periods=days_ahead,
            freq='D'
        )
        
        forecast_df = pd.DataFrame({
            'date': future_dates,
            'predicted_revenue': future_predictions
        })
        
        self.forecasts['linear'] = {
            'model': model,
            'forecast': forecast_df,
            'historical_predictions': historical_predictions,
            'mae': mae,
            'rmse': rmse,
            'accuracy': 1 - (mae / y.mean())
        }
        
        return forecast_df
    
    def forecast_seasonal_naive(self, days_ahead=30):
        """
        Seasonal naive forecasting (using weekly patterns)
        """
        # Calculate weekly averages
        weekly_avg = self.daily_sales.groupby('day_of_week')['revenue'].mean()
        
        # Generate future dates and day of week
        future_dates = pd.date_range(
            start=self.daily_sales['date'].max() + pd.Timedelta(days=1),
            periods=days_ahead,
            freq='D'
        )
        
        future_dow = future_dates.dayofweek
        future_predictions = [weekly_avg[dow] for dow in future_dow]
        
        # Historical predictions for accuracy
        historical_predictions = [weekly_avg[dow] for dow in self.daily_sales['day_of_week']]
        mae = mean_absolute_error(self.daily_sales['revenue'], historical_predictions)
        rmse = np.sqrt(mean_squared_error(self.daily_sales['revenue'], historical_predictions))
        
        forecast_df = pd.DataFrame({
            'date': future_dates,
            'predicted_revenue': future_predictions
        })
        
        self.forecasts['seasonal_naive'] = {
            'forecast': forecast_df,
            'historical_predictions': historical_predictions,
            'mae': mae,
            'rmse': rmse,
            'accuracy': 1 - (mae / self.daily_sales['revenue'].mean())
        }
        
        return forecast_df
    
    def compare_models(self, days_ahead=30):
        """
        Compare different forecasting models
        """
        print("Running forecasting models...")
        
        # Run all models
        self.forecast_linear_trend(days_ahead)
        self.forecast_seasonal_naive(days_ahead)
        
        if PROPHET_AVAILABLE:
            self.forecast_prophet(days_ahead)
        
        # Compare accuracy
        comparison = pd.DataFrame({
            'Model': [],
            'MAE': [],
            'RMSE': [],
            'Accuracy': []
        })
        
        for model_name, results in self.forecasts.items():
            comparison = pd.concat([
                comparison,
                pd.DataFrame({
                    'Model': [model_name.title()],
                    'MAE': [results['mae']],
                    'RMSE': [results['rmse']],
                    'Accuracy': [results['accuracy']]
                })
            ], ignore_index=True)
        
        comparison = comparison.sort_values('Accuracy', ascending=False)
        
        print("\n📊 Model Comparison:")
        print(comparison.round(4))
        
        return comparison
    
    def visualize_forecasts(self):
        """
        Visualize all forecasting results
        """
        if not self.forecasts:
            print("Please run forecasting models first!")
            return
            
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=self.daily_sales['date'],
            y=self.daily_sales['revenue'],
            mode='lines',
            name='Historical Revenue',
            line=dict(color='blue', width=2)
        ))
        
        # Add forecasts
        colors = ['red', 'green', 'orange', 'purple']
        for i, (model_name, results) in enumerate(self.forecasts.items()):
            forecast_data = results['forecast']
            
            if model_name == 'prophet' and 'ds' in forecast_data.columns:
                # Prophet format
                future_data = forecast_data[forecast_data['ds'] > self.daily_sales['date'].max()]
                fig.add_trace(go.Scatter(
                    x=future_data['ds'],
                    y=future_data['yhat'],
                    mode='lines',
                    name=f'{model_name.title()} Forecast',
                    line=dict(color=colors[i % len(colors)], width=2, dash='dash')
                ))
            else:
                # Other models format
                fig.add_trace(go.Scatter(
                    x=forecast_data['date'],
                    y=forecast_data['predicted_revenue'],
                    mode='lines',
                    name=f'{model_name.title()} Forecast',
                    line=dict(color=colors[i % len(colors)], width=2, dash='dash')
                ))
        
        fig.update_layout(
            title='Revenue Forecasting Comparison',
            xaxis_title='Date',
            yaxis_title='Revenue ($)',
            hovermode='x unified',
            height=500
        )
        
        fig.show()
        return fig
    
    def get_business_insights(self):
        """
        Generate business insights from forecasting analysis
        """
        trends = self.analyze_trends()
        
        insights = {
            'revenue_growth': {
                'monthly_trend': 'Increasing' if trends['monthly_sales']['revenue'].iloc[-1] > trends['monthly_sales']['revenue'].iloc[0] else 'Decreasing',
                'best_month': trends['monthly_pattern'].loc[trends['monthly_pattern']['revenue'].idxmax(), 'month'],
                'worst_month': trends['monthly_pattern'].loc[trends['monthly_pattern']['revenue'].idxmin(), 'month']
            },
            'weekly_patterns': {
                'best_day': trends['weekly_pattern'].loc[trends['weekly_pattern']['revenue'].idxmax(), 'day_name'],
                'worst_day': trends['weekly_pattern'].loc[trends['weekly_pattern']['revenue'].idxmin(), 'day_name'],
                'weekend_performance': 'Higher' if trends['weekly_pattern'][trends['weekly_pattern']['day_name'].isin(['Sat', 'Sun'])]['revenue'].mean() > trends['weekly_pattern'][~trends['weekly_pattern']['day_name'].isin(['Sat', 'Sun'])]['revenue'].mean() else 'Lower'
            },
            'forecasting': {
                'best_model': max(self.forecasts.keys(), key=lambda x: self.forecasts[x]['accuracy']) if self.forecasts else 'None',
                'avg_daily_revenue': self.daily_sales['revenue'].mean(),
                'revenue_volatility': self.daily_sales['revenue'].std() / self.daily_sales['revenue'].mean()
            }
        }
        
        return insights

def main():
    # Load data
    try:
        data = pd.read_csv('data/synthetic_data.csv')
        print(f"Loaded {len(data)} transactions for demand forecasting")
    except FileNotFoundError:
        print("Please run data_generator.py first to create the dataset!")
        return
    
    # Initialize forecasting
    forecaster = DemandForecasting(data)
    
    # Prepare time series
    daily_sales = forecaster.prepare_time_series()
    print(f"\n📅 Time series prepared: {len(daily_sales)} days of data")
    
    # Analyze trends
    print("\n📈 Analyzing sales trends...")
    forecaster.visualize_trends()
    
    # Compare forecasting models
    print("\n🔮 Comparing forecasting models...")
    comparison = forecaster.compare_models(days_ahead=30)
    
    # Visualize forecasts
    print("\n📊 Generating forecast visualizations...")
    forecaster.visualize_forecasts()
    
    # Business insights
    insights = forecaster.get_business_insights()
    print("\n💡 Business Insights:")
    print(f"Revenue Trend: {insights['revenue_growth']['monthly_trend']}")
    print(f"Best Performing Month: {insights['revenue_growth']['best_month']}")
    print(f"Best Performing Day: {insights['weekly_patterns']['best_day']}")
    print(f"Weekend vs Weekday: {insights['weekly_patterns']['weekend_performance']} weekend performance")
    print(f"Best Forecasting Model: {insights['forecasting']['best_model']}")
    print(f"Average Daily Revenue: ${insights['forecasting']['avg_daily_revenue']:.2f}")
    print(f"Revenue Volatility: {insights['forecasting']['revenue_volatility']:.2%}")

if __name__ == "__main__":
    main()