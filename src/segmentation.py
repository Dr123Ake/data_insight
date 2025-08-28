import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class CustomerSegmentation:
    def __init__(self, data):
        self.data = data
        self.scaler = StandardScaler()
        self.kmeans = None
        self.segments = None
        
    def prepare_features(self):
        """
        Prepare features for clustering (RFM + demographics)
        """
        features = ['recency', 'frequency', 'monetary', 'age', 'income']
        
        # Handle missing values
        self.data = self.data.dropna(subset=features)
        
        # Log transform monetary and income (right-skewed)
        self.data['log_monetary'] = np.log1p(self.data['monetary'])
        self.data['log_income'] = np.log1p(self.data['income'])
        
        # Create feature matrix
        feature_cols = ['recency', 'frequency', 'log_monetary', 'age', 'log_income']
        self.features = self.data[feature_cols]
        
        # Scale features
        self.features_scaled = self.scaler.fit_transform(self.features)
        
        return self.features_scaled
    
    def find_optimal_clusters(self, max_k=10):
        """
        Find optimal number of clusters using elbow method and silhouette score
        """
        if self.features_scaled is None:
            self.prepare_features()
            
        inertias = []
        silhouette_scores = []
        k_range = range(2, max_k + 1)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(self.features_scaled)
            
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(self.features_scaled, kmeans.labels_))
        
        # Plot results
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Elbow curve
        ax1.plot(k_range, inertias, 'bo-')
        ax1.set_xlabel('Number of Clusters (k)')
        ax1.set_ylabel('Inertia')
        ax1.set_title('Elbow Method')
        ax1.grid(True)
        
        # Silhouette scores
        ax2.plot(k_range, silhouette_scores, 'ro-')
        ax2.set_xlabel('Number of Clusters (k)')
        ax2.set_ylabel('Silhouette Score')
        ax2.set_title('Silhouette Analysis')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Recommend optimal k
        optimal_k = k_range[np.argmax(silhouette_scores)]
        print(f"Recommended number of clusters: {optimal_k}")
        print(f"Silhouette score: {max(silhouette_scores):.3f}")
        
        return optimal_k
    
    def perform_clustering(self, n_clusters=4):
        """
        Perform K-means clustering
        """
        if self.features_scaled is None:
            self.prepare_features()
            
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.data['cluster'] = self.kmeans.fit_predict(self.features_scaled)
        
        # Calculate silhouette score
        silhouette_avg = silhouette_score(self.features_scaled, self.data['cluster'])
        print(f"Silhouette Score: {silhouette_avg:.3f}")
        
        return self.data
    
    def analyze_segments(self):
        """
        Analyze and interpret customer segments
        """
        if 'cluster' not in self.data.columns:
            print("Please run clustering first!")
            return None
            
        # Segment summary statistics
        segment_summary = self.data.groupby('cluster').agg({
            'recency': ['mean', 'median'],
            'frequency': ['mean', 'median'],
            'monetary': ['mean', 'median'],
            'age': ['mean', 'median'],
            'income': ['mean', 'median'],
            'customer_id': 'count'
        }).round(2)
        
        segment_summary.columns = ['_'.join(col).strip() for col in segment_summary.columns]
        segment_summary = segment_summary.rename(columns={'customer_id_count': 'customer_count'})
        
        # Add percentage
        segment_summary['percentage'] = (segment_summary['customer_count'] / len(self.data) * 100).round(1)
        
        # Segment interpretation
        interpretations = self._interpret_segments(segment_summary)
        
        self.segments = {
            'summary': segment_summary,
            'interpretations': interpretations
        }
        
        return self.segments
    
    def _interpret_segments(self, summary):
        """
        Provide business interpretation for each segment
        """
        interpretations = {}
        
        for cluster in summary.index:
            recency = summary.loc[cluster, 'recency_mean']
            frequency = summary.loc[cluster, 'frequency_mean']
            monetary = summary.loc[cluster, 'monetary_mean']
            count = summary.loc[cluster, 'customer_count']
            
            # Simple rule-based interpretation
            if recency < 50 and frequency > 3 and monetary > 200:
                segment_type = "Champions"
                description = "Best customers: recent, frequent, high-value purchases"
            elif recency < 100 and monetary > 150:
                segment_type = "Loyal Customers"
                description = "Regular customers with good purchase history"
            elif recency > 150 and frequency < 2:
                segment_type = "At Risk"
                description = "Haven't purchased recently, need re-engagement"
            elif monetary < 100:
                segment_type = "Price Sensitive"
                description = "Low-value customers, focus on value propositions"
            else:
                segment_type = "Potential Loyalists"
                description = "Recent customers with growth potential"
            
            interpretations[cluster] = {
                'type': segment_type,
                'description': description,
                'size': count,
                'avg_recency': recency,
                'avg_frequency': frequency,
                'avg_monetary': monetary
            }
        
        return interpretations
    
    def visualize_segments(self):
        """
        Create comprehensive visualizations of customer segments
        """
        if 'cluster' not in self.data.columns:
            print("Please run clustering first!")
            return
            
        # 1. RFM 3D Scatter Plot
        fig1 = px.scatter_3d(
            self.data, 
            x='recency', 
            y='frequency', 
            z='monetary',
            color='cluster',
            title='Customer Segments - RFM Analysis',
            labels={'cluster': 'Segment'},
            color_discrete_sequence=px.colors.qualitative.Set1
        )
        fig1.show()
        
        # 2. Segment Size Distribution
        segment_counts = self.data['cluster'].value_counts().sort_index()
        fig2 = px.pie(
            values=segment_counts.values,
            names=[f'Segment {i}' for i in segment_counts.index],
            title='Customer Segment Distribution'
        )
        fig2.show()
        
        # 3. Segment Characteristics Heatmap
        segment_features = self.data.groupby('cluster')[['recency', 'frequency', 'monetary', 'age', 'income']].mean()
        
        fig3 = px.imshow(
            segment_features.T,
            labels=dict(x="Segment", y="Feature", color="Value"),
            title="Segment Characteristics Heatmap",
            aspect="auto"
        )
        fig3.show()
        
        # 4. Box plots for key metrics
        fig4 = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Recency', 'Frequency', 'Monetary Value', 'Age')
        )
        
        metrics = ['recency', 'frequency', 'monetary', 'age']
        positions = [(1,1), (1,2), (2,1), (2,2)]
        
        for metric, (row, col) in zip(metrics, positions):
            for cluster in sorted(self.data['cluster'].unique()):
                cluster_data = self.data[self.data['cluster'] == cluster][metric]
                fig4.add_trace(
                    go.Box(y=cluster_data, name=f'Segment {cluster}', showlegend=(metric=='recency')),
                    row=row, col=col
                )
        
        fig4.update_layout(height=600, title_text="Segment Comparison - Key Metrics")
        fig4.show()
        
        return fig1, fig2, fig3, fig4
    
    def get_segment_recommendations(self):
        """
        Provide marketing recommendations for each segment
        """
        if not self.segments:
            print("Please run analyze_segments() first!")
            return None
            
        recommendations = {}
        
        for cluster, info in self.segments['interpretations'].items():
            segment_type = info['type']
            
            if segment_type == "Champions":
                recommendations[cluster] = {
                    'strategy': 'Reward & Retain',
                    'actions': [
                        'Offer exclusive products and early access',
                        'Implement VIP loyalty program',
                        'Request referrals and reviews',
                        'Upsell premium products'
                    ]
                }
            elif segment_type == "Loyal Customers":
                recommendations[cluster] = {
                    'strategy': 'Nurture & Grow',
                    'actions': [
                        'Cross-sell complementary products',
                        'Offer loyalty rewards',
                        'Send personalized recommendations',
                        'Invite to exclusive events'
                    ]
                }
            elif segment_type == "At Risk":
                recommendations[cluster] = {
                    'strategy': 'Re-engage',
                    'actions': [
                        'Send win-back campaigns',
                        'Offer special discounts',
                        'Conduct satisfaction surveys',
                        'Provide customer support'
                    ]
                }
            elif segment_type == "Price Sensitive":
                recommendations[cluster] = {
                    'strategy': 'Value Focus',
                    'actions': [
                        'Highlight value propositions',
                        'Offer bundle deals',
                        'Promote sales and discounts',
                        'Focus on cost-effective products'
                    ]
                }
            else:  # Potential Loyalists
                recommendations[cluster] = {
                    'strategy': 'Convert & Develop',
                    'actions': [
                        'Onboarding campaigns',
                        'Product education content',
                        'Incentivize repeat purchases',
                        'Build brand awareness'
                    ]
                }
        
        return recommendations

def main():
    # Load data
    try:
        data = pd.read_csv('data/processed_data.csv')
        print(f"Loaded {len(data)} customers for segmentation analysis")
    except FileNotFoundError:
        print("Please run data_generator.py first to create the dataset!")
        return
    
    # Initialize segmentation
    segmenter = CustomerSegmentation(data)
    
    # Find optimal clusters
    optimal_k = segmenter.find_optimal_clusters()
    
    # Perform clustering
    segmented_data = segmenter.perform_clustering(n_clusters=optimal_k)
    
    # Analyze segments
    segments = segmenter.analyze_segments()
    
    # Display results
    print("\n📊 Segment Analysis Results:")
    print(segments['summary'])
    
    print("\n🎯 Segment Interpretations:")
    for cluster, info in segments['interpretations'].items():
        print(f"\nSegment {cluster}: {info['type']}")
        print(f"  Description: {info['description']}")
        print(f"  Size: {info['size']} customers")
        print(f"  Avg Recency: {info['avg_recency']:.1f} days")
        print(f"  Avg Frequency: {info['avg_frequency']:.1f} purchases")
        print(f"  Avg Monetary: ${info['avg_monetary']:.2f}")
    
    # Get recommendations
    recommendations = segmenter.get_segment_recommendations()
    print("\n💡 Marketing Recommendations:")
    for cluster, rec in recommendations.items():
        print(f"\nSegment {cluster} - {rec['strategy']}:")
        for action in rec['actions']:
            print(f"  • {action}")
    
    # Create visualizations
    print("\n📈 Generating visualizations...")
    segmenter.visualize_segments()

if __name__ == "__main__":
    main()