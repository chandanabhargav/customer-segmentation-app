import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from flask import Flask, render_template_string
import io
import base64

app = Flask(__name__)

@app.route('/')
def visualize():
    # Load dataset
    df = pd.read_csv('./Mall_Customers.csv')

    # Preprocessing
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    features = ['Annual Income (k$)', 'Spending Score (1-100)']
    
    # KMeans Clustering
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df[features])
    kmeans = KMeans(n_clusters=5, random_state=0)
    df['Cluster'] = kmeans.fit_predict(df_scaled)

    images = []

    def plot_to_base64():
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        encoded = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        plt.close()
        return encoded

    # Histograms
    plt.figure(figsize=(12, 6))
    for i, col in enumerate(numerical_cols):
        plt.subplot(2, 3, i + 1)
        sns.histplot(x=df[col], kde=True, bins=30)
        plt.title(col)
    plt.suptitle('Histogram of Numerical Columns')
    plt.tight_layout()
    images.append(plot_to_base64())

    # Boxplots
    plt.figure(figsize=(12, 6))
    for i, col in enumerate(numerical_cols):
        plt.subplot(2, 3, i + 1)
        sns.boxplot(y=df[col])
        plt.title(col)
    plt.suptitle('Boxplot of Numerical Columns')
    plt.tight_layout()
    images.append(plot_to_base64())

    # Countplots for categorical columns
    plt.figure(figsize=(12, 6))
    for i, col in enumerate(categorical_cols):
        plt.subplot(2, 3, i + 1)
        sns.countplot(x=df[col])
        plt.title(col)
    plt.suptitle('Countplot of Categorical Columns')
    plt.tight_layout()
    images.append(plot_to_base64())

    # Scatterplot: Age vs Annual Income
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x='Age', y='Annual Income (k$)', data=df)
    plt.title('Scatter: Age vs Annual Income')
    plt.tight_layout()
    images.append(plot_to_base64())

    # Scatterplot: Age vs Spending Score
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x='Age', y='Spending Score (1-100)', data=df)
    plt.title('Scatter: Age vs Spending Score')
    plt.tight_layout()
    images.append(plot_to_base64())

    # Heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(df[numerical_cols].corr(), annot=True, fmt='.1f', cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    images.append(plot_to_base64())

    # KMeans Clusters
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', data=df, palette='viridis', s=100)
    plt.title('Clusters: Income vs Spending Score')
    plt.grid(True)
    plt.tight_layout()
    images.append(plot_to_base64())

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Age', y='Spending Score (1-100)', hue='Cluster', data=df, palette='viridis', s=100)
    plt.title('Clusters: Age vs Spending Score')
    plt.grid(True)
    plt.tight_layout()
    images.append(plot_to_base64())

    # Render results
    html = '''
    <html>
    <head><title>Mall Customers Clustering Dashboard</title></head>
    <body style="font-family: Arial; padding: 20px;">
        <h1>Mall Customers Clustering</h1>
        {% for img in images %}
            <div style="margin-bottom: 40px;">
                <img src="data:image/png;base64,{{ img }}" style="width: 100%; max-width: 900px;" />
            </div>
            <hr>
        {% endfor %}
    </body>
    </html>
    '''
    return render_template_string(html, images=images)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # use the PORT from env or default to 5000
    app.run(host="0.0.0.0", port=port)
