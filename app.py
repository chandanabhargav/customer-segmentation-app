from flask import Flask, render_template, request
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    cluster_summary = None
    plot_path = None

    if request.method == 'POST':
        # Load uploaded CSV
        file = request.files['file']
        data = pd.read_csv(file)

        print(data.head())
        # Clean data
        data.fillna(data.mean(numeric_only=True), inplace=True)
        scaler = StandardScaler()
        features = ['Annual Income (k$)', 'Spending Score (1-100)']
        data_scaled = scaler.fit_transform(data[features])

        # Step 5: Apply K-Means Clustering
        kmeans = KMeans(n_clusters=5, random_state=0)
        y_kmeans = kmeans.fit_predict(data_scaled)
        data['Cluster'] = y_kmeans

        # Save the plot 
        plt.figure(figsize=(8,6))
        plt.scatter(data['Annual Income (k$)'], data['Spending Score (1-100)'], c=data['Cluster'], cmap='viridis')
        plt.xlabel('Annual Income (k$)')
        plt.ylabel('Spending Score (1-100)')
        plt.title('Customer Segments')

        plot_path = os.path.join('static', 'cluster_plot.png')
        plt.savefig(plot_path)  
        plt.close() 

        # Cluster summary (only numeric)
        cluster_summary = data.groupby('Cluster')[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].mean().round(2)

    return render_template('index.html', cluster_summary=cluster_summary, plot_path=plot_path)

# if __name__ == '__main__':
#     app.run(debug=False)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # use the PORT from env or default to 5000
    app.run(host="0.0.0.0", port=port)
