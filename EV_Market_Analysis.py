import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
pd.set_option('expand_frame_repr', False)


# Exploring the dataset
ev_data = pd.read_csv('EV_Dataset1.csv')
print("Attribute of the Dataset: ", ev_data.columns)
print("Dimention of the Dataset: ", ev_data.shape)

ev_data.fillna(0, inplace=True)   # to remove the none values
ev_data[ev_data.select_dtypes([float]).columns] = ev_data.select_dtypes([float]).astype(int)   # to convert the float values to integer
print(ev_data)
ev_data = ev_data.iloc[:-1]

x = np.arange(len(ev_data['State Name']))
width = 0.8

for data in ev_data.columns[1:7]:

    plt.figure(figsize=(12, 8))
    plt.bar(x, ev_data[data], width, label=data, color='skyblue')
    plt.xlabel("States", fontsize=14)
    plt.ylabel("Sales", fontsize=14)
    plt.title(f"States vs Sales ({data})", fontsize=16)
    plt.xticks(x, ev_data['State Name'])
    plt.xticks(rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.show()

plt.figure(figsize=(12, 8))
plt.bar(x, ev_data['Grand Total'], label='Grand total', color='blue')
plt.xlabel("State", fontsize=14)
plt.ylabel("Sales", fontsize=14)
plt.title(f"States vs Total Sales of Electric Vehicles", fontsize=16)
plt.xticks(x, ev_data['State Name'])
plt.xticks(rotation=90)
plt.legend()
plt.tight_layout()
plt.show()

# Extracting Segments

Extracted_features = ev_data[['Two Wheeler', 'Three Wheeler', 'Four Wheeler', 'Goods Vehicles', 'Public Service Vehicle', 'Other Vehicles']]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(Extracted_features)

inertia = []
k_range = range(2, 10)

for k in k_range:

    kmean = KMeans(n_clusters=k, random_state=42)
    kmean.fit(scaled_features)
    inertia.append(kmean.inertia_)

plt.plot(k_range, inertia, 'bx-')
plt.xlabel('Number of clusters K')
plt.ylabel('Inertia')
plt.title('Elbow method to determine optimal k')
plt.show()


kmeans = KMeans(n_clusters=5, random_state=42)
Extracted_features['cluster'] = kmeans.fit_predict(scaled_features)
clustered_states = Extracted_features['cluster'].value_counts()
sorted_clusters = clustered_states.sort_index(ascending=True)
grouped = Extracted_features.groupby('cluster').sum()
print(sorted_clusters)
print(grouped)


x = np.arange(len(grouped))
colors = ['#03045E', '#0077B6', '#00B4D8', '#90E0EF', '#CAF0F8']
width = 0.2
for data1, j in zip(grouped.columns[:6], range(0, 6)):

    plt.figure(figsize=(12, 8))
    plt.bar(x, grouped[data1], width, label=[f'Cluster {i} ' for i in grouped.index], color=[colors[i] for i in grouped.index])
    plt.xlabel("cluster", fontsize=14)
    plt.ylabel("sales by the cluster", fontsize=14)
    plt.title(f"Clusters vs Sales ({data1})", fontsize=16)
    for i in grouped.index:
        plt.text(i, grouped.iloc[i, j] + 1,  
                str(grouped.iloc[i, j]),       
                ha='center', va='bottom') 
    plt.xticks(x, labels = [f'Cluster {i} \nwith {sorted_clusters.iloc[i:i+1,].values} states \nhaving {grouped.iloc[i:i+1, j].values} sales' for i in grouped.index])
    plt.legend()
    plt.tight_layout()
    plt.show()

pca = PCA(n_components=2)
pca_features = pca.fit_transform(scaled_features)

scatter = plt.scatter(scaled_features[:, 0], scaled_features[:, 1], c=Extracted_features['cluster'], label='o', cmap='viridis')
plt.colorbar(scatter, label='Cluster')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('KMeans Clustering with PCA')
plt.show()

Total_Sales_by_each_cluster = []

for i in grouped.index:

    total_sales = grouped.iloc[i, :].sum()
    Total_Sales_by_each_cluster.append(total_sales)

print(Total_Sales_by_each_cluster)

plt.figure(figsize=(10, 8))
plt.bar(x, Total_Sales_by_each_cluster, width=width, label=[f"cluster {i}" for i in grouped.index], color=[colors[i] for i in grouped.index])
plt.xlabel('Clusters', fontsize=14)
plt.ylabel('Total sales', fontsize=14)
plt.title('Cluster vs Total sales of All Type EVs', fontsize=16)
for i in grouped.index:
    plt.text(i, Total_Sales_by_each_cluster[i] + 1,  
             str(Total_Sales_by_each_cluster[i]),       
             ha='center', va='bottom') 
plt.xticks(grouped.index, labels=[f"cluster {i} \n{sorted_clusters.iloc[i,]} states" for i in grouped.index])
plt.legend()
plt.tight_layout()
plt.show()

# Selecting the Target Segment

max_sales = grouped.iloc[1:, :]    # Neglecting cluster 0 as it contains 26 states which are hard to target
print(max_sales, '\n')

for i, colunm in zip(range(0, 6), max_sales.columns):

    max = max_sales.iloc[:, i].max()
    max_cluster = max_sales.iloc[:, i].idxmax()
    print(f"Cluster {max_cluster} has Maximum sales of {colunm} which is {max}.")
    m = Extracted_features[Extracted_features['cluster'] == max_cluster].index
    if len(m) == 1:
        print(f"It has state: {ev_data.iloc[m, 0].values} which is best for {colunm}. \n")
    else:
        print(f"It has states: {ev_data.iloc[m, 0].values} which are best for {colunm}. \n")







