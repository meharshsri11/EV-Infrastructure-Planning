import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import folium

# Step 1: Simulate Data - EV Demand, Population Density, and Location Coordinates
data = {
    "Area": ['Area A', 'Area B', 'Area C', 'Area D', 'Area E', 'Area F', 'Area G', 'Area H'],
    "Latitude": [28.7041, 28.5355, 28.4089, 28.4595, 28.6435, 28.7512, 28.5273, 28.6022],
    "Longitude": [77.1025, 77.3910, 77.3178, 77.0266, 77.1828, 77.1173, 77.2085, 77.1600],
    "EV_Demand": [300, 500, 450, 200, 700, 600, 400, 350],  # Estimated EV demand in each area
    "Population_Density": [1500, 5000, 4000, 2000, 6000, 5500, 3500, 3000]  # Population density (people per sq. km)
}

# Step 2: Create DataFrame
df = pd.DataFrame(data)

# Step 3: Perform K-Means Clustering based on EV demand and population density
X = df[['EV_Demand', 'Population_Density']]
kmeans = KMeans(n_clusters=3)
df['Cluster'] = kmeans.fit_predict(X)

# Step 4: Display Clustered Data
print("Clustered Data:\n", df)

# Step 5: Plot the clusters using Matplotlib
plt.figure(figsize=(10, 6))
plt.scatter(df['EV_Demand'], df['Population_Density'], c=df['Cluster'], cmap='viridis')
plt.title("Clustering of Areas Based on EV Demand and Population Density")
plt.xlabel("EV Demand")
plt.ylabel("Population Density")
plt.colorbar(label="Cluster")
plt.show()

# Step 6: Plot clustered areas on the map using Folium
def plot_clusters_on_map(df):
    # Create a map centered around an average location
    map_center = [df['Latitude'].mean(), df['Longitude'].mean()]
    city_map = folium.Map(location=map_center, zoom_start=12)

    # Assign colors to different clusters
    colors = ['blue', 'green', 'red']

    # Add each area's location on the map with color representing its cluster
    for i, row in df.iterrows():
        folium.Marker(
            location=[row['Latitude'], row['Longitude']],
            popup=f"{row['Area']} (Cluster {row['Cluster']})",
            icon=folium.Icon(color=colors[row['Cluster']])
        ).add_to(city_map)

    return city_map

# Step 7: Generate the map and save it as an HTML file
ev_map = plot_clusters_on_map(df)
ev_map.save("ev_charging_station_clusters.html")

print("Map has been saved as 'ev_charging_station_clusters.html'. Open it in a browser to view.")
