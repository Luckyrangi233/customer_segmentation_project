import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# load data
df = pd.read_csv("data.csv")

# drop useless column
df.drop("customerID", axis=1, inplace=True)

# fix TotalCharges
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

# convert churn to numeric
df["Churn"] = df["Churn"].map({"Yes":1, "No":0})

# convert categorical → numeric
df = pd.get_dummies(df, drop_first=True)

# separate features
X = df.drop("Churn", axis=1)

# scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# find best clusters
inertia = []
for k in range(1,10):
    model = KMeans(n_clusters=k, random_state=42)
    model.fit(X_scaled)
    inertia.append(model.inertia_)

plt.plot(range(1,10), inertia, marker="o")
plt.xlabel("Clusters")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.show()

# FINAL MODEL (k=4)
kmeans = KMeans(n_clusters=4, random_state=42)
df["Cluster"] = kmeans.fit_predict(X_scaled)

print("\nCluster Counts:")
print(df["Cluster"].value_counts())

print("\nCluster Profiles:")
print(df.groupby("Cluster").mean())