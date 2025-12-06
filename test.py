import joblib

kmeans = joblib.load("models/mb_kmeans.joblib")
print(kmeans.cluster_centers_)