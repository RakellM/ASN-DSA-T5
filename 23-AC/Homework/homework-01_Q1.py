# %%
# Library imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform


# %%
# Data
Students = ['Dri', 'Li', 'Bru', 'Mi', 'Re', 'Ze']
Mathematics = [9, 5, 6, 10, 4, 4]
Portuguese = [7, 4, 6, 8, 4, 9]

# DataFrame
df = pd.DataFrame({
    'Students': Students,
    'Mathematics': Mathematics,
    'Portuguese': Portuguese
})

# %%
# Plotting - dispersion plot
plt.figure(figsize=(10, 6))
plt.scatter(df['Mathematics'], df['Portuguese'], color='blue', s=100, alpha=0.6)
plt.title('Dispersion Plot of Students\' Grades')
plt.xlabel('Mathematics Grades')
plt.ylabel('Portuguese Grades') 

# add student lable on the points
for i, txt in enumerate(df['Students']):    
    plt.annotate(txt, (df['Mathematics'][i] , df['Portuguese'][i]), fontsize=10, ha='right') 


# %%
# Calculate dendogram

Z = linkage(df[['Mathematics', 'Portuguese']], method='centroid')
# Plot dendrogram
plt.figure(figsize=(10, 6))
dendrogram(Z, labels=df['Students'].values, leaf_rotation=90)
plt.title('Dendrogram of Students\' Grades')
plt.xlabel('Students')
plt.ylabel('Distance')
plt.show()


# %%
# Compute pairwise Euclidean distances
distance_matrix = squareform(pdist(X, metric='euclidean'))
print("Initial Distance Matrix:\n", distance_matrix)

# %%

# Compute initial distance matrix
X = df[['Mathematics', 'Portuguese']].values
dist_matrix = squareform(pdist(X))
print("Initial distance matrix:\n", np.round(dist_matrix, 2))

# Manually check your merge distances
print("\nManual checks:")
print("d(Li, Re):", np.linalg.norm(X[1] - X[4]))  # Li vs Re
print("d(Dri, Mi):", np.linalg.norm(X[0] - X[3]))  # Dri vs Mi
print("d(Li-Re, Bru):", 0.5*(np.linalg.norm(X[1]-X[2]) + np.linalg.norm(X[4]-X[2])) - 0.25*np.linalg.norm(X[1]-X[4]))

# %%

# Custom linkage matrix matching your merges
Z = np.array([
    [1, 4, 1.0, 2],      # Li-Re merge (indices 1 and 4, distance 1.0, new cluster size=2)
    [0, 3, 1.41, 2],     # Dri-Mi merge (indices 0 and 3, distance 1.41, new cluster size=2)
    [6, 2, 2.28, 3],     # Li-Re (now cluster 6) + Bru (index 2), distance 2.28, size=3
    [7, 5, 3.9, 4],      # Li-Re-Bru (now cluster 7) + Ze (index 5), distance 3.9, size=4
    [8, 9, 3.85, 6]      # Final merge: Li-Re-Bru-Ze (cluster 8) + Dri-Mi (cluster 9), distance 3.85, size=6
])

# Student labels (in original order: Dri=0, Li=1, Bru=2, Mi=3, Re=4, Ze=5)
labels = ['Dri', 'Li', 'Bru', 'Mi', 'Re', 'Ze']

plt.figure(figsize=(10, 5))
dendrogram(Z, labels=labels)
plt.title("Dendrogram (Manual UPGMC/Centroid Method)")
plt.xlabel("Students")
plt.ylabel("Distance")
plt.show()
# %%
