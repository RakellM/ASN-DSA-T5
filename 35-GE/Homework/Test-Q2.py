# %%
# LIBRARY
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import folium
import mapclassify
import branca.colormap as cm
import os
import numpy as np

# %%
## PATH
main_dir = os.path.join(os.path.expanduser("~"), 
                           "OneDrive", 
                           "Project_Code")

project_dir = os.path.join(main_dir,
                           "ASN-DSA-T5", 
                           "35-GE")

# %%
# FILES
localidade = gpd.read_file(os.path.join(project_dir, "data", "Localidades", "distritos_sp.gpkg"))
enderecos = gpd.read_file(os.path.join(project_dir, "data", "Enderecos", "pontos_sp.shp"))


print(localidade.crs)
print(enderecos.crs)

# %%
# MAP VIEW
localidade.plot(edgecolor="black", figsize=(8,8)) # Distritos de São Paulo
plt.show()

enderecos.plot(markersize=1, figsize=(8,8)) # Pontos localizados em São Paulo - residências e pontos comerciais
plt.show()

# %%
# Filtering domicílios (only domiciles)

domicilios = enderecos[enderecos["COD_ESP"] == 1].copy()

print(domicilios.crs)

domicilios.plot(color="red", markersize=2)
plt.show()

# %%
# Using a projected CRS suitable for your area (e.g., UTM zone)
common_crs = "EPSG:4674"
# "EPSG:31983"  # SIRGAS 2000 / UTM zone 23S (covers much of South America)
# "EPSG:5880" for SIRGAS 2000 / UTM zone 24S

domicilios_reprojected = domicilios.to_crs(common_crs)
localidade_reprojected = localidade.to_crs(common_crs)

joined_results = gpd.sjoin(domicilios_reprojected, localidade_reprojected, predicate="within")

# %%
fig, ax = plt.subplots(figsize=(12, 10))

# Plot all polygons (neighborhoods)
localidade_reprojected.plot(
    ax=ax, 
    edgecolor='black', 
    facecolor='lightblue', 
    alpha=0.5,
    linewidth=1.5
)

# Plot all points (houses) - gray
domicilios_reprojected.plot(
    ax=ax, 
    color='gray', 
    markersize=2, 
    alpha=0.3,
    label='All houses'
)

# Plot only joined points - red (houses inside polygons)
joined_results.plot(
    ax=ax, 
    color='red', 
    markersize=5, 
    alpha=0.7,
    label='Houses within polygons'
)

ax.set_title('Spatial Join: Houses within Neighborhoods')
ax.legend()
plt.tight_layout()
plt.show()

# %%
# Perform the join
# "within" = point inside polygon
# "contains" = polygon contains point (opposite of within)
# "intersects" = point touches or crosses polygon boundary
# "crosses" = point crosses polygon boundary
joined_results = gpd.sjoin(domicilios_reprojected, localidade_reprojected, predicate="within")

# Quick visualization
fig, ax = plt.subplots(figsize=(10, 8))

# Plot polygons
localidade_reprojected.plot(ax=ax, alpha=0.3, edgecolor='blue', facecolor='none', linewidth=1.5)

# Create color list: green if inside, red if outside
inside_indices = set(joined_results.index)
colors = ['green' if i in inside_indices else 'red' for i in domicilios_reprojected.index]

# Plot all points with colors
domicilios_reprojected.plot(
    ax=ax, 
    color=colors, 
    markersize=3, 
    alpha=0.6
)

ax.set_title(f'Green = {len(joined_results)} houses inside polygons | Red = {len(domicilios_reprojected) - len(joined_results)} houses outside')
plt.show()

# Print summary
print(f"Total houses: {len(domicilios_reprojected)}")
print(f"Houses inside polygons: {len(joined_results)}")
print(f"Houses outside polygons: {len(domicilios_reprojected) - len(joined_results)}")
print(f"Percentage inside: {len(joined_results)/len(domicilios_reprojected)*100:.2f}%")

# %%
import geopandas as gpd
from shapely.geometry import Point, Polygon
import pandas as pd

# ============================================
# 1. CREATE SAMPLE POLYGONS (Neighborhoods)
# ============================================
polygons_data = {
    'name': ['Centro', 'Norte', 'Sul'],
    'geometry': [
        Polygon([(0, 0), (0, 3), (3, 3), (3, 0)]),  # Square 0-3
        Polygon([(4, 0), (4, 3), (7, 3), (7, 0)]),  # Square 4-7
        Polygon([(0, 4), (0, 7), (3, 7), (3, 4)])   # Square 0-3, y 4-7
    ]
}

localidades = gpd.GeoDataFrame(polygons_data, crs="EPSG:4326")
print("POLYGONS (Neighborhoods):")
print(localidades)
print()

# ============================================
# 2. CREATE SAMPLE POINTS (Houses)
# ============================================
points_data = {
    'house_id': [1, 2, 3, 4, 5, 6],
    'owner': ['João', 'Maria', 'Carlos', 'Ana', 'Paulo', 'Lucas'],
    'price': [200, 300, 250, 400, 350, 150],
    'geometry': [
        Point(1, 1),   # Inside Centro (0-3, 0-3)
        Point(2, 2),   # Inside Centro (0-3, 0-3)
        Point(5, 1),   # Inside Norte (4-7, 0-3)
        Point(6, 2),   # Inside Norte (4-7, 0-3)
        Point(1, 5),   # Inside Sul (0-3, 4-7)
        Point(8, 8)    # Outside ALL polygons!
    ]
}

domicilios = gpd.GeoDataFrame(points_data, crs="EPSG:4326")
print("POINTS (Houses):")
print(domicilios)
print()

# ============================================
# 3. PERFORM THE SPATIAL JOIN
# ============================================
result = gpd.sjoin(domicilios, localidades, predicate="within")

print("RESULT AFTER SPATIAL JOIN:")
print(result)
print()

# ============================================
# 4. VISUALIZE THE RESULTS
# ============================================
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# LEFT PLOT: BEFORE join (all points)
localidades.plot(ax=ax1, edgecolor='blue', facecolor='lightblue', alpha=0.5)
domicilios.plot(ax=ax1, color='red', markersize=100, alpha=0.7)

# Add labels to points
for idx, row in domicilios.iterrows():
    ax1.annotate(f'House {row["house_id"]}', 
                xy=(row.geometry.x, row.geometry.y),
                xytext=(5, 5), textcoords='offset points')
ax1.set_title('BEFORE: All 6 Houses')
ax1.set_xlim(-1, 10)
ax1.set_ylim(-1, 10)

# RIGHT PLOT: AFTER join (only houses inside polygons)
localidades.plot(ax=ax2, edgecolor='blue', facecolor='lightblue', alpha=0.5)
result.plot(ax=ax2, color='green', markersize=100, alpha=0.7)

# Add labels to joined points
for idx, row in result.iterrows():
    ax2.annotate(f'House {row["house_id"]} -> {row["name"]}', 
                xy=(row.geometry.x, row.geometry.y),
                xytext=(5, 5), textcoords='offset points')
ax2.set_title(f'AFTER: Only {len(result)} houses inside polygons')
ax2.set_xlim(-1, 10)
ax2.set_ylim(-1, 10)

plt.tight_layout()
plt.show()

# ============================================
# 5. EXPLAIN WHAT HAPPENED
# ============================================
print("\n" + "="*50)
print("EXPLANATION:")
print("="*50)
print(f"Total houses: {len(domicilios)}")
print(f"Houses that joined: {len(result)}")
print(f"Houses that were DROPPED: {len(domicilios) - len(result)}")
print()

print("Which houses joined to which polygon?")
for idx, row in result.iterrows():
    print(f"  • House {row['house_id']} (owner: {row['owner']}) -> {row['name']}")

print(f"\nHouse 6 (at 8,8) was DROPPED because it's outside ALL polygons")
print("House 3 (at 5,1) went to 'Norte' because 5 is between 4-7")

# %%
