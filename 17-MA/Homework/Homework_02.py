# %% 
# LIBRARY
import os
import pandas as pd
import numpy as np

from matplotlib import pyplot as plt 
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch

import xgboost as xgb
import graphviz

# %%
data = {
    "X": [ 30,  10, 35, 20, 25, 21],
    "Y": [-20, -10, -7,  8, 20,  7],
}

df = pd.DataFrame(data).sort_values(by="X").reset_index(drop=True)
df

# %%
# Hyperparameters
hp_loss_reduction=80   # γ     
hp_min_n = 1 
hp_regularization = 0  # λ
hp_learn_rate = 0.3
hp_trees = 2
hp_tree_depth = 2
hp_mtry = 1   # 100%


# %%
# Initialize the XGBoost 
model = xgb.XGBRegressor(
    objective ='reg:squarederror',
    colsample_bytree = hp_mtry,     # Subsample ratio of columns when constructing each tree
    learning_rate = hp_learn_rate,  # Step size shrinkage to prevent overfitting (also called `eta`)
    max_depth = hp_tree_depth,      # Maximum depth of a tree
    alpha = hp_regularization,      # L1 regularization term on weights (also called `reg_alpha`)
    gamma = hp_loss_reduction,      # Minimum loss reduction required to make a further partition on a leaf node of the tree (also called `min_split_loss`)
    min_child_weight = hp_min_n,    # Minimum sum of instance weight needed in a child
    # subsample = ,                 # Subsample ratio of the training instances
    n_estimators = hp_trees,        # Number of boosting rounds (trees)
    random_state = 21
)

# Fit the model
model.fit(df[["X"]], df["Y"])

# %%
# Check model configuration to see number of trees
print("Model parameters:", model.get_params())

# %%
# Plot the tree
xgb.plot_tree(model, num_trees=0)
plt.rcParams['figure.figsize'] = [8, 4]
plt.show()



# %%
xgb.plot_tree(model, num_trees=1)
plt.rcParams['figure.figsize'] = [8, 4]
plt.show()


# %%
# Feature importance
xgb.plot_importance(model)
plt.rcParams['figure.figsize'] = [6, 4]
plt.show()






# %%
# FIXED AUTOMATIC TREE BUILDING FUNCTION
print("FIXED AUTOMATIC TREE BUILDING")
print("=" * 60)

def calculate_similarity(residuals, lambda_=0):
    """
    residuals = Y - Y_hat (actual - prediction)
    But for gradient: g = Y_hat - Y = -residuals
    """
    sum_residuals = np.sum(residuals)
    n = len(residuals)
    # For similarity score, we use the GRADIENTS (g), not residuals
    sum_gradients = -sum_residuals  # g = Y_hat - Y = -(Y - Y_hat)
    similarity = (sum_gradients ** 2) / (n + lambda_)
    return similarity, sum_gradients, sum_residuals


def calculate_gain(left_mask, right_mask, residuals, gamma=80, lambda_=0):
    """Calculate gain for a potential split"""
    left_residuals = residuals[left_mask]
    right_residuals = residuals[right_mask]
    
    sim_left, sum_g_left, sum_r_left = calculate_similarity(left_residuals, lambda_)
    sim_right, sum_g_right, sum_r_right = calculate_similarity(right_residuals, lambda_)
    sim_root, sum_g_root, sum_r_root = calculate_similarity(residuals, lambda_)
    
    gain = sim_left + sim_right - sim_root - gamma
    
    # Calculate leaf values: leaf = -sum_g / (n + lambda)
    # But since we want to predict residuals (Y - Y_hat), and g = Y_hat - Y = -residuals
    # So to reduce residual, we want: new_Y_hat = old_Y_hat + η * (-residual_approx)
    # But in XGBoost: new_Y_hat = old_Y_hat + η * leaf_value
    # So leaf_value should approximate -residual, which means leaf_value ≈ (Y - Y_hat)
    left_leaf = sum_r_left / (len(left_residuals) + lambda_)  
    right_leaf = sum_r_right / (len(right_residuals) + lambda_) 
    
    return gain, left_leaf, right_leaf, sim_left, sim_right


def build_tree(residuals, data, feature="X", depth=0, max_depth=2, gamma=80, lambda_=0, min_child_weight=1):
    """
    Recursively build a tree using XGBoost's splitting criteria
    """
    indent = "    " * depth
    
    # Ensure residuals and data have the same indices
    if not residuals.index.equals(data.index):
        residuals = residuals.reindex(data.index)
    
    # Calculate current node similarity
    current_similarity, current_sum_gradients, sum_residuals = calculate_similarity(residuals, lambda_)
    n_samples = len(residuals)
    
    print(f"{indent}📊 Node (depth {depth}): {n_samples} samples, Σ residuals={current_sum_gradients:.1f}, Similarity={current_similarity:.1f}")
    
    # Stop conditions
    if depth >= max_depth or n_samples < 2 * min_child_weight:
        # leaf_value = sum_gradients / (n_samples + lambda_)
        # leaf_value = current_sum_gradients / (n_samples + lambda_) # using gradients
        leaf_value = sum_residuals / (n_samples + lambda_) # using residuals
        print(f"{indent}🍃 Leaf: value = {leaf_value:.2f}")
        return {"type": "leaf", "value": leaf_value}
    
    # Find best split
    best_gain = -float('inf')
    best_split = None
    best_left_data = None
    best_right_data = None
    best_left_residuals = None
    best_right_residuals = None
    
    # Generate potential splits (midpoints between sorted X values)
    sorted_x = sorted(data[feature].unique())
    if len(sorted_x) < 2:
        # leaf_value = current_sum_gradients / (n_samples + lambda_) # using gradients
        leaf_value = sum_residuals / (n_samples + lambda_) # using residuals
        print(f"{indent}🍃 Leaf (not enough unique values): value = {leaf_value:.2f}")
        return {"type": "leaf", "value": leaf_value}
    
    potential_splits = [(sorted_x[i] + sorted_x[i+1]) / 2 for i in range(len(sorted_x)-1)]
    
    for split in potential_splits:
        left_mask = data[feature] < split
        right_mask = data[feature] >= split
        
        left_data = data[left_mask]
        right_data = data[right_mask]
        left_residuals = residuals[left_mask]
        right_residuals = residuals[right_mask]
        
        if len(left_residuals) >= min_child_weight and len(right_residuals) >= min_child_weight:
            gain, left_leaf, right_leaf, sim_left, sim_right = calculate_gain(
                left_mask, right_mask, residuals, gamma, lambda_)
            
            if gain > best_gain:
                best_gain = gain
                best_split = split
                best_left_data = left_data
                best_right_data = right_data
                best_left_residuals = left_residuals
                best_right_residuals = right_residuals
                best_left_leaf = left_leaf
                best_right_leaf = right_leaf
    
    # Check if any split is worthwhile
    if best_gain <= 0 or best_split is None:
        # leaf_value = current_sum_gradients / (n_samples + lambda_) # using gradients
        leaf_value = sum_residuals / (n_samples + lambda_) # using residuals
        print(f"{indent}🍃 Leaf (no good split): value = {leaf_value:.2f}")
        return {"type": "leaf", "value": leaf_value}
    
    print(f"{indent}🎯 Split: {feature} < {best_split:.1f}, gain = {best_gain:.1f}")
    print(f"{indent}    Left leaf: {best_left_leaf:.2f}, Right leaf: {best_right_leaf:.2f}")
    
    # Recursively build left and right subtrees
    left_subtree = build_tree(best_left_residuals, best_left_data, feature, depth+1, max_depth, gamma, lambda_, min_child_weight)
    right_subtree = build_tree(best_right_residuals, best_right_data, feature, depth+1, max_depth, gamma, lambda_, min_child_weight)
    
    return {
        "type": "split",
        "feature": feature,
        "split_value": best_split,
        "gain": best_gain,
        "left": left_subtree,
        "right": right_subtree
    }

def predict_tree(tree, x):
    """Predict using the built tree"""
    if tree["type"] == "leaf":
        return tree["value"]
    else:
        if x < tree["split_value"]:
            return predict_tree(tree["left"], x)
        else:
            return predict_tree(tree["right"], x)



# %%
# Initial prediction (base_score)
base_score = 0.5
print(f"Base prediction (ŷ_initial): {base_score}")

# Calculate initial residuals (Y - ŷ)
all_residuals = df["Y"] - base_score
print("Initial residuals (Y - ŷ_initial):")
for i, (x, y, residual) in enumerate(zip(df["X"], df["Y"], all_residuals)):
    print(f"  Sample {i}: X={x}, Y={y}, ŷ={base_score}, Residual={residual:.1f}")

print(f"Sum of residuals: {np.sum(all_residuals):.1f}")
print(f"Initial MSE: {np.mean(all_residuals**2):.2f}")



# %%
# BUILD TREE 0 AUTOMATICALLY
print("BUILDING TREE 0:")
print("-" * 40)
tree0 = build_tree(all_residuals, df, "X", max_depth=hp_tree_depth, gamma=hp_loss_reduction, 
                   lambda_=hp_regularization, min_child_weight=hp_min_n)

# %%
# TREE 0 PREDICTIONS
print("\nTREE 0 PREDICTIONS:")
print("-" * 40)
tree0_predictions = np.array([predict_tree(tree0, x) for x in df["X"]])
print("Sample |   X   | Tree0 Pred |")
print("-" * 30)
for i, (x, pred) in enumerate(zip(df["X"], tree0_predictions)):
    print(f"{i:6} | {x:4} | {pred:10.2f} |")

# Apply learning rate
current_predictions = 0.5 + hp_learn_rate * tree0_predictions
current_residuals = df["Y"] - current_predictions

print(f"\nAfter learning rate (η={hp_learn_rate}):")
print("Sample |   X   | Final Pred | Residual |")
print("-" * 40)
for i, (x, pred, residual) in enumerate(zip(df["X"], current_predictions, current_residuals)):
    print(f"{i:6} | {x:4} | {pred:10.2f} | {residual:8.2f} |")

print(f"\nMSE after Tree 0: {np.mean(current_residuals**2):.2f}")

# %%
# BUILD TREE 1 AUTOMATICALLY
print("\n" + "="*60)
print("BUILDING TREE 1:")
print("-" * 40)
tree1 = build_tree(current_residuals, df, "X", max_depth=hp_tree_depth, gamma=hp_loss_reduction,
                   lambda_=hp_regularization, min_child_weight=hp_min_n)

# %%
# TREE 1 PREDICTIONS AND FINAL MODEL
print("\nTREE 1 PREDICTIONS:")
print("-" * 40)
tree1_predictions = np.array([predict_tree(tree1, x) for x in df["X"]])
print("Sample |   X   | Tree1 Pred |")
print("-" * 30)
for i, (x, pred) in enumerate(zip(df["X"], tree1_predictions)):
    print(f"{i:6} | {x:4} | {pred:10.2f} |")

# Final predictions after both trees
final_predictions = current_predictions + hp_learn_rate * tree1_predictions
final_residuals = df["Y"] - final_predictions

print(f"\nFINAL PREDICTIONS (after Tree 0 + Tree 1):")
print("Sample |   X   | Actual Y | Predicted | Residual |")
print("-" * 50)
for i, (x, y, pred, residual) in enumerate(zip(df["X"], df["Y"], final_predictions, final_residuals)):
    print(f"{i:6} | {x:4} | {y:8} | {pred:9.2f} | {residual:8.2f} |")

print(f"\nFinal MSE: {np.mean(final_residuals**2):.2f}")
print(f"Final R²: {1 - np.sum(final_residuals**2) / np.sum((df['Y'] - df['Y'].mean())**2):.3f}")

# %%
# COMPARE WITH ACTUAL XGBOOST
print("\n" + "="*60)
print("COMPARISON WITH ACTUAL XGBOOST")
print("="*60)

# Get actual XGBoost predictions
xgb_predictions = model.predict(df[["X"]])

print("Sample |   X   | Manual Pred | XGBoost Pred | Difference |")
print("-" * 60)
for i, (x, manual_pred, xgb_pred) in enumerate(zip(df["X"], final_predictions, xgb_predictions)):
    diff = manual_pred - xgb_pred
    print(f"{i:6} | {x:4} | {manual_pred:11.2f} | {xgb_pred:12.2f} | {diff:10.4f} |")

print(f"\nMean absolute difference: {np.mean(np.abs(final_predictions - xgb_predictions)):.6f}")

# %%
# TREE STRUCTURE COMPARISON
print("\nTREE STRUCTURE COMPARISON:")
print("-" * 40)

print("OUR MANUAL TREE 0:")
def print_tree_structure(tree, depth=0):
    indent = "    " * depth
    if tree["type"] == "leaf":
        print(f"{indent}leaf = {tree['value']:.2f}")
    else:
        print(f"{indent}[X < {tree['split_value']:.1f}]")
        print_tree_structure(tree["left"], depth + 1)
        print_tree_structure(tree["right"], depth + 1)

print_tree_structure(tree0)

print("\nXGBOOST'S ACTUAL TREE 0:")
trees = model.get_booster().get_dump()
print(trees[0])

# %%
print("OUR MANUAL TREE 0:")
print_tree_structure(tree1)


print("\nXGBOOST'S ACTUAL TREE 1:")
trees = model.get_booster().get_dump()
print(trees[1])


# %%












# %%
# MATPLOTLIB TREE VISUALIZATION
print("MATPLOTLIB TREE VISUALIZATION")
print("=" * 60)

def plot_tree_matplotlib(tree, tree_name="Tree", feature_name="X"):
    """Plot tree using matplotlib"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Calculate tree dimensions
    max_depth = get_tree_depth(tree)
    max_leaves = 2 ** max_depth
    
    # Tree positioning
    y_start = 0.9
    level_height = 0.15
    node_radius = 0.03
    
    def plot_node(node, x, y, depth, node_id=""):
        if node["type"] == "leaf":
            # Leaf node
            circle = patches.Circle((x, y), node_radius, facecolor='lightgreen', edgecolor='black')
            ax.add_patch(circle)
            ax.text(x, y-0.02, f"Leaf\n{node['value']:.2f}", 
                   ha='center', va='center', fontsize=8, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
        else:
            # Decision node
            circle = patches.Circle((x, y), node_radius, facecolor='lightblue', edgecolor='black')
            ax.add_patch(circle)
            ax.text(x, y-0.02, f"{feature_name} < {node['split_value']:.1f}", 
                   ha='center', va='center', fontsize=8, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
            
            # Calculate child positions
            left_x = x - (0.5 / (2 ** depth))
            right_x = x + (0.5 / (2 ** depth))
            child_y = y - level_height
            
            # Draw connections
            ax.plot([x, left_x], [y-node_radius, child_y+node_radius], 'k-', lw=1)
            ax.plot([x, right_x], [y-node_radius, child_y+node_radius], 'k-', lw=1)
            
            # Recursively plot children
            plot_node(node["left"], left_x, child_y, depth + 1, node_id + "L")
            plot_node(node["right"], right_x, child_y, depth + 1, node_id + "R")
    
    # Start plotting from root
    plot_node(tree, 0.5, y_start, 0)
    
    plt.title(f"{tree_name} (Max Depth: {max_depth})", fontsize=14, pad=20)
    plt.tight_layout()
    plt.show()

def get_tree_depth(tree):
    """Calculate the depth of the tree"""
    if tree["type"] == "leaf":
        return 0
    return 1 + max(get_tree_depth(tree["left"]), get_tree_depth(tree["right"]))

# %%
# PLOT TREE 0
print("\n📊 Plotting Tree 0...")
plot_tree_matplotlib(tree0, "Tree 0", "X")

# %%
# PLOT TREE 1
print("📊 Plotting Tree 1...")
plot_tree_matplotlib(tree1, "Tree 1", "X")

# %%
# PREDICTION VISUALIZATION
print("\n📈 PREDICTION VISUALIZATION")
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Actual vs Predicted
axes[0,0].scatter(df["X"], df["Y"], color='blue', label='Actual', s=100, alpha=0.7)
axes[0,0].scatter(df["X"], final_predictions, color='red', label='Predicted', s=100, alpha=0.7)
for i, (x, y, pred) in enumerate(zip(df["X"], df["Y"], final_predictions)):
    axes[0,0].plot([x, x], [y, pred], 'gray', linestyle='--', alpha=0.5)
axes[0,0].set_xlabel('X')
axes[0,0].set_ylabel('Y')
axes[0,0].set_title('Actual vs Predicted Values')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

# Plot 2: Residuals progression
residuals_initial = df["Y"] - base_score
residuals_after_t0 = df["Y"] - (base_score + hp_learn_rate * tree0_predictions)
residuals_final = final_residuals

x_pos = np.arange(len(df))
width = 0.25
axes[0,1].bar(x_pos - width, residuals_initial, width, label='Initial', alpha=0.7)
axes[0,1].bar(x_pos, residuals_after_t0, width, label='After Tree 0', alpha=0.7)
axes[0,1].bar(x_pos + width, residuals_final, width, label='Final', alpha=0.7)
axes[0,1].set_xlabel('Sample')
axes[0,1].set_ylabel('Residuals')
axes[0,1].set_title('Residuals Progression Through Boosting')
axes[0,1].set_xticks(x_pos)
axes[0,1].set_xticklabels([f'Sample {i}' for i in range(len(df))])
axes[0,1].legend()
axes[0,1].grid(True, alpha=0.3)

# Plot 3: Prediction components
x_pos = np.arange(len(df))
components = np.column_stack([
    np.full(len(df), base_score),
    hp_learn_rate * tree0_predictions,
    hp_learn_rate * tree1_predictions
])
axes[1,0].bar(x_pos, components[:, 0], width, label='Base', alpha=0.7)
axes[1,0].bar(x_pos, components[:, 1], width, bottom=components[:, 0], label='Tree 0', alpha=0.7)
axes[1,0].bar(x_pos, components[:, 2], width, bottom=components[:, 0] + components[:, 1], label='Tree 1', alpha=0.7)
axes[1,0].set_xlabel('Sample')
axes[1,0].set_ylabel('Prediction Components')
axes[1,0].set_title('Prediction Breakdown by Component')
axes[1,0].set_xticks(x_pos)
axes[1,0].set_xticklabels([f'Sample {i}' for i in range(len(df))])
axes[1,0].legend()
axes[1,0].grid(True, alpha=0.3)

# Plot 4: Error reduction
mse_values = [
    np.mean(residuals_initial**2),
    np.mean(residuals_after_t0**2), 
    np.mean(residuals_final**2)
]
stages = ['Initial', 'After Tree 0', 'Final']
axes[1,1].plot(stages, mse_values, 'o-', linewidth=2, markersize=8)
axes[1,1].set_xlabel('Boosting Stage')
axes[1,1].set_ylabel('MSE')
axes[1,1].set_title('Error Reduction Through Boosting')
axes[1,1].grid(True, alpha=0.3)
for i, (stage, mse) in enumerate(zip(stages, mse_values)):
    axes[1,1].annotate(f'{mse:.2f}', (stage, mse), textcoords="offset points", xytext=(0,10), ha='center')

plt.tight_layout()
plt.show()

# %%
# FINAL SUMMARY PLOT
print("\n🎯 FINAL MODEL SUMMARY")

fig, ax = plt.subplots(1, 1, figsize=(10, 6))

# Create a summary table
cell_text = []
for i, (x, y, pred, residual) in enumerate(zip(df["X"], df["Y"], final_predictions, final_residuals)):
    improvement = abs(df["Y"][i] - base_score) - abs(residual)
    cell_text.append([f'{x}', f'{y}', f'{pred:.2f}', f'{residual:.2f}', f'{improvement:.2f}'])

# Create table
table = ax.table(cellText=cell_text,
                colLabels=['X', 'Actual Y', 'Predicted', 'Residual', 'Improvement'],
                loc='center',
                cellLoc='center')

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

ax.axis('off')
ax.set_title('Manual XGBoost - Final Predictions Summary', fontsize=14, pad=20)

# Add model info as text
model_info = f"""
Model Configuration:
• Base prediction: {base_score}
• Learning rate (η): {hp_learn_rate}
• Trees: {hp_trees}
• Max depth: {hp_tree_depth}
• Gamma (γ): {hp_loss_reduction}

Performance:
• Initial MSE: {np.mean((df['Y'] - base_score)**2):.2f}
• Final MSE: {np.mean(final_residuals**2):.2f}
• R² Score: {1 - np.sum(final_residuals**2) / np.sum((df['Y'] - df['Y'].mean())**2):.3f}
"""

ax.text(0.02, 0.02, model_info, transform=ax.transAxes, fontsize=10, 
        verticalalignment='bottom', bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray"))

plt.tight_layout()
plt.show()


# %%
