# ==================================================
# 1. Carregar Pacotes
# ==================================================
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Pré-processamento
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Modelos a serem utilizados
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Métricas e Visualização
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score
)

# ==================================================
# 2. Carregar e Preparar os Dados
# ==================================================
print("--- 2. Carregando e Preparando os Dados ---")
df = pd.read_csv("aula_02_exemplo_02.csv")
df["custo_medico"] = df["custo_medico"].apply(lambda x : 1 if x == "Alto_Custo" else 0)
X = df.drop("custo_medico", axis=1)
y = df["custo_medico"]
print("Dados carregados e variável alvo transformada.")

# ==================================================
# 3. Divisão dos Dados
# ==================================================
print("\n--- 3. Dividindo os Dados em Treino e Teste ---")
x_treino, x_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print("Dados divididos: 80% para treino, 20% para teste.")

# ==================================================
# 4. Pré-processamento dos Dados
# ==================================================
print("\n--- 4. Criando e Aplicando o Pipeline de Pré-processamento ---")
variaveis_categoricas = ["sex", "smoker", "region"]
variaveis_numericas = ["age", "bmi", "children"]

etapas_numericas = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
etapas_categoricas = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore"))])
preprocessador = ColumnTransformer([("num", etapas_numericas, variaveis_numericas), ("cat", etapas_categoricas, variaveis_categoricas)])

x_treino_transformado = preprocessador.fit_transform(x_treino)
x_teste_transformado = preprocessador.transform(x_teste)
print("Pré-processamento concluído.")

# =============================================================================
# 5. Treinamento e Avaliação Individual dos Modelos
# =============================================================================
print("\n--- 5. Treinando e Avaliando os Modelos Individualmente ---")

# --- Regressão Logística ---
print("\n--- Treinando Regressão Logística ---")
modelo_lr = LogisticRegression(max_iter=1000, random_state=42)
param_grid_lr = {'C': [0.1, 1, 10, 100]}
grid_search_lr = GridSearchCV(modelo_lr, param_grid_lr, cv=5, scoring='f1', n_jobs=-1)
grid_search_lr.fit(x_treino_transformado, y_treino)
melhor_modelo_lr = grid_search_lr.best_estimator_

print("\n=== Métricas de Treino (Regressão Logística) ===")
y_pred_treino_lr = melhor_modelo_lr.predict(x_treino_transformado)
print(classification_report(y_treino, y_pred_treino_lr))

# --- Random Forest ---
print("\n--- Treinando Random Forest ---")
modelo_rf = RandomForestClassifier(random_state=42)
param_grid_rf = {'n_estimators': [100, 200], 'max_depth': [5, 10, None], 'min_samples_leaf': [2, 4]}
grid_search_rf = GridSearchCV(modelo_rf, param_grid_rf, cv=5, scoring='f1', n_jobs=-1)
grid_search_rf.fit(x_treino_transformado, y_treino)
melhor_modelo_rf = grid_search_rf.best_estimator_

print("\n=== Métricas de Treino (Random Forest) ===")
y_pred_treino_rf = melhor_modelo_rf.predict(x_treino_transformado)
print(classification_report(y_treino, y_pred_treino_rf))


# --- XGBoost ---
print("\n--- Treinando XGBoost ---")
modelo_xgb = XGBClassifier(random_state=42)
param_grid_xgb = {'n_estimators': [100, 500], 'max_depth': [3, 5], 'learning_rate': [0.01, 0.1]}
grid_search_xgb = GridSearchCV(modelo_xgb, param_grid_xgb, cv=5, scoring='f1', n_jobs=-1)
grid_search_xgb.fit(x_treino_transformado, y_treino)
melhor_modelo_xgb = grid_search_xgb.best_estimator_

print("\n=== Métricas de Treino (XGBoost) ===")
y_pred_treino_xgb = melhor_modelo_xgb.predict(x_treino_transformado)
print(classification_report(y_treino, y_pred_treino_xgb))


# =============================================================================
# 6. Análise de Limiar (Threshold) no Conjunto de TREINO
# =============================================================================
print("\n--- 6. Análise de Limiar para Otimizar Precision vs. Recall ---")

# --- Modelo 1: Regressão Logística ---
print("\n[INFO] Análise de limiar para Regressão Logística (em dados de Treino)...")
y_probas_treino_lr = melhor_modelo_lr.predict_proba(x_treino_transformado)[:, 1]
limiares = np.arange(0.1, 1.0, 0.05)
scores_recall_lr = []
scores_precision_lr = []
for limiar in limiares:
    y_pred_limiar = (y_probas_treino_lr >= limiar).astype(int)
    scores_precision_lr.append(precision_score(y_treino, y_pred_limiar, zero_division=0))
    scores_recall_lr.append(recall_score(y_treino, y_pred_limiar, zero_division=0))
plt.figure(figsize=(10, 6)) 
plt.plot(limiares, scores_precision_lr, label='Precision', marker='o')
plt.plot(limiares, scores_recall_lr, label='Recall', marker='x') 
plt.title('Precision vs. Recall por Limiar - Regressão Logística (Treino)')
plt.xlabel('Limiar de Classificação')
plt.ylabel('Score')
plt.grid(True)
plt.legend()
plt.show()

# --- Modelo 2: Random Forest ---
print("\n[INFO] Análise de limiar para Random Forest (em dados de Treino)...")
y_probas_treino_rf = melhor_modelo_rf.predict_proba(x_treino_transformado)[:, 1]
scores_recall_rf = []
scores_precision_rf = []
for limiar in limiares:
    y_pred_limiar = (y_probas_treino_rf >= limiar).astype(int)
    scores_precision_rf.append(precision_score(y_treino, y_pred_limiar, zero_division=0))
    scores_recall_rf.append(recall_score(y_treino, y_pred_limiar, zero_division=0))
plt.figure(figsize=(10, 6))
plt.plot(limiares, scores_precision_rf, label='Precision', marker='o')
plt.plot(limiares, scores_recall_rf, label='Recall', marker='x')
plt.title('Precision vs. Recall por Limiar - Random Forest (Treino)')
plt.xlabel('Limiar de Classificação')
plt.ylabel('Score')
plt.grid(True)
plt.legend()
plt.show()

# --- Modelo 3: XGBoost ---
print("\n[INFO] Análise de limiar para XGBoost (em dados de Treino)...")
y_probas_treino_xgb = melhor_modelo_xgb.predict_proba(x_treino_transformado)[:, 1]
scores_recall_xgb = []
scores_precision_xgb = []
for limiar in limiares:
    y_pred_limiar = (y_probas_treino_xgb >= limiar).astype(int)
    scores_precision_xgb.append(precision_score(y_treino, y_pred_limiar, zero_division=0))
    scores_recall_xgb.append(recall_score(y_treino, y_pred_limiar, zero_division=0))
plt.figure(figsize=(10, 6))
plt.plot(limiares, scores_precision_xgb, label='Precision', marker='o')
plt.plot(limiares, scores_recall_xgb, label='Recall', marker='x')
plt.title('Precision vs. Recall por Limiar - XGBoost (Treino)')
plt.xlabel('Limiar de Classificação')
plt.ylabel('Score')
plt.grid(True)
plt.legend()
plt.show()

# ==================================================
# 7. Consolidação e Análise Comparativa dos Resultados
# ==================================================
y_pred_teste_xgb = melhor_modelo_xgb.predict(x_teste_transformado)
y_pred_teste_rf = melhor_modelo_rf.predict(x_teste_transformado)
y_pred_teste_lr = melhor_modelo_lr.predict(x_teste_transformado)

print("\n--- 7. Tabela Comparativa de Performance (F1-Score) ---")
results_list = []
results_list.append({'Modelo': 'Regressão Logística', 'Etapa': 'Treino', 'F1_Score': f1_score(y_treino, y_pred_treino_lr)})
results_list.append({'Modelo': 'Regressão Logística', 'Etapa': 'Teste', 'F1_Score': f1_score(y_teste, y_pred_teste_lr)})
results_list.append({'Modelo': 'Random Forest', 'Etapa': 'Treino', 'F1_Score': f1_score(y_treino, y_pred_treino_rf)})
results_list.append({'Modelo': 'Random Forest', 'Etapa': 'Teste', 'F1_Score': f1_score(y_teste, y_pred_teste_rf)})
results_list.append({'Modelo': 'XGBoost', 'Etapa': 'Treino', 'F1_Score': f1_score(y_treino, y_pred_treino_xgb)})
results_list.append({'Modelo': 'XGBoost', 'Etapa': 'Teste', 'F1_Score': f1_score(y_teste, y_pred_teste_xgb)})

results_df = pd.DataFrame(results_list)
results_df_sorted = results_df.sort_values(by=['Modelo', 'Etapa'])
print(results_df_sorted)

# ==================================================
# 8. Treinamento do Modelo Final e Simulação de Produção
# ==================================================
print("\n--- 8. Preparando o Modelo Final para Produção ---")
print("Com base nos relatórios de classificação e na análise de limiar, o LR foi escolhido.")
modelo_final = Pipeline([('preprocessador', preprocessador), ('modelo', melhor_modelo_lr)])
modelo_final.fit(X, y)
joblib.dump(modelo_final, 'modelo_final_classificacao.joblib')
print("\nModelo final salvo como 'modelo_final_classificacao.joblib'")

print("\n--- Simulação de Uso em Produção ---")
modelo_producao = joblib.load('modelo_final_classificacao.joblib')
dados_novos = pd.read_csv("dados_producao_classificacao.csv")
predicoes_prod = modelo_producao.predict(dados_novos)

dados_novos['pred'] = predicoes_prod
print(dados_novos)