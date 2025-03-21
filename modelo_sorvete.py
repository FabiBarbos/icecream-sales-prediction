import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pickle
import mlflow
import mlflow.sklearn

# Carregando os dados
df = pd.read_csv('dados.csv')

# Visualização (opcional, para análise)
plt.scatter(df['temperatura'], df['vendas'])
plt.xlabel("Temperatura (°C)")
plt.ylabel("Vendas de Sorvete")
plt.title("Temperatura vs Vendas de Sorvete")
plt.grid(True)
plt.show()

# Preparando os dados
X = df[['temperatura']]
y = df['vendas']

# Iniciando experimento MLflow
mlflow.set_experiment("Previsao_Vendas_Sorvete")

with mlflow.start_run():
    modelo = LinearRegression()
    modelo.fit(X, y)

    # Avaliação
    score = modelo.score(X, y)
    coef = modelo.coef_[0]
    intercept = modelo.intercept_

    # Logs para MLflow
    mlflow.log_param("modelo", "LinearRegression")
    mlflow.log_param("coeficiente", coef)
    mlflow.log_param("intercepto", intercept)
    mlflow.log_metric("r2_score", score)

    # Salvando o modelo dentro do MLflow
    mlflow.sklearn.log_model(modelo, "modelo_sklearn")

    print(f"Modelo treinado com R² = {score:.2f}")

    # Também salvar localmente
    with open('models/modelo.pkl', 'wb') as f:
        pickle.dump(modelo, f)
