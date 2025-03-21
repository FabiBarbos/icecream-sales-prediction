import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pickle

# Carregando os dados
df = pd.read_csv('dados.csv')

# Análise exploratória
plt.scatter(df['temperatura'], df['vendas'])
plt.xlabel("Temperatura (°C)")
plt.ylabel("Vendas de Sorvete")
plt.title("Temperatura vs Vendas de Sorvete")
plt.grid(True)
plt.show()

# Preparando os dados
X = df[['temperatura']]
y = df['vendas']

# Treinando o modelo
modelo = LinearRegression()
modelo.fit(X, y)

# Avaliação
print(f"Coeficiente angular (a): {modelo.coef_[0]:.2f}")
print(f"Intercepto (b): {modelo.intercept_:.2f}")
print(f"Score do modelo (R²): {modelo.score(X, y):.2f}")

# Salvando o modelo
with open('models/modelo.pkl', 'wb') as f:
    pickle.dump(modelo, f)
