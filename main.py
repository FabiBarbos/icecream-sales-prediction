from fastapi import FastAPI
from pydantic import BaseModel
import pickle

# Carregando modelo
with open("models/modelo.pkl", "rb") as f:
    modelo = pickle.load(f)

app = FastAPI()

class TemperaturaInput(BaseModel):
    temperatura: float

@app.post("/prever")
def prever_vendas(data: TemperaturaInput):
    predicao = modelo.predict([[data.temperatura]])
    return {
        "temperatura": data.temperatura,
        "vendas_previstas": round(predicao[0], 2)
    }
