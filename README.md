# Previsão de Vendas de Sorvete com Machine Learning 🍦📈

Este projeto usa regressão linear para prever a quantidade de sorvetes vendidos com base na temperatura do dia. Ideal para donos de sorveterias otimizarem a produção!

## 📊 Dataset
Simulado com dados fictícios de temperatura (°C) e vendas.

## 💻 Tecnologias
- Python
- Pandas, Matplotlib
- Scikit-learn
- FastAPI
- Uvicorn

## 🚀 Como rodar
1. Instale as dependências:
```bash
pip install -r requirements.txt
```

2. Rode a API:
```bash
uvicorn app.main:app --reload
```

3. Acesse em: [http://localhost:8000/docs](http://localhost:8000/docs) e envie uma temperatura como:
```json
{
  "temperatura": 26
}
```

## 📷 Exemplo
Temperatura: 26°C → Vendas previstas: 67 unidades
