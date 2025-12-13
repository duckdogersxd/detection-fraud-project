# 🚨 Detecção de Anomalias em Transações Financeiras

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.2%2B-orange)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12%2B-FF6F00)
![License](https://img.shields.io/badge/License-MIT-green)

**Detecção de transações fraudulentas usando aprendizado não supervisionado em dados extremamente desbalanceados**

</div>

---

## 📋 Sobre o Projeto

Projeto desenvolvido para a disciplina **CIN0144 - Aprendizado de Máquina e Ciência de Dados** do Centro de Informática da UFPE. O objetivo é implementar e comparar diferentes abordagens de detecção de anomalias para identificar transações fraudulentas em um cenário de extremo desbalanceamento (apenas 0.172% de fraudes).

### 🎯 Objetivos

- Desenvolver modelos robustos para detecção de anomalias em dados desbalanceados
- Implementar e comparar 3 categorias de algoritmos:
    - Modelos probabilísticos (Isolation Forest)
    - Modelos baseados em densidade (Local Outlier Factor)
    - Deep Learning (Autoencoders)
- Avaliar métricas adequadas para cenários de desbalanceamento extremo
- Analisar a aplicabilidade prática em contextos reais de fraude

---

## 📊 Dataset

**Credit Card Fraud Detection** - [Kaggle Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

| Estatística              | Valor    |
|--------------------------|----------|
| Total de transações      | 284,807  |
| Transações legítimas     | 284,315  |
| **Transações fraudulentas** | **492**   |
| **Taxa de fraude**       | **0.172%**|
| Features                 | 31 (Time + V1-V28 + Amount + Class) |

---

## 🛠️ Tecnologias Utilizadas

- **Python 3.8+**
- **Scikit-learn** (Modelos tradicionais de ML)
- **TensorFlow/Keras** (Autoencoders)
- **Pandas & NumPy** (Manipulação de dados)
- **Matplotlib & Seaborn** (Visualizações)
- **Imbalanced-learn** (Técnicas de balanceamento)

---

## 📁 Estrutura do Projeto

```
fraud-detection-project/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_modeling.ipynb
│
├── requirements.txt
└── README.md
```

---

## 🚀 Instalação e Uso

1. **Clone o repositório**
     ```bash
     git clone https://github.com/seu-usuario/fraud-detection-project.git
     cd fraud-detection-project
     ```

2. **Instale as dependências**
     ```bash
     pip install -r requirements.txt
     ```

3. **Baixe o dataset**
     - Coloque o arquivo `creditcard.csv` na pasta `data/raw/`
     - Download manual do Kaggle: [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
     ```bash
     mkdir -p data/raw
     ```

4. **Execute os notebooks em ordem**
     ```bash
     jupyter notebook notebooks/01_eda.ipynb
     ```

---

## 📊 Metodologia

### 🔍 Análise Exploratória (Notebook 01)
- Estatísticas descritivas e distribuições
- Análise de correlação entre features
- Visualização do desbalanceamento
- Detecção de outliers

### ⚙️ Pré-processamento (Notebook 02)
- Normalização das features
- Validação de dados missing
- Split estratificado
- Técnicas de balanceamento

### 🤖 Modelagem (Notebook 03)
- Implementação de 3 algoritmos:
    - Isolation Forest
    - Local Outlier Factor (LOF)
    - Autoencoder
- Otimização de hiperparâmetros

### 📈 Avaliação (Notebook 04)
- Métricas para dados desbalanceados
- Comparação entre modelos
- Análise estatística
- Visualização de resultados

---

## 👥 Equipe

| Nome      | Função                | Responsabilidades                   |
|-----------|-----------------------|-------------------------------------|
| Membro 1  | Líder Técnico         | Autoencoders, Análise Estatística   |
| Membro 2  | Especialista em Dados | EDA, Pré-processamento              |
| Membro 3  | Especialista em Modelos| IF, LOF, Otimização                |
| Membro 4  | Documentação          | Relatório, Slides, Qualidade        |

---

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo LICENSE para detalhes.

---

<div align="center">
Desenvolvido para CIN0144 - Aprendizado de Máquina e Ciência de Dados  
Centro de Informática - UFPE · 2025
</div>