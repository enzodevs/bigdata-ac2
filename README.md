# AC2 - Projeto Final de Big Data

Projeto final da disciplina de Big Data usando **Apache Spark** sobre o dataset
validado de **Airbnb Rio de Janeiro**.

## Dataset

- Base validada pelo professor: Airbnb Rio de Janeiro
- Dados brutos usados: `data/raw/airbnb_rio/total_data.csv`
- Dados processados: `data/processed/airbnb_rio/listings_clean.parquet`
- Alvo do projeto: valor numerico da diaria na coluna `price`

## Requisito de volume

O requisito do projeto pede dataset com mais de 1GB. A pasta bruta
`data/raw/airbnb_rio/` tem aproximadamente 5.2GB, e o arquivo agregado
`total_data.csv` tem aproximadamente 2.4GB.

Na conversao para Parquet, o arquivo `total_data.csv` gerou 784.121 registros
validos.

## Workflow

1. Converter o `total_data.csv` validado do Airbnb para Parquet.
2. Executar o notebook/script de analise e modelagem.
3. Converter o notebook-style `.py` para `.ipynb`.
4. Entregar o `.ipynb` executado com as saidas visiveis.

Atalhos a partir de `ac2/`:

- `make help`
- `make parquet-airbnb`
- `make run`
- `make models-sample`
- `make notebook`

## Notebook principal

- `notebooks/01_airbnb_price_prediction.py`

O notebook usa Spark SQL para EDA e pre-processamento. A API do Spark ML entra
na etapa dos modelos porque os algoritmos precisam de vetores de features.

Para manter a execucao do notebook estavel em modo local, a EDA usa o Parquet
completo e os modelos usam uma amostra aleatoria configuravel por
`MODEL_SAMPLE_ROWS`.

## Modelos

O preco e previsto como valor numerico da diaria. Como o alvo e continuo, os
tres modelos foram trocados por regressores nativos do Spark:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

## Ambiente

- Docker
- `quay.io/jupyter/pyspark-notebook`
- Spark local dentro do container

Nao ha necessidade de montar cluster distribuido para esta entrega.
