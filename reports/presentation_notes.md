# Notas de apresentacao - AC2 Big Data

## Tema

Predicao do valor da diaria de anuncios do Airbnb Rio de Janeiro usando Apache
Spark.

## Dataset

- Dataset validado pelo professor.
- Arquivo bruto usado: `data/raw/airbnb_rio/total_data.csv`.
- Volume bruto aproximado: 2.4GB no `total_data.csv`.
- Dados convertidos para Parquet em `data/processed/airbnb_rio/listings_clean.parquet`.
- Total processado no Parquet: 784.121 registros.

## Por que Parquet

- Evita reler o CSV grande em toda execucao.
- Mantem schema organizado para o Spark.
- Reduz tempo de leitura no notebook.
- Separa etapa de ingestao da etapa de analise/modelagem.

## Alvo

O alvo e a coluna `price`, convertida para `double`. O modelo faz regressao,
ou seja, tenta prever o valor numerico da diaria em reais.

Nao usamos `price` como feature. Ela entra apenas como label.

## Pre-processamento

- Leitura do `total_data.csv` com schema textual explicito.
- Conversao de colunas monetarias para `double`, removendo `$` e `,`.
- Conversao de campos booleanos `t`/`f` para `1.0`/`0.0`.
- Conversao de campos numericos para `double`.
- Criacao de `snapshot_year` e `snapshot_month`.
- Filtro de valores extremos simples:
  - `price` entre 50 e 5000
  - `accommodates` entre 1 e 20
  - `bathrooms` entre 0 e 10
  - `bedrooms` entre 0 e 10
  - `beds` entre 0 e 20

## Features categoricas usadas

- `room_type`
- `property_type_grouped`
- `neighbourhood_grouped`
- `cancellation_policy`
- `bed_type`

## Features numericas usadas

- `snapshot_year`
- `snapshot_month`
- `latitude`
- `longitude`
- `accommodates`
- `bathrooms`
- `bedrooms`
- `beds`
- `guests_included`
- `minimum_nights`
- `maximum_nights`
- `availability_30`
- `availability_60`
- `availability_90`
- `availability_365`
- `number_of_reviews`
- `number_of_reviews_ltm`
- `review_scores_rating`
- `review_scores_accuracy`
- `review_scores_cleanliness`
- `review_scores_checkin`
- `review_scores_communication`
- `review_scores_location`
- `review_scores_value`
- `calculated_host_listings_count`
- `calculated_host_listings_count_entire_homes`
- `calculated_host_listings_count_private_rooms`
- `calculated_host_listings_count_shared_rooms`
- `reviews_per_month`
- `host_is_superhost`
- `host_has_profile_pic`
- `host_identity_verified`
- `is_location_exact`
- `has_availability`
- `instant_bookable`
- `require_guest_profile_picture`
- `require_guest_phone_verification`
- `has_reviews`
- `review_score_missing`
- `availability_rate`
- `beds_per_guest`
- `bedrooms_per_guest`
- `bathrooms_per_guest`
- `host_age_days`
- `log_number_of_reviews`
- `log_reviews_per_month`
- `log_minimum_nights`
- `log_host_listings_count`

## Feature engineering

- Agrupamos `property_type` nos 12 tipos mais frequentes e usamos `outros` para
  o restante.
- Agrupamos `neighbourhood_cleansed` nos 20 bairros mais frequentes e usamos
  `outros` para o restante.
- Criamos indicadores simples de review: `has_reviews` e `review_score_missing`.
- Criamos proporcoes por hospede: `beds_per_guest`, `bedrooms_per_guest` e
  `bathrooms_per_guest`.
- Criamos `availability_rate` a partir de `availability_365`.
- Criamos `host_age_days` a partir de `host_since` e `last_scraped`.
- Criamos versoes logaritmicas para contagens muito assimetricas:
  `log_number_of_reviews`, `log_reviews_per_month`, `log_minimum_nights` e
  `log_host_listings_count`.

## Colunas excluidas do treino

- `source_file`: metadado do arquivo.
- `listing_id` e `host_id`: identificadores com alta cardinalidade.
- `last_scraped`: data bruta; usamos ano/mes do snapshot.
- `host_since`: data bruta; usamos `host_age_days`.
- `property_type`: substituida por `property_type_grouped`.
- `neighbourhood_cleansed`: substituida por `neighbourhood_grouped`.
- `price`: alvo da regressao, nao feature.
- `weekly_price` e `monthly_price`: precos derivados/proximos do alvo.
- `security_deposit`, `cleaning_fee` e `extra_people`: valores monetarios
  definidos junto da politica de preco, com risco de vazamento.
- Colunas redundantes de noites minimas/maximas:
  `minimum_minimum_nights`, `maximum_minimum_nights`,
  `minimum_maximum_nights`, `maximum_maximum_nights`,
  `minimum_nights_avg_ntm`, `maximum_nights_avg_ntm`.
- `is_business_travel_ready`: campo legado do Airbnb, pouco informativo neste
  dataset.
- Textos longos, nomes, URLs e descricoes do CSV bruto nao foram levados para o
  Parquet porque exigiriam NLP e fugiriam do escopo basico do trabalho.

## Modelos

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

## Relacao com as aulas do professor

Seguimos a mesma estrutura geral das atividades:

- Criar uma `SparkSession`.
- Ler o dataset com Spark.
- Ver volume, schema e primeiras estatisticas.
- Registrar DataFrames como tabelas temporarias.
- Usar Spark SQL para consultas e preparacao principal.
- Fazer limpeza e transformacoes antes do modelo.
- Montar vetor de features com `VectorAssembler`.
- Separar treino e teste.
- Treinar modelos do Spark ML.
- Avaliar com metricas adequadas.
- Usar Parquet para salvar/ler dados processados.

A diferenca principal e que a atividade de `aula_07` usa classificacao, enquanto
este projeto usa regressao porque queremos prever o valor exato de `price`.
Por isso usamos modelos regressivos equivalentes no Spark ML.

## Metricas

Usamos metricas de regressao:

- RMSE: erro quadratico medio em reais.
- MAE: erro absoluto medio em reais.
- R2: proporcao da variacao explicada pelo modelo.

Resultado da execucao do notebook com `MODEL_SAMPLE_ROWS=5000`:

- Linear Regression: RMSE 516,39; MAE 303,70; R2 0,423904.
- Decision Tree Regressor: RMSE 556,28; MAE 308,13; R2 0,331456.
- Random Forest Regressor: RMSE 505,13; MAE 274,77; R2 0,448741.

O melhor resultado desta execucao foi o Random Forest Regressor. Mesmo assim, o
erro ainda e alto porque preco de Airbnb tem muita variacao por fatores que nao
estao totalmente estruturados no dataset, como qualidade das fotos, descricao,
sazonalidade especifica e regras comerciais do anfitriao.

## Observacao

O professor confirmou que nao precisamos montar ambiente distribuido. O foco da
entrega e um notebook simples, executado, usando Spark e o ecossistema Apache de
forma correta.
