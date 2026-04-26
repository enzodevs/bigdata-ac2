# %%
# AC2 - Predicao de preco no Airbnb Rio com Apache Spark
#
# Este notebook usa o dataset validado do Airbnb Rio. O alvo e uma versao
# numerica do preco (`price`) em reais. Assim o resultado do modelo e uma
# estimativa direta do valor da diaria, em vez de uma classe de preco.

from pathlib import Path
import os

from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import (
    Imputer,
    OneHotEncoder,
    StandardScaler,
    StringIndexer,
    VectorAssembler,
)
from pyspark.ml.regression import (
    DecisionTreeRegressor,
    LinearRegression,
    RandomForestRegressor,
)
from pyspark.sql import SparkSession


PARQUET_RELATIVE_PATH = Path("data/processed/airbnb_rio/listings_clean.parquet")

SEED = int(os.environ.get("AC2_SEED", "42"))
TRAIN_RATIO = float(os.environ.get("AC2_TRAIN_RATIO", "0.8"))
MODEL_SAMPLE_ROWS = int(os.environ.get("AC2_MODEL_SAMPLE_ROWS", "5000"))

LABEL_COLUMN = "price"

CATEGORICAL_FEATURES = [
    "room_type",
    "property_type_grouped",
    "neighbourhood_grouped",
    "cancellation_policy",
    "bed_type",
]

NUMERIC_FEATURES = [
    "snapshot_year",
    "snapshot_month",
    "latitude",
    "longitude",
    "accommodates",
    "bathrooms",
    "bedrooms",
    "beds",
    "guests_included",
    "minimum_nights",
    "maximum_nights",
    "availability_30",
    "availability_60",
    "availability_90",
    "availability_365",
    "number_of_reviews",
    "number_of_reviews_ltm",
    "review_scores_rating",
    "review_scores_accuracy",
    "review_scores_cleanliness",
    "review_scores_checkin",
    "review_scores_communication",
    "review_scores_location",
    "review_scores_value",
    "calculated_host_listings_count",
    "calculated_host_listings_count_entire_homes",
    "calculated_host_listings_count_private_rooms",
    "calculated_host_listings_count_shared_rooms",
    "reviews_per_month",
    "host_is_superhost",
    "host_has_profile_pic",
    "host_identity_verified",
    "is_location_exact",
    "has_availability",
    "instant_bookable",
    "require_guest_profile_picture",
    "require_guest_phone_verification",
    "has_reviews",
    "review_score_missing",
    "availability_rate",
    "beds_per_guest",
    "bedrooms_per_guest",
    "bathrooms_per_guest",
    "host_age_days",
    "log_number_of_reviews",
    "log_reviews_per_month",
    "log_minimum_nights",
    "log_host_listings_count",
]

EXCLUDED_COLUMNS = [
    ("source_file", "metadado do arquivo; no total_data.csv sempre e a mesma origem"),
    ("listing_id", "identificador do anuncio; alta cardinalidade e nao generaliza"),
    ("host_id", "identificador do anfitriao; alta cardinalidade e nao generaliza"),
    ("last_scraped", "data bruta; usamos snapshot_year e snapshot_month"),
    ("host_since", "data bruta; usamos host_age_days"),
    ("property_type", "categoria original com muitos valores; usamos property_type_grouped"),
    (
        "neighbourhood_cleansed",
        "categoria original com muitos bairros; usamos neighbourhood_grouped",
    ),
    ("price", "alvo da regressao; nao entra como feature"),
    ("weekly_price", "preco derivado/proximo do alvo; risco de vazamento"),
    ("monthly_price", "preco derivado/proximo do alvo; risco de vazamento"),
    ("security_deposit", "valor monetario definido junto da politica de preco"),
    ("cleaning_fee", "valor monetario definido junto da politica de preco"),
    ("extra_people", "valor monetario definido junto da politica de preco"),
    ("minimum_minimum_nights", "redundante para o modelo basico; usamos minimum_nights"),
    ("maximum_minimum_nights", "redundante para o modelo basico; usamos minimum_nights"),
    ("minimum_maximum_nights", "redundante para o modelo basico; usamos maximum_nights"),
    ("maximum_maximum_nights", "redundante para o modelo basico; usamos maximum_nights"),
    ("minimum_nights_avg_ntm", "redundante para o modelo basico; usamos minimum_nights"),
    ("maximum_nights_avg_ntm", "redundante para o modelo basico; usamos maximum_nights"),
    (
        "is_business_travel_ready",
        "campo legado do Airbnb; pouco informativo neste dataset",
    ),
]


def section(title: str) -> None:
    rule = "=" * 80
    print(f"\n{rule}\n{title}\n{rule}")


def resolve_parquet_path() -> str:
    candidates: list[Path] = []
    env_path = os.environ.get("AC2_PARQUET_PATH")
    if env_path:
        candidates.append(Path(env_path).expanduser())

    if "__file__" in globals():
        project_root = Path(__file__).resolve().parents[1]
        candidates.append(project_root / PARQUET_RELATIVE_PATH)

    cwd = Path.cwd().resolve()
    candidates.extend([
        cwd / PARQUET_RELATIVE_PATH,
        cwd / "ac2" / PARQUET_RELATIVE_PATH,
        cwd.parent / PARQUET_RELATIVE_PATH,
        Path("/home/jovyan/work") / PARQUET_RELATIVE_PATH,
        Path("/home/jovyan/work/ac2") / PARQUET_RELATIVE_PATH,
        Path("/home/rrghost/SoftEng/facens/bigdata/ac2") / PARQUET_RELATIVE_PATH,
    ])

    unique_candidates: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        unique_candidates.append(candidate)
        seen.add(key)

    for candidate in unique_candidates:
        if candidate.exists():
            return str(candidate)

    checked = "\n- ".join(str(candidate) for candidate in unique_candidates)
    raise FileNotFoundError(
        "Parquet do Airbnb nao encontrado. Caminhos verificados:\n- " + checked
    )


def build_spark() -> SparkSession:
    spark = (
        SparkSession.builder.appName("ac2_airbnb_price_prediction")
        .master(os.environ.get("AC2_SPARK_MASTER", "local[4]"))
        .config("spark.driver.memory", os.environ.get("AC2_DRIVER_MEMORY", "8g"))
        .config("spark.executor.memory", os.environ.get("AC2_EXECUTOR_MEMORY", "8g"))
        .config("spark.sql.shuffle.partitions", os.environ.get("AC2_SHUFFLE_PARTITIONS", "32"))
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel(os.environ.get("AC2_SPARK_LOG_LEVEL", "ERROR"))
    return spark


def vector_size(df, column_name: str) -> int:
    metadata = df.schema[column_name].metadata
    if "ml_attr" in metadata and "num_attrs" in metadata["ml_attr"]:
        return int(metadata["ml_attr"]["num_attrs"])
    return int(df.take(1)[0][column_name].size)


def evaluate_model(model_name: str, predictions) -> None:
    predictions = predictions.cache()
    predictions.createOrReplaceTempView("predictions")

    rmse = RegressionEvaluator(
        labelCol=LABEL_COLUMN,
        predictionCol="prediction",
        metricName="rmse",
    ).evaluate(predictions)
    mae = RegressionEvaluator(
        labelCol=LABEL_COLUMN,
        predictionCol="prediction",
        metricName="mae",
    ).evaluate(predictions)
    r2 = RegressionEvaluator(
        labelCol=LABEL_COLUMN,
        predictionCol="prediction",
        metricName="r2",
    ).evaluate(predictions)

    section(f"{model_name} - metricas")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R2: {r2:.6f}")

    section(f"{model_name} - exemplos de previsao")
    spark.sql(
        """
        SELECT
            ROUND(price, 2) AS price_real,
            ROUND(prediction, 2) AS price_previsto,
            ROUND(ABS(price - prediction), 2) AS erro_absoluto,
            room_type,
            neighbourhood_grouped,
            accommodates,
            bedrooms,
            bathrooms
        FROM predictions
        ORDER BY erro_absoluto
        LIMIT 12
        """
    ).show(truncate=False)

    predictions.unpersist()


def print_columns(title: str, columns: list[str]) -> None:
    section(title)
    for column_name in columns:
        print(f"- {column_name}")


def print_excluded_columns() -> None:
    section("Colunas excluidas do treino")
    for column_name, reason in EXCLUDED_COLUMNS:
        print(f"- {column_name}: {reason}")


# %%
# 1. Ambiente Spark
#
# O projeto usa Spark no container Docker da disciplina. O foco e manter um
# notebook simples, executavel e alinhado ao ecossistema Apache/Spark.

spark = build_spark()

section("Ambiente Spark")
print(f"Master: {spark.sparkContext.master}")
print(f"App name: {spark.sparkContext.appName}")
print(f"Application ID: {spark.sparkContext.applicationId}")

# %%
# 2. Leitura do Parquet
#
# O arquivo bruto `total_data.csv` foi convertido previamente para Parquet. Isso
# deixa o notebook mais simples e evita reler um CSV grande em toda execucao.

parquet_path = resolve_parquet_path()
section("Leitura do Parquet")
print(f"Parquet: {parquet_path}")

raw_df = spark.read.parquet(parquet_path)
raw_df.createOrReplaceTempView("airbnb_raw")

spark.sql("SELECT COUNT(*) AS total_rows FROM airbnb_raw").show()
raw_df.printSchema()

# %%
# 3. EDA com Spark SQL
#
# As consultas abaixo mostram volume, distribuicao do preco e campos basicos do
# dataset. Mantemos SQL aqui porque esse foi o formato pedido pelo professor.

section("Distribuicao do preco")
spark.sql(
    """
    SELECT
        COUNT(*) AS total,
        ROUND(AVG(price), 2) AS avg_price,
        percentile_approx(price, 0.25) AS p25,
        percentile_approx(price, 0.50) AS median,
        percentile_approx(price, 0.75) AS p75,
        MAX(price) AS max_price
    FROM airbnb_raw
    WHERE price IS NOT NULL
    """
).show(truncate=False)

section("Tipos de acomodacao")
spark.sql(
    """
    SELECT room_type, COUNT(*) AS total, ROUND(AVG(price), 2) AS avg_price
    FROM airbnb_raw
    WHERE price IS NOT NULL
    GROUP BY room_type
    ORDER BY total DESC
    """
).show(truncate=False)

section("Bairros com mais registros")
spark.sql(
    """
    SELECT neighbourhood_cleansed, COUNT(*) AS total, ROUND(AVG(price), 2) AS avg_price
    FROM airbnb_raw
    WHERE price IS NOT NULL
    GROUP BY neighbourhood_cleansed
    ORDER BY total DESC
    LIMIT 15
    """
).show(truncate=False)

# %%
# 4. Pre-processamento e feature engineering com SQL
#
# O alvo e a propria coluna `price`, com o valor numerico da diaria em reais.
# Para reduzir ruido, removemos precos muito baixos/altos e valores fisicamente
# estranhos em capacidade, banheiros, quartos e camas.
#
# Importante: `price` vira o label da regressao, mas nao entra no vetor de
# features. Outras colunas monetarias, como `weekly_price` e `monthly_price`,
# tambem ficam fora para evitar vazamento de informacao do preco.

model_base = spark.sql(
    """
    WITH filtered AS (
        SELECT *
        FROM airbnb_raw
        WHERE price BETWEEN 50 AND 5000
          AND accommodates BETWEEN 1 AND 20
          AND (bathrooms IS NULL OR bathrooms BETWEEN 0 AND 10)
          AND (bedrooms IS NULL OR bedrooms BETWEEN 0 AND 10)
          AND (beds IS NULL OR beds BETWEEN 0 AND 20)
    ),
    top_neighbourhoods AS (
        SELECT neighbourhood_cleansed
        FROM filtered
        GROUP BY neighbourhood_cleansed
        ORDER BY COUNT(*) DESC
        LIMIT 20
    ),
    top_property_types AS (
        SELECT property_type
        FROM filtered
        GROUP BY property_type
        ORDER BY COUNT(*) DESC
        LIMIT 12
    ),
    prepared AS (
        SELECT
            price,
            room_type,
            CASE
                WHEN p.property_type IS NOT NULL THEN f.property_type
                ELSE 'outros'
            END AS property_type_grouped,
            CASE
                WHEN n.neighbourhood_cleansed IS NOT NULL THEN f.neighbourhood_cleansed
                ELSE 'outros'
            END AS neighbourhood_grouped,
            cancellation_policy,
            bed_type,
            snapshot_year,
            snapshot_month,
            latitude,
            longitude,
            accommodates,
            bathrooms,
            bedrooms,
            beds,
            guests_included,
            minimum_nights,
            maximum_nights,
            availability_30,
            availability_60,
            availability_90,
            availability_365,
            number_of_reviews,
            number_of_reviews_ltm,
            review_scores_rating,
            review_scores_accuracy,
            review_scores_cleanliness,
            review_scores_checkin,
            review_scores_communication,
            review_scores_location,
            review_scores_value,
            calculated_host_listings_count,
            calculated_host_listings_count_entire_homes,
            calculated_host_listings_count_private_rooms,
            calculated_host_listings_count_shared_rooms,
            reviews_per_month,
            host_is_superhost,
            host_has_profile_pic,
            host_identity_verified,
            is_location_exact,
            has_availability,
            instant_bookable,
            require_guest_profile_picture,
            require_guest_phone_verification,
            CASE WHEN number_of_reviews > 0 THEN 1.0 ELSE 0.0 END AS has_reviews,
            CASE WHEN review_scores_rating IS NULL THEN 1.0 ELSE 0.0 END AS review_score_missing,
            availability_365 / 365.0 AS availability_rate,
            CASE WHEN accommodates > 0 THEN beds / accommodates ELSE NULL END AS beds_per_guest,
            CASE WHEN accommodates > 0 THEN bedrooms / accommodates ELSE NULL END AS bedrooms_per_guest,
            CASE WHEN accommodates > 0 THEN bathrooms / accommodates ELSE NULL END AS bathrooms_per_guest,
            DATEDIFF(last_scraped, host_since) AS host_age_days,
            LOG(1.0 + GREATEST(COALESCE(number_of_reviews, 0.0), 0.0)) AS log_number_of_reviews,
            LOG(1.0 + GREATEST(COALESCE(reviews_per_month, 0.0), 0.0)) AS log_reviews_per_month,
            LOG(1.0 + GREATEST(COALESCE(minimum_nights, 0.0), 0.0)) AS log_minimum_nights,
            LOG(1.0 + GREATEST(COALESCE(calculated_host_listings_count, 0.0), 0.0)) AS log_host_listings_count
        FROM filtered f
        LEFT JOIN top_neighbourhoods n
            ON f.neighbourhood_cleansed = n.neighbourhood_cleansed
        LEFT JOIN top_property_types p
            ON f.property_type = p.property_type
    )
    SELECT *
    FROM prepared
    WHERE price IS NOT NULL
    """
)

model_base.createOrReplaceTempView("airbnb_model_base")

section("Dataset apos limpeza")
spark.sql("SELECT COUNT(*) AS rows_after_cleaning FROM airbnb_model_base").show()

section("Resumo do alvo")
spark.sql(
    """
    SELECT
        COUNT(*) AS total,
        ROUND(AVG(price), 2) AS avg_price,
        percentile_approx(price, 0.50) AS median_price,
        ROUND(STDDEV(price), 2) AS stddev_price,
        MIN(price) AS min_price,
        MAX(price) AS max_price
    FROM airbnb_model_base
    """
).show(truncate=False)

# %%
# 5. Amostra para modelagem
#
# A EDA usa o Parquet completo gerado a partir de `total_data.csv`. Para o
# notebook executar em tempo razoavel em modo local, a etapa de ML usa uma
# amostra aleatoria configuravel por `MODEL_SAMPLE_ROWS`.

sample_clause = "" if MODEL_SAMPLE_ROWS <= 0 else f"ORDER BY rand({SEED}) LIMIT {MODEL_SAMPLE_ROWS}"

model_df = spark.sql(
    f"""
    SELECT
        price,
        room_type,
        property_type_grouped,
        neighbourhood_grouped,
        cancellation_policy,
        bed_type,
        {", ".join(NUMERIC_FEATURES)}
    FROM airbnb_model_base
    {sample_clause}
    """
)

model_df = model_df.cache()
model_df.createOrReplaceTempView("airbnb_model_sample")

section("Amostra usada nos modelos")
spark.sql(
    """
    SELECT
        COUNT(*) AS total,
        ROUND(AVG(price), 2) AS avg_price,
        percentile_approx(price, 0.50) AS median_price,
        MIN(price) AS min_price,
        MAX(price) AS max_price
    FROM airbnb_model_sample
    """
).show(truncate=False)

# %%
# 6. Pipeline de Machine Learning
#
# A parte de ML usa APIs do Spark ML porque os modelos precisam de vetores de
# features. A preparacao conceitual dos dados ficou em SQL.

print_columns("Features categoricas usadas", CATEGORICAL_FEATURES)
print_columns("Features numericas usadas", NUMERIC_FEATURES)
print_excluded_columns()

categorical_indexers = [
    StringIndexer(
        inputCol=column_name,
        outputCol=f"{column_name}_index",
        handleInvalid="keep",
    )
    for column_name in CATEGORICAL_FEATURES
]

encoder = OneHotEncoder(
    inputCols=[f"{column_name}_index" for column_name in CATEGORICAL_FEATURES],
    outputCols=[f"{column_name}_encoded" for column_name in CATEGORICAL_FEATURES],
    handleInvalid="keep",
)

imputer = Imputer(
    inputCols=NUMERIC_FEATURES,
    outputCols=[f"{column_name}_imputed" for column_name in NUMERIC_FEATURES],
    strategy="median",
)

assembler = VectorAssembler(
    inputCols=[
        *[f"{column_name}_encoded" for column_name in CATEGORICAL_FEATURES],
        *[f"{column_name}_imputed" for column_name in NUMERIC_FEATURES],
    ],
    outputCol="features_raw",
    handleInvalid="skip",
)

scaler = StandardScaler(
    inputCol="features_raw",
    outputCol="features",
    withStd=True,
    withMean=False,
)

preprocess_pipeline = Pipeline(stages=[
    *categorical_indexers,
    encoder,
    imputer,
    assembler,
    scaler,
])

train_df, test_df = model_df.randomSplit([TRAIN_RATIO, 1.0 - TRAIN_RATIO], seed=SEED)

section("Split treino/teste")
print(f"Treino: {train_df.count()}")
print(f"Teste: {test_df.count()}")

preprocess_model = preprocess_pipeline.fit(train_df)
train_prepared = preprocess_model.transform(train_df).cache()
test_prepared = preprocess_model.transform(test_df).cache()

feature_count = vector_size(train_prepared, "features")

section("Alvo e vetor de features")
print(f"Alvo da regressao: {LABEL_COLUMN}")
print(f"Total de features no vetor: {feature_count}")

# %%
# 7. Treinamento dos tres modelos
#
# Como agora prevemos valor numerico, usamos regressores nativos do Spark ML.
# Linear Regression usa `features` padronizado. As arvores usam `features_raw`.

models = [
    (
        "Linear Regression",
        LinearRegression(
            labelCol=LABEL_COLUMN,
            featuresCol="features",
            maxIter=50,
            regParam=0.1,
            elasticNetParam=0.0,
        ),
    ),
    (
        "Decision Tree Regressor",
        DecisionTreeRegressor(
            labelCol=LABEL_COLUMN,
            featuresCol="features_raw",
            maxDepth=8,
            minInstancesPerNode=10,
            seed=SEED,
        ),
    ),
    (
        "Random Forest Regressor",
        RandomForestRegressor(
            labelCol=LABEL_COLUMN,
            featuresCol="features_raw",
            numTrees=40,
            maxDepth=8,
            minInstancesPerNode=5,
            subsamplingRate=0.8,
            seed=SEED,
        ),
    ),
]

for model_name, estimator in models:
    section(f"Treinando {model_name}")
    model = estimator.fit(train_prepared)
    predictions = model.transform(test_prepared)
    evaluate_model(model_name, predictions)

# %%
section("Fim")
# spark.stop()
