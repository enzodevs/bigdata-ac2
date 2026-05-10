# %% Cell 1
import os
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

# visualization
import seaborn as sns
import matplotlib.pyplot as plt


from pyspark.ml.feature import VectorAssembler

# %% Cell 2
LABEL_COLUMN='price'
SEED=42

spark = SparkSession.builder \
    .appName("Airbnb") \
    .getOrCreate()

raw_df = (
    spark.read
    .option("header", "true")
    .option("sep", ",")
    .option("multiLine", "true")
    .option("quote", "\"")
    .option("escape", "\"")
    .option("mode", "PERMISSIVE")
    .parquet("./total_data.parquet")
)

raw_df.createOrReplaceTempView('raw_listing')

# %% Cell 3
# as the dataset is has a lot of columns, we did an analysis and selected the most 
# good appearing to the prediction. 

selected_df = spark.sql("""
    SELECT
        latitude,
        longitude,
        host_is_superhost,
        host_listings_count,
        property_type,
        accommodates,
        bathrooms,
        bedrooms,
        beds,
        amenities,
        price,
        require_guest_profile_picture,
        require_guest_phone_verification, 
        month,
        security_deposit,
        cleaning_fee
    FROM raw_listing
""")

selected_df.createOrReplaceTempView("selected_df")

# %% Cell 5
selected_df.show()

# %% Cell 7
# change this to pure SQL
print(f'Rows: {selected_df.count()}, Columns: {len(selected_df.columns)}')

# %% Cell 9
selected_df.createOrReplaceTempView("selected_df")
null_count_expr = ",\n    ".join([
    f"SUM(CASE WHEN `{c}` IS NULL THEN 1 ELSE 0 END) AS `{c}`"
    for c in selected_df.columns
])
spark.sql(f"SELECT\n    {null_count_expr}\nFROM selected_df").show()

# %% Cell 11
rows_before_drop = selected_df.count()
selected_df = selected_df.drop('security_deposit', 'cleaning_fee')
# selected_df = selected_df.fillna({'cleaning_fee': '0'})

# %% Cell 12
selected_df = selected_df.dropna()
print(f"({selected_df.count()}, {len(selected_df.columns)}) - {rows_before_drop - selected_df.count()} rows dropped")

# %% Cell 14
selected_df.createOrReplaceTempView("selected_df")
null_count_expr = ",\n    ".join([
    f"SUM(CASE WHEN `{c}` IS NULL THEN 1 ELSE 0 END) AS `{c}`"
    for c in selected_df.columns
])
spark.sql(f"SELECT\n    {null_count_expr}\nFROM selected_df").show()

# %% Cell 15
selected_df.createOrReplaceTempView('selected_df')

# %% Cell 17
selected_df.printSchema()

# %% Cell 20
selected_df = spark.sql("""
    SELECT
        TRY_CAST(latitude AS DOUBLE) AS latitude,
        TRY_CAST(longitude AS DOUBLE) AS longitude,
        TRY_CAST(host_is_superhost AS BOOLEAN) AS host_is_superhost,
        TRY_CAST(TRY_CAST(host_listings_count AS DOUBLE) AS INT) AS host_listings_count,
        property_type,
        TRY_CAST(TRY_CAST(accommodates AS DOUBLE) AS INT) AS accommodates,
        TRY_CAST(TRY_CAST(bathrooms AS DOUBLE) AS INT) AS bathrooms,
        TRY_CAST(TRY_CAST(bedrooms AS DOUBLE) AS INT) AS bedrooms,
        TRY_CAST(TRY_CAST(beds AS DOUBLE) AS INT) AS beds,
        amenities,
        TRY_CAST(REGEXP_REPLACE(price, '[\\\\$,]', '') AS DOUBLE) AS price,
        TRY_CAST(require_guest_profile_picture AS BOOLEAN) AS require_guest_profile_picture,
        TRY_CAST(require_guest_phone_verification AS BOOLEAN) AS require_guest_phone_verification, 
        TRY_CAST(month AS INT) AS month
    FROM selected_df
""")

# TRY_CAST(REGEXP_REPLACE(cleaning_fee, '[\\\\$,]', '') AS DOUBLE) AS cleaning_fee,

selected_df.createOrReplaceTempView("selected_df")

null_count_expr = ",\n    ".join([
    f"SUM(CASE WHEN `{c}` IS NULL THEN 1 ELSE 0 END) AS `{c}`"
    for c in selected_df.columns
])
spark.sql(f"SELECT\n    {null_count_expr}\nFROM selected_df").show()

# %% Cell 23
def get_max_fence(column):
    qt = selected_df.approxQuantile(column, [0.25,0.75], 0.01)
    q1 = qt[0]
    upper = qt[1]
    iqr = upper - q1
    max_fence = upper + 1.5 * iqr
    return max_fence

# %% Cell 25
listing_count_max_fence = get_max_fence('host_listings_count')
print(listing_count_max_fence)

# %% Cell 26
rows_before = selected_df.count()
selected_df.createOrReplaceTempView('selected_df')
selected_df = spark.sql(f"""
    SELECT *
    FROM selected_df
    WHERE host_listings_count <= {listing_count_max_fence}
""")
print(f"{rows_before - selected_df.count()} rows were deleted.")

# %% Cell 27
selected_df.createOrReplaceTempView('selected_df')
selected_df = spark.sql("""
    SELECT
        latitude,
        longitude,
        host_is_superhost,
        CASE WHEN host_listings_count = 0.0 THEN 1.0 ELSE host_listings_count END AS host_listings_count,
        property_type,
        accommodates,
        bathrooms,
        bedrooms,
        beds,
        amenities,
        price,
        require_guest_profile_picture,
        require_guest_phone_verification,
        month
    FROM selected_df
""")

# %% Cell 29
def box_plot(column):
    column_pd = selected_df.select(column).toPandas()[column].dropna()
    fig, (ax1, ax2) = plt.subplots(1,2)
    fig.set_size_inches(16,6)
    _ = sns.boxplot(x=column_pd, ax = ax1)
    ax1.set_title(f'{column} boxplot')
    ax2.set_title(f'Zooming in the {column} boxplot')
    ax2.set_xlim((-0.1,1.1*get_max_fence(column)))
    _ = sns.boxplot(x=column_pd, ax = ax2)

price_pd = selected_df.select("price").toPandas()

plt.figure(figsize=(10, 6))
plt.hist(price_pd["price"].dropna(), bins=50, edgecolor="black")
plt.title("Distribuicao de valores do price")
plt.xlabel("Price")
plt.ylabel("Frequencia")
plt.show()

plt.figure(figsize=(10, 3))
plt.boxplot(price_pd["price"].dropna(), vert=False)
plt.title("Boxplot de price")
plt.xlabel("Price")
plt.show()

price_q1 = price_pd["price"].quantile(0.25)
price_q3 = price_pd["price"].quantile(0.75)
price_iqr = price_q3 - price_q1
price_zoom_min = max(0, price_q1 - 1.5 * price_iqr)
price_zoom_max = price_q3 + 1.5 * price_iqr

plt.figure(figsize=(10, 3))
plt.boxplot(price_pd["price"].dropna(), vert=False)
plt.xlim(price_zoom_min, price_zoom_max)
plt.title("Boxplot de price com zoom")
plt.xlabel("Price")
plt.show()

box_plot("price")

price_max_fence = get_max_fence('price')
print(price_max_fence)

# %% Cell 30
rows_before = selected_df.count()
selected_df.createOrReplaceTempView('selected_df')
selected_df = spark.sql(f"""
    SELECT *
    FROM selected_df
    WHERE price <= {price_max_fence}
""")
print(f"{rows_before - selected_df.count()} rows were deleted.")

# %% Cell 32
categories_to_append = ('Aparthotel', 'Earth house', 'Chalet', 'Cottage', 'Tiny house',
                        'Boutique hotel', 'Hotel', 'Casa particular (Cuba)', 'Bungalow',
                        'Nature lodge', 'Cabin', 'Castle', 'Treehouse', 'Island', 'Boat', 'Tent',
                        'Resort', 'Hut', 'Campsite', 'Barn', 'Dorm', 'Camper/RV', 'Farm stay', 'Yurt',
                        'Tipi', 'Pension (South Korea)', 'Dome house', 'Igloo', 'Casa particular',
                        'Houseboat', 'Lighthouse', 'Plane', 'Train', 'Parking Space')

category_list_sql = ", ".join([f"'" + c.replace("'", "''") + "'" for c in categories_to_append])
selected_df.createOrReplaceTempView('selected_df')
selected_df = spark.sql(f"""
    SELECT
        latitude,
        longitude,
        host_is_superhost,
        host_listings_count,
        CASE WHEN property_type IN ({category_list_sql}) THEN 'Other' ELSE property_type END AS property_type,
        accommodates,
        bathrooms,
        bedrooms,
        beds,
        amenities,
        price,
        require_guest_profile_picture,
        require_guest_phone_verification,
        month
    FROM selected_df
""")

# %% Cell 34
beds_max_fence = get_max_fence('beds')
print(beds_max_fence)

# %% Cell 35
rows_before = selected_df.count()
selected_df.createOrReplaceTempView('selected_df')
selected_df = spark.sql(f"""
    SELECT *
    FROM selected_df
    WHERE beds <= {beds_max_fence}
""")
print(f"{rows_before - selected_df.count()} rows were deleted.")

# %% Cell 37
selected_df.createOrReplaceTempView('selected_df')
selected_df = spark.sql("""
    SELECT
        latitude,
        longitude,
        host_is_superhost,
        host_listings_count,
        property_type,
        accommodates,
        bathrooms,
        bedrooms,
        beds,
        amenities,
        price,
        require_guest_profile_picture,
        require_guest_phone_verification,
        month,
        SIZE(SPLIT(amenities, ',')) + 1 AS n_amenities
    FROM selected_df
""")
selected_df.createOrReplaceTempView('selected_df')
mode_value = spark.sql("""
    SELECT n_amenities
    FROM selected_df
    WHERE amenities <> '{}'
    GROUP BY n_amenities
    ORDER BY COUNT(*) DESC
    LIMIT 1
""").first()['n_amenities']
selected_df = spark.sql(f"""
    SELECT
        latitude,
        longitude,
        host_is_superhost,
        host_listings_count,
        property_type,
        accommodates,
        bathrooms,
        bedrooms,
        beds,
        price,
        require_guest_profile_picture,
        require_guest_phone_verification,
        month,
        CASE WHEN amenities = '{{}}' THEN {mode_value} ELSE n_amenities END AS n_amenities
    FROM selected_df
""")

# %% Cell 38
amenities_max_fence = get_max_fence('n_amenities')
print(amenities_max_fence)

# %% Cell 39
rows_before = selected_df.count()
selected_df.createOrReplaceTempView('selected_df')
selected_df = spark.sql(f"""
    SELECT *
    FROM selected_df
    WHERE n_amenities <= {amenities_max_fence}
""")
print(f"{rows_before - selected_df.count()} rows were deleted.")

# %% Cell 41
selected_df.createOrReplaceTempView('selected_df')
df_label_encoder = spark.sql("""
    WITH property_type_index AS (
        SELECT
            property_type,
            ROW_NUMBER() OVER (ORDER BY cnt DESC, property_type ASC) - 1 AS property_type_encoded
        FROM (
            SELECT property_type, COUNT(*) AS cnt
            FROM selected_df
            GROUP BY property_type
        )
    )
    SELECT
        s.latitude,
        s.longitude,
        CASE
            WHEN s.host_is_superhost IS TRUE THEN 1
            WHEN s.host_is_superhost IS FALSE THEN 0
            ELSE CAST(s.host_is_superhost AS INT)
        END AS host_is_superhost,
        s.host_listings_count,
        p.property_type_encoded AS property_type,
        s.accommodates,
        s.bathrooms,
        s.bedrooms,
        s.beds,
        s.price,
        CASE
            WHEN s.require_guest_profile_picture IS TRUE THEN 1
            WHEN s.require_guest_profile_picture IS FALSE THEN 0
            ELSE CAST(s.require_guest_profile_picture AS INT)
        END AS require_guest_profile_picture,
        CASE
            WHEN s.require_guest_phone_verification IS TRUE THEN 1
            WHEN s.require_guest_phone_verification IS FALSE THEN 0
            ELSE CAST(s.require_guest_phone_verification AS INT)
        END AS require_guest_phone_verification,
        s.month,
        s.n_amenities
    FROM selected_df s
    LEFT JOIN property_type_index p
        ON s.property_type = p.property_type
""")
print('Columns encoded')

# %% Cell 43
from pyspark.ml.regression import DecisionTreeRegressor
models = [
    (
        "Linear Regression",
        LinearRegression(
            labelCol=LABEL_COLUMN,
            featuresCol="features",
            maxIter=50,
            regParam=0.1,
            elasticNetParam=0.0
        )
    ),
    (
        "Decision Tree Regressor",
        DecisionTreeRegressor(
            labelCol=LABEL_COLUMN,
            featuresCol="features",
            maxDepth=30,
            minInstancesPerNode=20,
            minInfoGain=0.0,
            maxBins=64,
            seed=SEED
        )
    )
]

# %% Cell 44
feature_cols = [
    "host_is_superhost",
    "host_listings_count",
    "latitude",
    "longitude",
    "accommodates",
    "bathrooms",
    "bedrooms",
    "beds",
    "month",
    "property_type",
    "require_guest_profile_picture",
    "require_guest_phone_verification",
    "n_amenities"
]

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
prepared_df = assembler.transform(df_label_encoder)

train_prepared, test_prepared = prepared_df.randomSplit([0.8, 0.2], seed=SEED)

print(f"Linhas no Treino: {train_prepared.count()}")
print(f"Linhas no Teste: {test_prepared.count()}")

# %% Cell 45
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

    print(f"\n{model_name}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R2: {r2:.6f}")

    predictions.unpersist()

# %% Cell 46
for model_name, estimator in models:
    model = estimator.fit(train_prepared)
    predictions = model.transform(test_prepared)
    evaluate_model(model_name, predictions)
