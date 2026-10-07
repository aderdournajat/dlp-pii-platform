from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = SparkSession.builder.appName("pii-detection-test").getOrCreate()

df = spark.read.json("/tmp/data.jsonl")

# Ordre important : l'IPv6 d'abord, puis le plus spécifique
patterns = {
    "IPV6":         r"([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}",
    "EMAIL":        r"[\w.+-]+@[\w-]+\.[\w.-]+",
    "PHONENUMBER":  r"\+\d{2} \d{2} \d{3}-\d{4}",
    "MASKEDNUMBER": r"\b\d{16}\b",
}

# 1. Masquage
masked = F.col("source_text")
for label, regex in patterns.items():
    masked = F.regexp_replace(masked, regex, f"[{label}]")
df2 = df.withColumn("masked_text", masked)

print("=== Textes masqués ===")
df2.select("id", "masked_text").show(truncate=False)

# 2. Nombre de textes contenant chaque type
print("=== Nombre de textes par type de donnée personnelle ===")
df.select([
    F.sum(F.col("source_text").rlike(regex).cast("int")).alias(label)
    for label, regex in patterns.items()
]).show()

spark.stop()
