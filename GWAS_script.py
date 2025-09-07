# Importing required packages
import numpy as np
import requests
import pandas as pd
import gwaslab as gl
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt


# API Address for GWAS Catalog
apiUrl = "https://www.ebi.ac.uk/gwas/rest/api"

# Accessing data from specific study
study = "GCST90245848"  # Use unique accession ID for study of interest
requestUrl = (
    f"https://www.ebi.ac.uk/gwas/rest/api/studies/{study}"  # Obtain the .JSON file
)

# Make "response" object
# get() is the HTTP verb used to retrieve a resource
response = requests.get(
    requestUrl, headers={"Content-Type": "application/hal+json;charset=UTF-8"}
)

# Now extract and parse the information from the returned "response" object:
decoded = response.json()  # The returned information is parsed as a python dictionary


# We can extract specific associations (rsIDs) using the apiUrl
# Here we extract all associations from the study by obtaining the associations url from the dictionary
associationsUrl = decoded["_links"]["associations"]["href"]

# Run the same process as we did for the study JSON file
response_associations = requests.get(
    associationsUrl, headers={"Content-Type": "application/hal+json;charset=UTF-8"}
)
decoded_associations = response_associations.json()

# Generate a dataframe for each association (12,111 for this study)
associations_df = pd.DataFrame(decoded_associations["_embedded"]["associations"])


# Let's extract the riskAlleleName (e.g.,'rs17103260-C') from loci
allele_names = []

for association in decoded_associations["_embedded"]["associations"]:
    for locus in association["loci"]:
        for risk_allele in locus["strongestRiskAlleles"]:
            allele_names.append(risk_allele["riskAlleleName"])


allele_names_df = pd.DataFrame(allele_names)


# Now we need to obtain the chromosome and position of each variant/association (rsid)
snpsUrl = decoded["_links"]["snps"]["href"]

response_snps = requests.get(
    snpsUrl, headers={"Content-Type": "application/hal+json;charset=UTF-8"}
)
decoded_snps = response_snps.json()

# Now we pull each SNP's rsid, chromosome, class, last update, and location
snps = decoded_snps["_embedded"]["singleNucleotidePolymorphisms"]
rows = []

for snp in snps:
    rs_id = snp["rsId"]
    functional_class = snp["functionalClass"]
    last_update_date = snp["lastUpdateDate"]

    # Obtain locations
    for location in snp["locations"]:
        chromosome_name = location["chromosomeName"]
        chromosome_position = location["chromosomePosition"]
        region_name = location["region"]["name"]

        # Create a row for each location
        row = {
            "rsId": rs_id,
            "functionalClass": functional_class,
            "lastUpdateDate": last_update_date,
            "chromosomeName": chromosome_name,
            "chromosomePosition": chromosome_position,
            "regionName": region_name,
        }
        rows.append(row)


snp_df = pd.DataFrame(rows)
snp_df.columns.values[0] = "rsid"


# Identify SNPs that have missing rsIDs
# Can adjust manually if needed
snps_with_locations = 0
snps_no = 0
rsids_without_locations = []

for snp in snps:
    if "locations" in snp and len(snp["locations"]) > 0:
        snps_with_locations += 1
    else:
        snps_no += 1
        rsids_without_locations.append(snp["rsId"])

print(rsids_without_locations)


# Combine associations dataframe and allele names dataframe
result_df = pd.concat([allele_names_df, associations_df], axis=1).reindex(
    allele_names_df.index
)

# Rename 0 column to "rsid"
result_df.columns.values[0] = "rsid_nucleotide"

# Clean up column names by stripping whitespace
result_df.columns = result_df.columns.str.strip()
snp_df.columns = snp_df.columns.str.strip()

# Remove nucleotide from rsid
result_df["rsid"] = result_df["rsid_nucleotide"].str.replace(r"-.*$", "", regex=True)

# Add the SNP location dataframe to the first two
merged_df = snp_df.merge(result_df, on="rsid")
merged_df.to_csv("merged.csv")


# Let's use the gwaslab package to extract or stats from our dataframe (for EDA purposes)
mysumstats = gl.Sumstats(
    merged_df,
    rsid="rsid",
    chrom="chromosomeName",
    pos="chromosomePosition",
    beta="betaNum",
    p="pvalue",
    sep="\t",
)

mysumstats.data.to_csv("mysumstats.csv")

# Standardization & QC
mysumstats.basic_check()

# View Manhattan and Q-Q plots for sumstats
# gwaslab will conduct a minimal QC for sumstats when plotting
mysumstats.plot_mqq()

# Extract the lead variants
mysumstats.get_lead(anno=True)  # We extracted 1407 variants out of 500 kb widow size

# We can also view specific regions of interest
mysumstats.plot_mqq(
    mode="r", skip=2, cut=20, region=(10, 126253550, 128253550), region_grid=True
)


# Use our merged data frame to create ML pipeline for SNP prioritization
# Preprocess functional annotations
def preprocess_functional_data(df):
    df_processed = df.copy()

    # Encode functional classes
    le = LabelEncoder()
    df_processed["functionalClass_encoded"] = le.fit_transform(
        df_processed["functionalClass"]
    )

    # Extract and clean numerical features
    df_processed["riskFrequency"] = pd.to_numeric(
        df_processed["riskFrequency"], errors="coerce"
    )
    df_processed["pvalue"] = pd.to_numeric(df_processed["pvalue"], errors="coerce")
    df_processed["standardError"] = pd.to_numeric(
        df_processed["standardError"], errors="coerce"
    )

    # Create derived features
    df_processed["logP"] = -np.log10(df_processed["pvalue"])
    df_processed["effect_size"] = df_processed["orPerCopyNum"].fillna(
        df_processed["betaNum"]
    )
    df_processed["effect_magnitude"] = abs(df_processed["effect_size"])

    # Handle missing values
    df_processed = df_processed.fillna(df_processed.median(numeric_only=True))

    return df_processed


# Prepare features for ML
df_processed = preprocess_functional_data(merged_df)


# Create Impact Score and Target Variable
df_processed["impact_score"] = (
    df_processed["effect_magnitude"] * 0.4  # Effect size (40%)
    + (1 - df_processed["riskFrequency"]) * 0.3  # Rarity (30%)
    + df_processed["logP"] * 0.2  # Significance (20%)
    + (1 / (df_processed["standardError"] + 1e-10)) * 0.1  # Precision (10%)
)


# Create binary target: top 10% most impactful SNPs
top_10_percent = df_processed["impact_score"].quantile(0.9)
df_processed["is_high_impact"] = (df_processed["impact_score"] > top_10_percent).astype(
    int
)


# Select Features (using pre-GWAS features only)
feature_columns = ["functionalClass_encoded", "riskFrequency", "chromosomePosition"]

X = df_processed[feature_columns]
y = df_processed["is_high_impact"]


# Train-Test Split with Stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=11, stratify=y
)


# Train best model (Gradient Boosting accuracy: 0.989, AUC-ROC: 0.829)
best_model = GradientBoostingClassifier(n_estimators=100, random_state=11)

# Final Model Training and Evaluation
final_model = best_model.fit(X, y)  # Train on full dataset

# Make predictions on the full dataset
df_processed["predicted_impact_prob"] = final_model.predict_proba(X)[:, 1]
df_processed["predicted_high_impact"] = (
    df_processed["predicted_impact_prob"] > 0.5
).astype(int)


# Feature Importance Analysis
if hasattr(final_model, "feature_importances_"):
    feature_importance = pd.DataFrame(
        {"feature": feature_columns, "importance": final_model.feature_importances_}
    ).sort_values("importance", ascending=False)

    print("\n=== Feature Importance ===")
    print(feature_importance)

    # Plot feature importance
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance["feature"], feature_importance["importance"])
    plt.xlabel("Importance")
    plt.title("Feature Importance for SNP Impact Prediction")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()


# Results Analysis
print("\n=== Top 20 Most Impactful SNPs (Predicted) ===")
top_predicted = df_processed.nlargest(20, "predicted_impact_prob")[
    [
        "rsid",
        "functionalClass",
        "riskFrequency",
        "effect_magnitude",
        "logP",
        "impact_score",
        "predicted_impact_prob",
    ]
]
print(top_predicted.to_string(index=False))


# Performance Metrics
y_proba = final_model.predict_proba(X)[:, 1]
precision, recall, thresholds = precision_recall_curve(y, y_proba)
avg_precision = average_precision_score(y, y_proba)

print(f"\nAverage Precision: {avg_precision:.3f}")
