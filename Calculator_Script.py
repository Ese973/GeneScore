import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Load genotype data and SNP weights
def load_data(genotypes, snp_weights):
    """
    Load genotype data and SNP weights from CSV files.
    - Genotype file: Rows are samples, columns are SNPs (0, 1, 2 for genotypes).
    - Weights file: SNP IDs and their effect sizes (beta weights).
    """
    genotypes = pd.read_csv(genotypes, index_col=0)
    weights = pd.read_csv(snp_weights, index_col=0)
    return genotypes, weights


# Calculate Genetic Risk Score (GRS)
def calculate_grs(genotypes, weights):
    """
    Calculate the genetic risk score for each sample.
    - genotypes: DataFrame with genotype data.
    - weights: DataFrame with SNP IDs and effect sizes.
    """
    # Merge genotype data with weights
    data = genotypes.T.merge(weights, left_index=True, right_on="SNP", how="inner")

    # Calculate GRS for each sample
    grs_scores = {}
    for sample in genotypes.index:
        grs = np.sum(data[sample] * data["Effect Size"])
        grs_scores[sample] = grs

    return pd.Series(grs_scores)


# Main function
def main():
    # Load data
    genotype_file = "genotypes.csv"  # Replace with your genotype file
    weights_file = "snp_weights.csv"  # Replace with your SNP weights file
    genotypes, weights = load_data(genotype_file, weights_file)

    # Calculate GRS
    grs_scores = calculate_grs(genotypes, weights)

    # Save results
    grs_scores.to_csv("grs_scores.csv", header=["GRS"])
    print("Genetic Risk Scores saved to 'grs_scores.csv'.")


# Plot
def plot_grs_distribution(grs_scores, title="Genetic Risk Score Distribution"):
    """
    Plot distribution of genetic risk scores
    """
