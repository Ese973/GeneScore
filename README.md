# GeneScore: ML-Powered Variant Prioritization
Machine learning pipeline for prioritizing impactful genetic variants from GWAS data.


## Features

- **SNP Impact Scoring**: Composite scoring system combining effect size, frequency, and significance
- **Machine Learning Classification** GradientBoostingClassifier model to identify high-impact variants
- **Feature Importance Analysis**: Identify key biological factors driving predictions
- **Visualization Tools**: Interactive plots for results interpretation
- **Export Capabilities**: Save prioritized variants for further analysis

### Using GeneScore Pipeline
Using the GWAS catalog, provide a study ID to use the GeneScore pipeline. The pipeline will extract all associations and SNPs found in the study. Dataframes will be built from each JSON file, allowing for simple preprocessing. Summary statistics and viasualizations for all variants are obtained through the gwaslab package.

<div style="text-align:center"><img src="/Users/ese973/Downloads/gwas_output.png" /></div>