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

<div style="text-align:center"><img src<img width="2466" height="916" alt="gwas_output" src="https://github.com/user-attachments/assets/a3ce8c29-c7c0-4827-b5f5-9b1421aca6d5" /></div>

### Training the Model
Once you have your final dataframe, the pipeline will prepare your data for ML training and testing. In the provided example top 10% most impactful SNPs were obtained. These parameters can be changed based on the threshold of interest. Out of Random Forest Classifier, Logistic Regression, and Gradient Boosting Classifier, the latter had the best results for the study in the provided example (accuracy: 0.989, AUC-ROC: 0.829). 

### Results Analysis 
The pipeline will plot the most important features from your feature columns. In the provided example, the top 20 most impactful SNPs (predicted) were observed. Performance metrics from the model will also be provided. Results can be exported from the processed dataframe as a csv (or preferred format). 

<div style="text-align:center"><img src<img width="989" height="590" alt="importance_output" src="https://github.com/user-attachments/assets/60113c57-1634-4301-a4a3-adcd9691bdf0" /></div>
