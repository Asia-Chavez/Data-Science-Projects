# Penguin-Species-Clustering-Project
**Project Type:** Personal Clustering Project<br>
**Status:** In Progress<br>
**Skills Demonstrated:** Data Cleaning, Exploratory Data Analysis, Clustering, Python<br>
**Note:** Data and business scenario from DataCamp [https://projects.datacamp.com/projects/2264]

# Project Overview
This project explores unsupervised learning techniques to identify natural groupings among penguins in Antarctica.

Researchers at the Palmer Station have collected detailed measurements — but unfortunately, species labels are missing. My task as a Data Scientist is to uncover the hidden structure in this dataset and help identify possible penguin species clusters.

By leveraging exploratory data analysis (EDA) and clustering algorithms, this project demonstrates how machine learning can provide insight into biological data, even when labels are unavailable.

# Objectives
- Clean and preprocess the dataset to handle missing values and encode categorical data
- Explore data distributions and relationships among physical traits
- Apply clustering algorithms (K-Means, Hierarchical Clustering, DBSCAN) to identify potential species groups
- Evaluate clusters using silhouette scores and visual analysis
- Interpret clusters biologically — infer which group likely corresponds to each species

# Data Preparation
- Handling missing or inconsistent values in numeric and categorical columns
- Normalizing continuous variables 
- Encoding the categorical variables
- Visualizing distributions to detect outliers or anomalies

# Methodology
1. Feature Scaling
All continuous features are standardized using StandardScaler to ensure equal weight across dimensions.

2. Clustering Techniques
The project evaluates several clustering algorithms:
- K-Means Clustering – to form k=3 clusters (expected species)
- Hierarchical Clustering – to visualize relationships between data points using dendrograms
- DBSCAN – to explore density-based clustering and identify potential outliers

3. Cluster Evaluation
- Silhouette Score: Quantitative measure of cluster separation
- Pairplots and PCA visualization: Qualitative evaluation of cluster compactness and separability

# Expected Outcomes
- Discovery of three distinct clusters, likely corresponding to the Adelie, Chinstrap, and Gentoo species.
- Insight into how physical traits differ between species (e.g., flipper length or body mass).
- Demonstration of how unsupervised learning can be used for biological classification in the absence of labels.

#  Deliverables
- Notebook / Script: Clustering_Penguin_Species.ipynb / Clustering_Penguin_Species.py