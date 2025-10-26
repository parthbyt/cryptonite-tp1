# Short Summary of Results of Module 2

In this module, we have learned about Clustering algorithms, including K-Means Clustering and DBSCAN, as well as Decision Trees, Random Forests, SVMs, and PCAs.

## Clustering with DBSCAN (from scratch)
 - **Algorithm:** DBCSCAN (Density-Based Spatial Clustering of Applications with Noise) implemented manually using numpy.
 - **Pokémon:** Pokémon Legendary Dataset containing multiple parameters and around 600 data points.
 - **Parameters:**
     * eplison = 3
     * min_smaples = 16
 - **Results:**
     * Detected Clusters = 1
     * Noise Points = 10
 - **Observations:** DBSCAN correctly identifies the one large Pokémon of Pokémon with minimal noise.

## Random Forest Classification (Scikit-Learn)
 - **Dataset:** Online Payment Fraud Detection
 - **Results:**
     * **Accuracy:** 99.97%
     * **Precision:** 99%
     * **Recall:** 79%
 - **Observations:** Random Forest ended up achieving a pretty high accuracy; however, a low recall suggests that there is still a large number of fraud cases that were classified as not fraud.

## SVM vs PCA - Comparison Report
 - **SVM:** Supervised algorithm that finds the optimal margin hyperplane for classification.
 - **PCA:** Unsupervised technique that reduces dimensionality by capturing parameters with maximum variance. 
