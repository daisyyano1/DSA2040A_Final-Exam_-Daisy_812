# DSA2040A_Final-Exam_-Daisy_812
 Data Warehousing and Mining
 Section 1; Data Warehousing and Management
 
Retail Data Warehouse Project — ETL, Star Schema Design & OLAP Analysis

1. Introduction

This project implements a complete Data Warehousing workflow for a retail sales dataset, including:

Star Schema Data Warehouse Design

ETL Pipeline (Extract–Transform–Load) implemented in Python

OLAP Analytical Queries executed in SQLite and Python

Visualizations generated in Jupyter Notebook

The objective is to demonstrate the entire lifecycle of a data warehousing solution, from raw operational data to analytical insights.

2. Data Warehouse Design

A Star Schema architecture was selected due to its simplicity, query performance, and suitability for OLAP workloads.

2.1. Schema Overview

The warehouse consists of:

One Fact Table

fact_sales

Four Dimension Tables

dim_customer

dim_product

dim_category

dim_time

2.2. Star Schema Diagram (Textual)
                  dim_customer
                      |
                      |
dim_product ---- fact_sales ---- dim_time
      |                |
dim_category ----------+

2.3. Table Descriptions
2.3.1. Dimension Tables
Dimension	Description	Key
dim_customer	Customer demographics, location	customer_id
dim_product	Product-level attributes	product_id
dim_category	Product category grouping	category_id
dim_time	Calendar breakdown to support time-series analysis	date_id
2.3.2. Fact Table
Fact Table	Description
fact_sales	Transaction-level sales facts including price, quantity and totals
3. ETL Process Implementation

The ETL workflow was designed and executed using Python, Pandas, and SQLite3.

3.1 Extract

Raw sales data retrieved from CSV (amazon_sales.csv).

Loaded into a Pandas DataFrame.

3.2 Transform

Key transformations included:

Handling missing values

Standardizing dates

Generating surrogate keys for dimensions

Calculating total_amount per transaction

Mapping categorical fields to foreign keys

3.3 Load

Created all dimension and fact tables in retail_dw.db:

CREATE TABLE dim_customer (...);
CREATE TABLE dim_product (...);
CREATE TABLE dim_category (...);
CREATE TABLE dim_time (...);
CREATE TABLE fact_sales (...);


Data inserted using parameterized Python SQL statements.

3.4 ETL Automation Script

Executed via:

python etl/etl_process.py


The script handles the full ETL cycle end-to-end.

4. OLAP Queries and Analysis

OLAP queries were implemented in SQLite and executed in a Jupyter Notebook using Python.

4.1. Sample OLAP Operations
4.1.1. Rollup — Sales by Country
SELECT c.country, SUM(f.total) AS TotalSales
FROM fact_sales f
JOIN dim_customer c ON f.customer_id = c.customer_id
GROUP BY c.country;

4.1.2. Cube — Category and Country
SELECT cat.category, c.country, SUM(f.total)
FROM fact_sales f
JOIN dim_product p ON f.product_id = p.product_id
JOIN dim_category cat ON p.category_id = cat.category_id
JOIN dim_customer c ON f.customer_id = c.customer_id
GROUP BY cat.category, c.country;

4.1.3. Drill Down — Year → Month
SELECT t.year, t.month, SUM(f.total)
FROM fact_sales f
JOIN dim_time t ON f.date_id = t.date_id
GROUP BY t.year, t.month;

5. Visualization of OLAP Results

Visualizations were generated using Matplotlib.

Example: Sales by Country
plt.figure(figsize=(8,4))
plt.bar(df["country"], df["TotalSales"])
plt.title("Total Sales by Country")
plt.xlabel("Country")
plt.ylabel("Sales")
plt.show()

Example: Category Contribution
plt.figure(figsize=(8,4))
plt.bar(df["category"], df["TotalSales"])
plt.title("Sales by Category")
plt.xticks(rotation=45)
plt.show()


Each visualization provides interpretable, analytical insights into sales performance.

6. Tools & Technologies
Technology	Purpose
Python	ETL, scripting
Pandas	Data cleaning & transformation
SQLite	Data Warehouse storage
Jupyter Notebook	OLAP and visualization
Matplotlib	Charting & visual analytics
7. Results Summary

A fully functional star-schema warehouse was created.

ETL pipeline successfully transformed operational sales data.

OLAP queries revealed:

High-performing customer regions

Top-selling product categories

Temporal sales trends (monthly/annual)

Visualizations support decision-making and reporting.

Section 2: Data mining
README 1 – Data Preprocessing

📌 Overview

This module handles data loading, cleaning, transformation, and preparation of the Iris dataset for machine learning tasks including clustering, classification, and association rule mining. The preprocessing steps ensure that the dataset is standardized, consistent, and model-ready.

📂 Dataset Description

The Iris dataset contains 150 flower samples across 3 species:

Iris setosa

Iris versicolor

Iris virginica

Features:

Sepal Length

Sepal Width

Petal Length

Petal Width

Species (class label)

🔧 Preprocessing Steps
1. Load Dataset

The dataset is loaded via:

from sklearn.datasets import load_iris


Converted to a pandas DataFrame for analysis.

2. Missing Value Checks

Even though no missing values exist in the Iris dataset, validation is done:

df.isnull().sum()


This guarantees integrity before training.

3. Feature Scaling (Min-Max Normalization)

Scaled to range 0–1 using:

MinMaxScaler()


This ensures equal feature influence for models like K-Means, KNN, and PCA.

4. Label Encoding

Species labels are kept as integers (0, 1, 2).
Optionally converted to one-hot encoding for algorithms requiring binary vectors.

5. Train–Test Split

A reusable function splits the data into 80% training and 20% testing for classification tasks.

📤 Outputs

X_scaled: normalized features

y: encoded species labels

X_train, X_test: prepared splits

df: cleaned DataFrame

🎯 Purpose

The preprocessing pipeline guarantees:

Clean and usable data

Standardized feature scales

Compatibility with clustering, classification, and rule mining models

A reproducible workflow

⭐ End of Preprocessing README
✅ README 2 – K-Means Clustering
📌 Overview

This module applies K-Means clustering on the preprocessed Iris dataset to discover natural groupings of flowers without using labels. The clustering is evaluated and visualized to understand cluster quality and separability.

🔧 Methods Used
1. K-Means Clustering (k = 3)

Since the Iris dataset has three species, the algorithm is run with:

KMeans(n_clusters=3)


Cluster labels are compared to ground truth using:

Adjusted Rand Index (ARI)

2. Experimentation (k = 2 and k = 4)

Additional runs with k = 2 and k = 4 evaluate over- and under-clustering.

3. Elbow Method

An inertia plot from k = 1 to k = 10 identifies the optimal cluster number.

4. Visualizations

Scatter plot: Petal Length vs Petal Width

PCA 2D visualization of clusters

Centroid patterns (via K-Means results)

📈 Evaluation
Adjusted Rand Index (ARI)

Measures similarity between predicted clusters and true species labels.
Typical ARI ≈ 0.6–0.75, indicating strong clustering despite class overlap.

Elbow Curve

Shows a clear bend at k = 3, confirming three natural clusters.

🎨 Visualizations Produced

Elbow Curve

Petal Length vs Petal Width cluster scatter plot

PCA 2D Projection of clusters

📌 Interpretation Summary

Setosa clusters cleanly due to strong feature separation.

Versicolor and Virginica overlap more, creating some misclusterings.

K-Means performs well on normalized features.

PCA visuals confirm one distinct and two semi-overlapping clusters.

⭐ End of Clustering README
✅ README 3 – Classification & Association Rule Mining
📌 Part A – Classification
Overview

This module trains and evaluates supervised machine learning models to classify Iris species.

🔧 Models Used
1. Decision Tree Classifier

Trained on X_train and evaluated on X_test.
Metrics computed:

Accuracy

Precision

Recall

F1-Score

Classification Report

The tree is visualized using:

plot_tree()

2. KNN Classifier (k = 5)

Used to compare performance with the Decision Tree.

Comparison is based on:

Accuracy

Precision

Recall

F1-Score

📌 Summary of Findings

KNN (k=5) often performs slightly better due to smooth, distance-based classification.

Decision Tree provides higher interpretability but may overfit.

Both models achieve 93–100% accuracy on Iris dataset.

📌 Part B – Association Rule Mining
Overview

Synthetic transactional data is generated to simulate a small retail environment. Association rule mining is applied using the Apriori algorithm.

🛒 Synthetic Data Generation

20–50 transactions

3–8 items per basket

Items drawn from a pool of 20 grocery items

Patterns intentionally added (e.g., {milk, bread}, {beer, diapers} co-occurrences)

Generated using:

random.sample()
random.choices()

🔧 Apriori Algorithm

Using mlxtend, frequent itemsets are mined with:

min_support = 0.2

Association rules generated using:

min_confidence = 0.5

Rules are sorted by lift, and the top 5 strongest rules are displayed.

📈 Interpretation Example:

A rule such as:

{beer} → {diapers}


implies:

Strong customer buying pattern

Useful for product placement, cross-selling, and promotions

High lift indicates the combination occurs more often than random chance

📤 Outputs

Frequent itemsets

Association rules (support, confidence, lift)

Top 5 strongest rules

Business interpretation
