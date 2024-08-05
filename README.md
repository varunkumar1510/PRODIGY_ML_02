# PRODIGY_ML_02
Customer Segmentation with K-Means Clustering  TThis project implements K-means clustering to segment customers based on their purchase history, focusing on features like age, annual income, and spending score. The goal is to group customers into distinct clusters to enhance targeted marketing strategies and improve customer understanding.

# Customer Segmentation with K-Means Clustering

## Project Overview

This repository contains a project on customer segmentation using K-means clustering. The aim is to analyze customer data and group them into distinct segments based on their purchasing behavior. Effective segmentation helps in targeting marketing efforts and improving customer relations.

## Dataset

The dataset used in this project is from Kaggle and includes information on 200 customers. It contains the following features:
- **CustomerID**: Unique identifier for each customer.
- **Gender**: Customer's gender (Male/Female).
- **Age**: Age of the customer.
- **Annual Income (k$)**: Annual income in thousands of dollars.
- **Spending Score (1-100)**: Score assigned based on the customer's spending behavior.

You can access the dataset [here](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python).

## Methodology

1. **Data Preprocessing**:
   - Convert categorical variables (Gender) to numerical values.
   - Standardize features to ensure they have a similar scale.

2. **Optimal Clustering**:
   - Use the Elbow Method to determine the optimal number of clusters by plotting Within-Cluster Sum of Squares (WCSS).
   - Evaluate clustering effectiveness using the Silhouette Score.

3. **K-means Clustering**:
   - Apply the K-means algorithm with the chosen number of clusters.
   - Save the clustered results in a CSV file.

## Results

The clustering results are saved in `Clustered_Customers.csv`, which contains the original data along with cluster labels.

## Files

- `Customer_Segmentation_KMeans.ipynb`: Jupyter Notebook with the clustering implementation and analysis.
- `Clustered_Customers.csv`: Output file with the clustered customer data.

## Usage

1. Clone this repository.
2. Open `Customer_Segmentation_KMeans.ipynb` in a Jupyter Notebook environment or Google Colab.
3. Ensure the dataset path is correctly specified.
4. Run the notebook to perform clustering and analyze results.

## Requirements

- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

