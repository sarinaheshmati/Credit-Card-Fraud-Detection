# Credit Card Fraud Detection Project

## Overview

This project focuses on detecting fraudulent transactions from credit card data using classification and clustering algorithms. The dataset used is from Kaggle, detailing credit card transactions in Europe over two days in September 2013. Out of 284,807 transactions, 492 are fraudulent, indicating a highly imbalanced dataset. The project employs Logistic Regression for classification and K-means for clustering, along with essential data preprocessing steps.

## Dataset

The dataset contains 31 numerical columns, with features V1 to V28 derived from Principal Component Analysis (PCA), and two additional columns: `Time` and `Amount`, which are raw values. The `Class` column indicates whether a transaction is fraudulent (`1`) or not (`0`). The data is highly imbalanced, which necessitates special considerations during preprocessing and model evaluation.

**Link to the dataset:** [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

## Project Structure

- **Data Exploration:** 
  - Initial examination of the dataset revealed no null values, and all columns (except `Class`) consist of float64 numerical data.
  - The dataset’s class distribution was found to be 99.83% non-fraudulent and 0.17% fraudulent transactions, emphasizing the need for handling imbalanced data.
  - Distributions of `Time` and `Amount` were analyzed, revealing that `Time` is not a useful feature for model training, while `Amount` shows distinct distributions for fraudulent and non-fraudulent transactions. Plotting the density between the two types of labels, it became clear that the `Time` feature was not a significant differentiator between fraudulent and non-fraudulent transactions, in contrast with `Amount`. Below is the density plot that guided this decision:
<img width="674" alt="Screenshot 1403-06-06 at 8 38 09 AM" src="https://github.com/user-attachments/assets/9427dbcd-6032-42f2-8207-d790124e2e0c">
<img width="719" alt="Screenshot 1403-06-06 at 8 38 56 AM" src="https://github.com/user-attachments/assets/91adb821-f5d1-46f9-a5ae-92e495cc35b7">

- **Data Preprocessing:**
  - The `Time` column was removed, and features (`X`) and labels (`y`) were extracted.
  - The dataset was split into training (80%) and testing (20%) sets.
  - Only the `Amount` column was normalized using `StandardScaler` to avoid information leakage. Normalization was done separately for training and test data to avoid data leakage.

- **Modeling:**
  - **Logistic Regression:** This model was chosen due to its efficiency with large datasets, as opposed to SVM, which could be computationally expensive after oversampling.
    - **Without Oversampling:** Logistic Regression yielded a high accuracy of 99.91%, but due to the imbalanced nature of the dataset, accuracy alone was not a reliable metric. The AUPRC score was 0.72, indicating good performance.
    - **With Oversampling:** SMOTE (Synthetic Minority Over-sampling Technique) was used to balance the training data, but the model's performance decreased, with an AUPRC score of 0.49, indicating that oversampling might not always be beneficial for this dataset.

  - **K-means Clustering:**
    - **Elbow Method:** The optimal number of clusters was determined to be 2, aligning with the two classes (fraudulent and non-fraudulent).
    - **Clustering Results:** The clusters were analyzed to check their composition regarding fraudulent and non-fraudulent transactions. However, the clustering approach showed poor performance in correctly segregating fraudulent transactions, with an accuracy of 0.53 and an AUC score indicating subpar performance.

## Evaluation Metrics

Given the imbalanced nature of the dataset, AUPRC (Area Under the Precision-Recall Curve) was used for evaluation. This metric is more informative than accuracy for imbalanced datasets as it considers the model's precision and recall, which are critical for fraud detection tasks.

- **Precision:** The ratio of true positive predictions to the total predicted positives, indicating the correctness of positive predictions.
- **Recall:** The ratio of true positive predictions to all actual positives, measuring the model’s ability to identify fraudulent transactions.
- **AUPRC:** Used to evaluate the overall performance of the model, particularly in imbalanced datasets. The AUPRC score closer to 1 indicates better performance.

## Conclusion

- Logistic Regression without oversampling performed better for this dataset, achieving a good balance between precision and recall.
- K-means clustering was not effective in distinguishing between fraudulent and non-fraudulent transactions, highlighting the challenges of unsupervised learning in such scenarios.

## Future Work

- Explore other machine learning models like Random Forests, XGBoost, or Neural Networks to improve detection accuracy.
- Implement advanced techniques to handle imbalanced data, such as ensemble methods or anomaly detection approaches.
- Further feature engineering to uncover additional insights from the data.
