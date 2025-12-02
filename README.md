# Crop Recommendation System: A Machine Learning Approach

This project implements a machine learning-based system to recommend the most suitable crop for cultivation based on soil conditions and environmental factors. The primary model used is a **Support Vector Machine (SVM)**, which achieved extremely high accuracy in classifying the optimal crop.



---

## 1. Dataset Overview

The project uses the `Crop_recommendation.csv` dataset, which contains 2200 records of crop cultivation data.

| Feature Name | Description |
| :--- | :--- |
| **N** | Ratio of **Nitrogen** content in the soil |
| **P** | Ratio of **Phosphorus** content in the soil |
| **K** | Ratio of **Potassium** content in the soil |
| **temperature** | Temperature in Celsius (°C) |
| **humidity** | Relative humidity (%) |
| **ph** | pH value of the soil |
| **rainfall** | Rainfall in mm |
| **label** | The target crop (22 unique classes) |

**Key Statistics:**
* **Total Records:** 2200
* **Total Features:** 7 (Numerical)
* **Target Variable:** `label` (Categorical, 22 types of crops)
* **Data Quality:** The data was clean, with **no missing values** and **no duplicate records**.

---

## 2. Data Preparation

Before training, the data underwent several essential preprocessing steps:

1.  **Feature/Target Separation:** Features (**X**) were separated from the target variable (**y**).
2.  **Label Encoding:** The categorical crop names (`label`) were converted into numerical representations using the **LabelEncoder**.
3.  **Train-Test Split:** The dataset was split into **training (80%)** and **testing (20%)** sets.
4.  **Feature Scaling:** To normalize the data, prevent features with larger magnitudes from dominating, and standardize the distributions, both **MinMaxScaler** and **StandardScaler** were applied.

---

## 3. Models Used

### A. Primary Model: Support Vector Machine (SVM)

The project focused on the Support Vector Classifier (SVC), a highly effective and robust algorithm for multi-class classification tasks.

* **Core Concept:** SVM works by finding an optimal hyperplane in a high-dimensional feature space that maximally separates the 22 different crop classes. This boundary is chosen to have the largest margin between the data points of different classes, which improves the model's generalization ability to unseen data.
* **Non-Linearity (Kernel Trick):** Since the relationship between environmental factors and crop type is often complex and non-linear, the SVM utilizes a kernel function (such as the Radial Basis Function or RBF kernel) to implicitly map the original 7-dimensional feature data into a higher-dimensional space where the classes become linearly separable.
* **Hyperparameter Tuning:** GridSearchCV was employed to optimize the model's performance by searching for the best combination of critical hyperparameters:
  * ``C``**(Regularization Parameter):** Controls the trade-off between achieving a wide margin and minimizing the classification error (misclassifications). A smaller C prioritizes a larger margin, while a larger C enforces stricter separation of training points.
  * ``gamma`` **(Kernel Coefficient):** Defines how far the influence of a single training example reaches. A small gamma results in a smoother decision boundary; a large gamma creates a more rugged, complex boundary, which can lead to better fit on the training data but increased risk of overfitting.

### B. Secondary Model: Random Forest Classifier (RFC)

A Random Forest Classifier was also trained and evaluated as a high-performance, robust ensemble benchmark model.

* **Core Concept:** RFC operates by constructing a large number of independent Decision Trees during training (usually 100 or more). Each tree is built using a random subset of the data and a random subset of the features.
* **Prediction:** The final recommendation is determined by combining the predictions of all individual trees through majority voting. This ensemble approach significantly cancels out the variance and bias of individual trees, making the Random Forest highly resistant to overfitting and providing exceptional stability and accuracy, which is why it performed almost identically to the highly-tuned SVM.

---

## 4. Results and Evaluation

Both models achieved near-perfect performance on the test set, confirming the strong relationship between the environmental features and the optimal crop choice.

| Key Metrics | Performance by SVM | Performance by RFC |
| :--- | :--- | :--- |
| F1 Score | **99.32%** | **99.26%** |
| Precision | **99.37%** | **99.26%** |
| Support | **40** | **440** |
| Recall | **99.32%** | **99.33%** |

**Evaluation Metrics:**
* **Classification Report:** Used to assess **Precision, Recall, and F1 Score** for each crop class.
* **Confusion Matrix:** Used to visualize and confirm minimal misclassifications, indicating a highly effective classifier.


<table align="center">
  <tr>
    <td align="center">
      <div style="width:450px; height:450px; display:flex; align-items:center; justify-content:center; background:#fff; border:1px solid #ddd;">
        <img src="https://github.com/user-attachments/assets/97d252e0-d446-4417-8890-101f496d006d"
             alt="Confusion Matrix SVM"
             style="max-width:100%; max-height:100%; object-fit:contain;">
      </div>
      <em>Fig 1(a). Confusion Matrix for SVM</em>
    </td>

  <td align="center">
      <div style="width:450px; height:450px; display:flex; align-items:center; justify-content:center; background:#fff; border:1px solid #ddd;">
        <img src="https://github.com/user-attachments/assets/23dbaa2e-bd64-4022-9be6-9698679b5d33"
             alt="Confusion Matrix RFC"
             style="max-width:100%; max-height:100%; object-fit:contain;">
      </div>
      <em>Fig 1(b). Confusion Matrix for RFC</em>
    </td>
  </tr>
</table>



---

## 5. Predictive System

The final system uses the trained and optimized SVM model for making new recommendations:

* **Data Serialization:** The final fitted model and the **MinMaxScaler** and **StandardScaler** objects are saved using the `pickle` library.
* **Deployment Ready:** Saving the scalers is crucial, as any new input data must be transformed using the *exact same* scaling parameters learned during training before being passed to the model for a reliable prediction.
