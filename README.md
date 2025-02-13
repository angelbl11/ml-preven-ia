# Comorbidity Risk Prediction Model

## Overview

This repository contains a Machine Learning model designed to predict the risk of developing three common comorbidities: **obesity**, **diabetes**, and **hypertension**. The model utilizes synthetic patient data and Random Forest classifiers to estimate these risks based on various health parameters and demographic factors.

This `README.md` provides a detailed explanation of the model's functionality, training process, and evaluation, offering insights into how it works step-by-step.

## 1. Synthetic Data Generation

Due to data privacy concerns and for demonstration purposes, this project uses synthetic patient data generated using the `Faker` library in Python. The data generation process aims to simulate realistic health data distributions and correlations between comorbidities.

### Process Breakdown:

1.  **Initialization:**

    - The script starts by importing necessary libraries: `Faker` for generating fake data, `numpy` for numerical operations, and `pandas` for data manipulation.
    - `Faker` is initialized to generate demographic data such as age and gender.
    - An empty list `data` is created to store the generated patient data.

2.  **Defining Distribution Parameters (`dist_params`):**

    - A dictionary `dist_params` is defined to hold parameters for generating realistic distributions of key health indicators. These parameters include:
      - `height`, `weight`, `ldl` (LDL cholesterol), `triglycerides`, `glucose` (fasting glucose), `hba1c`, `systolic_bp` (systolic blood pressure), `diastolic_bp` (diastolic blood pressure), and `creatinine` (for both males and females).
    - For each indicator, `dist_params` specifies:
      - `loc`: Mean of the normal distribution.
      - `scale`: Standard deviation of the normal distribution.
      - `min`: Minimum realistic value.
      - `max`: Maximum realistic value.
      - For `triglycerides`, a log-normal distribution is used, specified by `meanlog` and `sigma`.
    - **Note:** The comment `TODO: NEED REAL RANGES DATA` indicates that these distribution parameters should ideally be refined with real-world data for a more accurate simulation.

3.  **Defining Comorbidity Correlation Parameters (`correlation_params`):**

    - The `correlation_params` dictionary defines probabilistic relationships between comorbidities, simulating the increased likelihood of one comorbidity given the presence of another:
      - `obesity_diabetes_prob_boost`: Increases the probability of diabetes if a patient is obese.
      - `obesity_hypertension_prob_boost`: Increases the probability of hypertension if a patient is obese.
      - `diabetes_hypertension_prob_boost`: Increases the probability of hypertension if a patient has diabetes.

4.  **Generating Patient Data:**

    - The code iterates `num_samples` (set to 10,000 in this script) times to create synthetic data for each patient.
    - For each patient, the following features are generated:
      - **Demographics:** `age` (random integer between 18 and 65), `gender` (randomly assigned as male or female, numerically encoded).
      - **Biometric measurements:** `height`, `weight`, `ldl`, `triglycerides`, `glucose`, `hba1c`, `systolic_bp`, `diastolic_bp`, `creatinine`. These are generated using `numpy.random.normal` or `numpy.random.lognormal` based on the distributions defined in `dist_params`, ensuring values are within realistic min/max ranges.
      - **Calculated feature:** `bmi` (Body Mass Index) is calculated from `weight` and `height`.
      - **Genetic predisposition:** `genetic_condition` is randomly assigned (binary: 0 or 1) with a 10% probability of being present.

5.  **Comorbidity Probability Calculation and Assignment:**

    - For each synthetic patient, the probabilities of developing obesity, diabetes, and hypertension are calculated based on a set of rules that consider:
      - Thresholds of `bmi`, `ldl`, `triglycerides`, `glucose`, `hba1c`, blood pressure, `creatinine`, and `age`.
      - Presence of `genetic_condition`.
      - The correlation parameters defined in `correlation_params` to model dependencies between comorbidities.
    - Based on these calculated probabilities and a random factor to introduce variability, binary labels (`obesity`, `diabetes`, `hypertension`) are assigned (1 if the condition is present, 0 otherwise).

6.  **DataFrame Creation:**

    - All generated patient data, stored as dictionaries in the `data` list, is converted into a Pandas DataFrame (`df`). Each row represents a patient, and columns represent the generated features.

7.  **Feature Engineering:**
    - New features are derived from existing ones to potentially improve model performance:
      - **Interaction Features:**
        - `imc_edad_interaccion`: Product of `bmi` and `age`.
        - `glucosa_hba1c_interaccion`: Product of `glucose` and `hba1c`.
      - **Ratio Feature:**
        - `ratio_presion_arterial`: Ratio of `systolic_bp` to `diastolic_bp`.
      - **Categorical Features:**
        - `edad_categoria`: `age` is categorized into age groups (18-29, 30-39, 40-49, 50+).
        - `imc_categoria`: `bmi` is categorized into BMI categories (normal, overweight, obesity grade 1, 2, 3).
    - Categorical features (`edad_categoria`, `imc_categoria`) are then one-hot encoded using `pd.get_dummies` to be used in the machine learning models.

## 2. Model Training

This section details the training process for the Random Forest classifiers to predict each comorbidity (obesity, diabetes, and hypertension).

### Steps:

1.  **Feature and Label Definition:**

    - For each comorbidity, specific feature sets and the corresponding label column from the generated DataFrame (`df`) are defined.
    - **Feature Mapping Dictionaries:** Dictionaries like `feature_name_mapping_obesity`, `feature_name_mapping_diabetes`, and `feature_name_mapping_hipertension` map the short, technical feature names (used in the code and DataFrame) to more descriptive, user-friendly labels for visualizations and interpretation.
    - **Feature Lists:** Lists like `features_obesity`, `features_diabetes`, and `features_hipertension` specify the actual feature columns from the DataFrame that will be used to train each model. These lists are tailored to each comorbidity based on relevant risk factors.
    - **Label Series:** `labels_obesity`, `labels_diabetes`, and `labels_hipertension` extract the target variables (binary comorbidity indicators) from the DataFrame.

2.  **Data Splitting:**

    - For each comorbidity, the data is split into training and testing sets using `train_test_split` from `sklearn.model_selection`.
    - `test_size=0.2` indicates that 20% of the data is reserved for testing, and 80% for training.
    - `random_state=42` ensures reproducibility of the split.
    - `stratify=df['comorbidity_label']` (e.g., `stratify=df['obesidad']`) is crucial for stratified splitting. This ensures that the class distribution (ratio of patients with and without the comorbidity) is maintained in both the training and testing sets, which is important for imbalanced datasets.

3.  **Handling Class Imbalance:**

    - Class imbalance is addressed by calculating class weights using `class_weight.compute_class_weight` and `sklearn.utils.class_weight`. This is especially important for medical datasets where the prevalence of certain conditions may be low.
    - Dictionaries like `class_weight_dict_obesity`, `class_weight_dict_diabetes`, and `class_weight_dict_hipertension` store these weights, which are later used during model training to give more importance to the minority class.

4.  **Feature Scaling:**

    - Numerical features are scaled using `StandardScaler` from `sklearn.preprocessing`. Scaling is essential for algorithms sensitive to feature scales, like Random Forests when interactions or distance-based metrics are involved (though Random Forests are less scale-sensitive than some other algorithms, it's generally good practice).
    - For each comorbidity model, a `StandardScaler` is initialized and fitted only on the **numerical features** within the **training set** (`X_train_...[numeric_features_...]`). This is critical to prevent data leakage from the test set into the training process.
    - Only the numerical features are transformed. Categorical features (after one-hot encoding) are left unscaled as they are already in a suitable binary representation.
    - Both the training and **testing** sets are transformed using the scaler fitted on the training data.

5.  **Model Training with GridSearchCV and Cross-Validation:**
    - **Hyperparameter Tuning with GridSearchCV:** `GridSearchCV` from `sklearn.model_selection` is used to find the best hyperparameters for the Random Forest Classifier for each comorbidity.
    - **RandomForestClassifier:** `RandomForestClassifier` from `sklearn.ensemble` is chosen as the base model due to its robustness, ability to handle mixed data types, and interpretability.
    - **Parameter Grids:** `param_grid_obesity`, `param_grid_diabetes`, and `param_grid_hipertension` define the hyperparameter values to be tested by `GridSearchCV`. These include:
      - `n_estimators`: Number of trees in the forest.
      - `max_depth`: Maximum depth of the trees.
      - `min_samples_split`: Minimum number of samples required to split an internal node.
      - `min_samples_leaf`: Minimum number of samples required to be at a leaf node.
      - `class_weight`: For diabetes and hypertension, both `'balanced'` and the calculated class weight dictionaries are tested. For obesity (in this simplified example), only `'balanced'` is tested.
    - **Cross-Validation:** `StratifiedKFold` with `n_splits=3` is used for cross-validation within `GridSearchCV`. Stratified K-fold ensures that in each fold, the class distribution is maintained, which is important for robust model evaluation, especially with imbalanced data. `shuffle=True` and `random_state=42` ensure reproducibility of the cross-validation process.
    - **Scoring Metric:** `scoring='accuracy'` is used to evaluate model performance during Grid Search. This can be changed to other metrics like 'f1', 'roc_auc', etc., depending on the project's goals.
    - **Parallel Processing:** `n_jobs=-1` utilizes all available CPU cores to speed up the Grid Search process.
    - **Model Fitting:** `grid_search_...fit(...)` performs the Grid Search, training and evaluating the `RandomForestClassifier` for all combinations of hyperparameters defined in the `param_grid` using cross-validation.
    - **Best Model Selection:** `best_obesity_model = grid_search_obesity.best_estimator_` extracts the best trained `RandomForestClassifier` model found by `GridSearchCV` based on the chosen scoring metric (accuracy). The best hyperparameters for each model are printed to the console.

## 3. Model Evaluation

After training, the best models for each comorbidity are evaluated on the held-out test set to estimate their performance on unseen data.

### Evaluation Metrics:

- **Accuracy:** Calculated using `accuracy_score`, provides a general measure of correctness (percentage of correctly classified instances).
- **Classification Report:** Generated using `classification_report`, provides detailed performance metrics per class, including:
  - **Precision:** Of all instances predicted as positive, what proportion is actually positive?
  - **Recall:** Of all actual positive instances, what proportion is correctly identified as positive?
  - **F1-score:** Harmonic mean of precision and recall, balancing both metrics.
  - **Support:** Number of actual instances in each class in the test set.

### Risk Categorization Function:

- A `risk_categority` function is defined to categorize the predicted probabilities into risk levels: "Bajo" (Low), "Medio" (Medium), and "Alto" based on predefined thresholds (`medium_low_umbral=0.5`, `medium_high_umbral=0.8`).
- This function is applied to the predicted probabilities (`y_pred_prob_...`) to get a risk category for each patient in the test set. However, in the current evaluation output, only binary predictions are directly evaluated for accuracy and classification report. The risk categories are calculated but not explicitly used in the performance metrics shown in the output.

### Evaluation Process:

1.  **Prediction on Test Set:** For each comorbidity model (`best_obesity_model`, `best_diabetes_model`, `best_hipertension_model`), predictions are made on the corresponding scaled test set (`X_test_obesity_scaled`, `X_test_diabetes_scaled`, `X_test_hipertension_scaled`):

    - `y_pred_obesity_binary = best_obesity_model.predict(...)`: Generates binary predictions (0 or 1).
    - `y_pred_prob_obesity = best_obesity_model.predict_proba(...)[:, 1]`: Generates probability scores for the positive class (class 1).

2.  **Performance Metrics Calculation and Output:**
    - For each comorbidity:
      - Accuracy is calculated and printed.
      - Classification report is generated and printed, providing precision, recall, F1-score, and support for each class.

## 4. Model and Scaler Export

For deployment and future use, the trained best models and their corresponding scalers are saved to files using `pickle`.

### Export Steps:

1.  **Directory Creation:** A directory named `models` is created (if it doesn't exist) in the parent directory of the script using `os.makedirs(MODEL_DIR, exist_ok=True)`.
2.  **Saving Scalers and Models:**
    - For each comorbidity (obesity, diabetes, hypertension):
      - The corresponding `StandardScaler` object (`scaler_obesity`, `scaler_diabetes`, `scaler_hipertension`) is saved to a `.pkl` file (e.g., `scaler_obesity.pkl`) in the `models` directory using `pickle.dump`.
      - The best trained `RandomForestClassifier` model (`best_obesity_model`, `best_diabetes_model`, `best_hipertension_model`) is saved to a `.pkl` file (e.g., `obesity_model.pkl`) in the `models` directory using `pickle.dump`.
3.  **Confirmation Message:** A message is printed to the console indicating that models and scalers have been successfully saved to the `models` directory.

## 5. Training Graphics

To visualize and understand the model's behavior and feature importance, several plots are generated after training:

### Feature Importance Plots (Bar Charts):

- For each comorbidity model:
  - **Feature Importances Extraction:** `feature_importances_... = best_..._model.feature_importances_` extracts the feature importance scores from the trained Random Forest model.
  - **DataFrame Creation:** `importance_df_... = pd.DataFrame({'feature': features_..., 'importance': feature_importances_...})` creates a Pandas DataFrame to store feature names and their importance scores.
  - **Feature Ordering (for Obesity and Hypertension):** For obesity and hypertension models, specific features (e.g., 'IMC categories', 'ldl', 'triglycerides' for obesity; 'presion_arterial_sistolica', 'presion_arterial_diastolica', 'creatinina', 'ldl' for hypertension) are prioritized in the plot order. This is done to highlight the importance of these clinically relevant features. The rest of the features are appended afterwards, sorted by importance.
  - **Descriptive Labels:** `descriptive_feature_labels_... = [feature_name_mapping_...get(feature_short_name, feature_short_name) for feature_short_name in ordered_importance_df_...['feature']]` retrieves descriptive labels for the features from the mapping dictionaries to make the plots more readable. If a short name is not found in the mapping, the short name itself is used as the label.
  - **Bar Plot Generation:** `plt.bar(...)` from `matplotlib.pyplot` creates a bar chart showing feature importance.
    - The x-axis displays the descriptive feature labels.
    - The y-axis represents the relative importance of each feature.
    - The plot is customized with labels, a title, rotated x-axis labels for readability, and tight layout adjustment.

### Confusion Matrices:

- For each comorbidity model:
  - **Confusion Matrix Calculation:** `cm_... = confusion_matrix(y_test_..., y_pred_..._binary)` calculates the confusion matrix using `sklearn.metrics.confusion_matrix`, comparing the true labels (`y_test_...`) with the binary predictions (`y_pred_..._binary`).
  - **Confusion Matrix Display:** `disp_... = ConfusionMatrixDisplay(confusion_matrix=cm_..., display_labels=['No Comorbidity', 'Comorbidity'])` creates a `ConfusionMatrixDisplay` object from `sklearn.metrics.ConfusionMatrixDisplay` for visualizing the confusion matrix with class labels ('No Comorbidity', 'Comorbidity').
  - **Plotting:** `disp_....plot(cmap=plt.cm.ColorMap)` generates and displays the confusion matrix plot with a specified colormap (Blues for obesity, Reds for diabetes, Greens for hypertension) using `disp_...plot()`.
    - The plot is customized with a title and axis labels ('Prediction' and 'Real Value').

### ROC Curves and AUC Scores:

- For each comorbidity model:
  - **ROC Curve and AUC Calculation:** `fpr_..., tpr_..., _ = roc_curve(y_test_..., y_pred_prob_...)` calculates the False Positive Rate (FPR), True Positive Rate (TPR), and thresholds for the ROC curve using `sklearn.metrics.roc_curve`. `roc_auc_... = auc(fpr_..., tpr_...)` calculates the Area Under the ROC Curve (AUC) using `sklearn.metrics.auc`.
  - **ROC Curve Plot Generation:** `plt.plot(...)` from `matplotlib.pyplot` creates the ROC curve plot.
    - The ROC curve is plotted in color with a label showing the AUC score.
    - A diagonal dashed line (navy color) representing a random classifier is added for reference.
    - The plot is customized with axis labels ('False Positive Rate', 'True Positive Rate'), a title ('ROC Curve for Comorbidity Model'), and a legend showing the AUC value in the lower right corner.

These visualizations provide insights into feature importance, model performance in terms of true/false positives and negatives (confusion matrix), and overall model discriminative ability across different classification thresholds (ROC curve and AUC).

## 6. Libraries Used

- **pandas**: Data manipulation and analysis.
- **numpy**: Numerical computing.
- **matplotlib**: Data visualization.
- **pickle**: Serialization of Python objects (saving models and scalers).
- **os**: Operating system functionalities (directory creation).
- **faker**: Generation of synthetic data.
- **scikit-learn (sklearn)**: Machine learning library:
  - `sklearn.model_selection`: Data splitting (`train_test_split`, `StratifiedKFold`), hyperparameter tuning (`GridSearchCV`).
  - `sklearn.ensemble`: Random Forest Classifier (`RandomForestClassifier`).
  - `sklearn.metrics`: Model evaluation metrics (`accuracy_score`, `classification_report`, `confusion_matrix`, `roc_curve`, `auc`, `ConfusionMatrixDisplay`).
  - `sklearn.preprocessing`: Feature scaling (`StandardScaler`).
  - `sklearn.utils`: Handling class imbalance (`class_weight`).

## 7. Usage

To run this model training and evaluation script, you need to:

1.  **Install the required libraries:**
    ```bash
    pip install pandas numpy matplotlib scikit-learn faker
    ```
2.  **Run the Python script:** Execute the Python notebook or script containing the provided code.

After execution, the trained models and scalers will be saved in the `models` directory, and performance evaluation metrics and visualizations will be displayed.

## 8. API Endpoints Documentation

The application also exposes an API built with FastAPI, allowing real-time predictions using the trained models. Below are the details of each endpoint:

### Base URL

When running the API server locally, the base URL is typically:

http://localhost:8000

API documentation is automatically generated and can be accessed via:

- **Swagger UI:** [http://localhost:8000/docs](http://localhost:8000/docs)
