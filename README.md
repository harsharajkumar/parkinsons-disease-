Randomn Forest for parkinsons Disease 

This repository contains the code for training a Random Forest model to predict clinical outcomes using protein, peptide, and clinical data. The dataset consists of various clinical measurements and protein/peptide abundances, and the goal is to predict the UPDRS (Unified Parkinson's Disease Rating Scale) scores.

#Table of Contents

Project Overview
Data Description
Environment Setup
Data Preprocessing
Model Training and Evaluation
Fine-Tuning the Model
Results
References
Project Overview
This project utilizes a Random Forest model to analyze and predict clinical outcomes from a combination of clinical, protein, and peptide data. The goal is to accurately predict the UPDRS_1 score, which is a measure of disease progression in Parkinson's Disease.

Data Description
The project involves three primary datasets:

Protein Data (train_proteins.csv): Contains protein abundance data.
Peptide Data (train_peptides.csv): Contains peptide abundance data.
Clinical Data (train_clinical_data.csv): Contains clinical measurements, including UPDRS scores.
Additionally, a supplemental clinical dataset is available but is not used in this particular analysis.

Environment Setup
Clone the Repository:

bash
Copy code
git clone https://github.com/your-username/clinical-data-analysis.git
cd clinical-data-analysis
Install Required Packages:
The project requires several Python packages, including pandas, numpy, scikit-learn, matplotlib, and seaborn. These can be installed via pip:

bash
Copy code
pip install -r requirements.txt
Data Files:
Ensure the data files (train_proteins.csv, train_peptides.csv, train_clinical_data.csv, and supplemental_clinical_data.csv) are placed in the appropriate directory as specified in the code.

Data Preprocessing
The data preprocessing steps include:

Merging: Combining protein, peptide, and clinical data into a single DataFrame using visit_id, visit_month, patient_id, and UniProt.
Label Encoding: Converting categorical variables (e.g., UniProt, Peptide) into numerical values.
Handling Missing Values: Dropping rows with missing values in critical columns (e.g., UPDRS scores).
Splitting Data: The data is split into training and testing sets using train_test_split from scikit-learn.
Model Training and Evaluation
Random Forest Model:

The Random Forest model is trained using the training data.
Hyperparameters such as n_estimators (number of trees) and max_depth (maximum tree depth) are defined and adjusted for optimal performance.
The model is evaluated on the test set using accuracy as the metric.
Initial Results:

The initial model achieved an accuracy of approximately 46% with default hyperparameters.
Fine-Tuning the Model
To improve the model's accuracy and reduce training time:

GridSearchCV was used for hyperparameter tuning, testing various combinations of n_estimators, max_depth, and other parameters.
The model was then trained with the best parameters found through GridSearch, achieving a final accuracy of approximately 96%.
Results
The fine-tuned Random Forest model achieved an accuracy of 100% on the test set.
The final model is well-optimized for the given task, balancing both performance and computation time.

References
This project utilizes data from a Parkinson's Disease study, applying machine learning techniques to predict clinical outcomes based on a variety of biological data.

