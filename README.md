# Titanic Survival Prediction (Kaggle Competition)

##  Overview
This project is a complete end-to-end solution for the **Kaggle Titanic: Machine Learning from Disaster** competition.  
It predicts passenger survival using demographic and socio-economic features such as age, gender, passenger class, and fare.  

The pipeline covers **data exploration, preprocessing, feature engineering, model training, and Kaggle submission generation**.  
Final model used: **Logistic Regression**  
**Leaderboard Score:** 0.767

---

##  Objectives
- Understand the structure and characteristics of the Titanic dataset.
- Explore relationships between passenger attributes and survival.
- Engineer meaningful new features to improve model accuracy.
- Build and evaluate predictive models using cross-validation.
- Generate a submission file for Kaggle evaluation.

---

##  Dataset
Dataset source: [Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic)

Files used:
- `train.csv` — labeled training data.  
- `test.csv` — unlabeled test data.  
- `titanic_submission_13.csv` — final predictions submitted to Kaggle.

---

##  Workflow Breakdown

### 1. Understanding the Nature of Data
- Loaded and merged train/test sets for unified processing.  
- Inspected dataset statistics with `describe()`.  
- Identified categorical and numerical columns.

### 2. Processing Data Based on Type
- Separated numeric (`Age`, `SibSp`, `Parch`, `Fare`) and categorical (`Sex`, `Pclass`, `Embarked`, etc.) features.  
- Visualized numeric distributions using **Matplotlib** and **Seaborn**.  
- Explored categorical variable distributions and their relationship with survival rates.

### 3. Feature Engineering
Created several new meaningful features:
- **cabin_multiple:** Number of cabins per passenger.  
- **cabin_adv:** First letter of the cabin (handles missing values).  
- **numeric_ticket:** Binary indicator for numeric vs alphanumeric tickets.  
- **ticket_letters:** Extracted ticket prefixes.  
- **name_title:** Extracted titles such as Mr, Mrs, Miss, Master, etc.  
- **norm_fare:** Log-normalized fare for stable scaling.

### 4. Data Preprocessing
- Filled missing `Age` and `Fare` values with their median.  
- Dropped null rows for `Embarked` (only 2 missing).  
- Converted `Pclass` to categorical type.  
- Applied **one-hot encoding** using `pd.get_dummies()` for categorical variables.

### 5. Train-Test Split
- Split combined dataset into training and test sets using the `train_test` flag.  
- Defined feature matrices (`X_train`, `X_test`) and target (`y_train`).

### 6. Scaling Data
- Standardized numerical columns (`Age`, `SibSp`, `Parch`, `norm_fare`) using **StandardScaler** to normalize data for Logistic Regression.

### 7. Model Training
- Model: **Logistic Regression**  
- Evaluation: 5-Fold Cross Validation (`cross_val_score`)  
- Average CV Score: ~0.767  
- Model trained on scaled training data.

### 8. Submission
- Predicted survival outcomes for test dataset.  
- Saved predictions to `titanic_submission_13.csv` in required Kaggle format.

---

##  Results
| Metric | Value |
|:--|:--|
| **Model** | Logistic Regression |
| **Cross-Validation Mean Accuracy** | ~0.767 |
| **Kaggle Leaderboard Score** | **0.767** |

---

##  Key Insights
- **Gender** and **Passenger Class** were strong indicators of survival.  
- **Female** and **higher-class passengers** had significantly higher chances of survival.  
- Titles extracted from names (e.g., Mr, Miss, Master) improved model accuracy.  
- Log-normalizing `Fare` stabilized its impact on the model.

---

##  Tools and Technologies
- **Language:** Python  
- **Libraries:** NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn  
- **Environment:** Kaggle Notebook / VS Code / Google Colab 
- **Version Control:** Git & GitHub  

---
#### Author --> Nilay Shahane (AI agent , Machine learning , Full Stack Developer) , nilayshahane@gmail.com


