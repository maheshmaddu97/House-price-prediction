# House-price-prediction
House Price Prediction is a Machine Learning web application built with Python, Pandas, Scikit-learn, and Flask. It uses Multiple Linear Regression to predict house prices based on various property-related features. The project includes data preprocessing, feature engineering, categorical encoding, model evaluation using R² and RMSE
# 🏠 House Price Prediction Using Machine Learning

## 📌 Project Overview

This project uses Machine Learning regression techniques to predict house prices based on property characteristics. It demonstrates the end-to-end workflow of a Machine Learning project, including data exploration, data preprocessing, feature analysis, model training, and performance evaluation.

The project uses Python and popular data science libraries to analyze housing data and build a model that estimates property prices from relevant features.

## 🎯 Objectives

- Analyze housing data to understand the factors affecting house prices.
- Perform Exploratory Data Analysis (EDA) using visualizations and statistical methods.
- Identify and handle missing values, duplicate records, and potential outliers.
- Prepare numerical and categorical features for Machine Learning.
- Train a regression model to predict house prices.
- Evaluate model performance using appropriate regression metrics.

## 📊 Dataset Description

The project uses the `Housing.csv` dataset, which contains **545 rows and 13 columns**.

The dataset includes the following features:

| Feature | Description |
|---|---|
| `price` | Target variable — house price |
| `area` | Area of the property |
| `bedrooms` | Number of bedrooms |
| `bathrooms` | Number of bathrooms |
| `stories` | Number of stories |
| `mainroad` | Whether the property is connected to the main road |
| `guestroom` | Whether a guest room is available |
| `basement` | Whether a basement is available |
| `hotwaterheating` | Whether hot-water heating is available |
| `airconditioning` | Whether air conditioning is available |
| `parking` | Number of parking spaces |
| `prefarea` | Whether the property is in a preferred area |
| `furnishingstatus` | Furnishing status of the property |

**Target variable:** `price`

The goal is to predict the target variable using the remaining property features.

## 🛠️ Technologies Used

- **Python** — Programming language
- **Pandas** — Data manipulation and analysis
- **NumPy** — Numerical computations
- **Matplotlib** — Data visualization
- **Seaborn** — Statistical visualization
- **Scikit-learn** — Data preprocessing, model training, and evaluation
- **Jupyter Notebook / Google Colab** — Development environment

## 🔄 Project Workflow

### 1. Data Loading
- Load the housing dataset using Pandas.
- Inspect the dataset using `head()`, `info()`, `shape`, and `describe()`.

### 2. Exploratory Data Analysis (EDA)
- Analyze numerical and categorical variables.
- Examine feature distributions.
- Visualize correlations using a heatmap.
- Identify potential outliers using boxplots.

### 3. Data Preprocessing
- Check for missing values and duplicate records.
- Investigate outliers and unusual values.
- Convert categorical variables into numerical representations using suitable encoding techniques.
- Separate independent variables (`X`) and the dependent variable (`y`).

### 4. Model Training
- Split the dataset into training and testing sets.
- Train a regression model using Scikit-learn.
- Generate house-price predictions from the input features.

### 5. Model Evaluation
Evaluate the trained model using appropriate regression metrics:

- **R² Score:** Measures how much variation in the target variable the model explains.
- **Mean Absolute Error (MAE):** Measures the average absolute prediction error.
- **Root Mean Squared Error (RMSE):** Measures prediction error while penalizing larger errors more heavily.

## 📈 Machine Learning Algorithm

**Linear Regression**

Linear Regression is a supervised Machine Learning algorithm used to predict continuous numerical values.

In this project, the model learns the relationship between property characteristics and house prices.

The model can be evaluated on unseen test data to understand how well it generalizes to new properties.

## 📁 Project Structure

```text
House-price-prediction/
│
├── Housing.csv
├── house-price-prediction.ipynb
└── README.md
```

*Note: Update the file names in this section if your repository uses a different structure.*

## 🚀 How to Run the Project

### Prerequisites

Install Python and the required libraries.

### Step 1: Clone the repository

```bash
git clone https://github.com/maheshmaddu97/House-price-prediction.git
```

### Step 2: Navigate to the project folder

```bash
cd House-price-prediction
```

### Step 3: Install the dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### Step 4: Launch Jupyter Notebook

```bash
jupyter notebook
```

Open the project notebook and execute the cells in sequence.

Alternatively, upload the notebook and dataset to Google Colab and run the project there.

## 💡 Key Learnings

- Data cleaning and preprocessing using Pandas.
- Exploratory Data Analysis and correlation analysis.
- Feature encoding for categorical variables.
- Training and evaluating regression models.
- Understanding prediction errors and model performance.
- Building a complete Machine Learning workflow.

## 🔮 Future Improvements

- Compare Linear Regression with Ridge, Lasso, and Random Forest Regression.
- Perform cross-validation and hyperparameter tuning.
- Improve feature engineering and investigate influential outliers.
- Build a web application using Flask or Streamlit.
- Deploy the model so users can enter property details and obtain price predictions.

## 👨‍💻 Author

**Mahesh**

Mechanical Engineering Graduate | Aspiring Data Analyst / Data Scientist

**GitHub:** https://github.com/maheshmaddu97

