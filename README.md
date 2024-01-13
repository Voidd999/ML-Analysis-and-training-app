# AutoML Streamlit App


This Streamlit app is a machine learning analysis and training tool that allows users to upload datasets, perform exploratory data analysis (EDA), and build predictive models. It provides an interactive interface for data exploration and model evaluation.


## Features

- Upload CSV datasets for analysis and modeling.
- Conduct exploratory data analysis (EDA) to understand the dataset.
- Visualize summary statistics, correlation matrices, and missing value information.
- Create histograms, bar plots, box plots, and violin plots for data visualization.
- Build and evaluate machine learning models, including linear regression, random forest, gradient boosting, and k-nearest neighbors.
- View model evaluation metrics such as mean absolute error (MAE), score, and max error.
- Visualize feature importance for tree-based models (Random Forest, Gradient Boosting).
- Make predictions on new data using the trained model.
- Save and download the trained model for future use.

This application is currently in experimental stage. Deployment is not recommended as data security concerns have not been addressed.
## Sample Datsets
## You can use these sample datasets from Kaggle -
https://www.kaggle.com/datasets/sinamhd9/concrete-comprehensive-strength
https://www.kaggle.com/datasets/joebeachcapital/subway-nutrition/code
https://www.kaggle.com/datasets/starbucks/starbucks-menu

## Usage

1. Clone this repository to your local machine

2. Install Requirements
   ```shell
   pip install -r requirements.txt
3. Run The App
   ```shell
   streamlit run app.py
