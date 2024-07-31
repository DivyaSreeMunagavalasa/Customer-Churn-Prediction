# Customer Churn Analysis and Prediction

## Overview
This project analyzes and predicts customer churn for a telecommunications company using a dataset of customer demographics and service usage.

## Dataset
- **File**: `telco-churn.csv`
- **Description**: Contains information on customer churn, demographics, and service usage.

## Prerequisites
- Python 3.x
- Libraries: pandas, numpy, seaborn, matplotlib, scikit-learn, imbalanced-learn

## Installation
1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
2. **Create a requirements.txt File**
   ```bash
   pandas
   numpy
   seaborn
   matplotlib
   scikit-learn
   imbalanced-learn
3. To run the project locally, you need to have Python installed. Clone the repository and install the required libraries using:
   ```bash
   pip install -r requirements.txt

4. **Usage**
   Run the Jupyter Notebook to see the analysis and predictions:
   ```bash
   jupyter notebook Customer_Churn_Analysis.ipynb

## Data Preprocessing
Data preprocessing steps include:
- Handling missing values.
- Encoding categorical variables.
- Normalizing numerical variables.
- Splitting the dataset into training and testing sets.

## Data Analysis
We analyze the data to find patterns and insights, such as the relationship between customer churn and features like contract type, payment method, and monthly charges.

## Model Building
We build several machine learning models to predict customer churn:
- Decision Tree Classifier
- Random Forest Classifier
We also address data imbalance using techniques like SMOTEENN.

## Results
The models' performance is evaluated using metrics such as recall, precision, and F1 score. The confusion matrix provides insights into the accuracy of the predictions.

## Contributing
Contributions are welcome! Please fork the repository and create a pull request to add new features or improve existing code.

   

   
