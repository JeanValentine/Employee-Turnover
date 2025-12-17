# Employee-Turnover

## Problem Scenario

Portobello Tech is an app innovator who has devised an intelligent way of predicting employee turnover within the company. It periodically evaluates employees' work details, including the number of projects they worked on, average monthly working hours, time spent in the company, promotions in the last five years, and salary level. 

Data from prior evaluations shows the employees’ satisfaction in the workplace. The data could be used to identify patterns in work style and their interest in continuing to work for the company.  

The HR Department owns the data and uses it to predict employee turnover. Employee turnover refers to the total number of workers who leave a company over time. 

As the ML Developer assigned to the HR Department, you have been asked to create ML programs to: 

* Perform data quality checks by checking for missing values, if any. 
* Understand what factors contributed most to employee turnover at EDA. 
* Perform clustering of employees who left based on their satisfaction and evaluation. 
* Handle the left Class Imbalance using the SMOTE technique. 
* Perform k-fold cross-validation model training and evaluate performance.
* Identify the best model and justify the evaluation metrics used.  
* Suggest various retention strategies for targeted employees. 

## Data will be modified from:  

satisfaction_level: Satisfaction level at the job of an employee 

last_evaluation: Rating between 0 and 1, received by an employee at his last evaluation 

number_project: The number of projects an employee is involved in 

average_montly_hours: Average number of hours in a month spent by an employee at the office 

time_spend_company: Number of years spent in the company 

Work_accident: 0 - no accident during employee stay, 1 - accident during employee stay 

left: 0 indicates an employee stays with the company and 
      1 indicates an employee left the company 

promotion_last_5years: Number of promotions in his stay 

Department: Department to which an employee belongs to 

salary: Salary in USD 

## Perform the following steps: 
* Perform data quality checks by checking for missing values, if any. 
* Understand what factors contributed most to employee turnover at EDA. 

  * Draw a heatmap of the correlation matrix between all numerical features or columns in the data. 

  * Draw the distribution plot of: 

  * Employee Satisfaction (use column satisfaction_level) 

  * Employee Evaluation (use column last_evaluation) 

  * Employee Average Monthly Hours (use column average_montly_hours)

  * Draw the bar plot of the employee project count of both employees who left and stayed in the organization (use column number_project and hue column left), and give your inferences from the plot.

* Perform clustering of employees who left based on their satisfaction and evaluation. 
  * Choose columns satisfaction_level, last_evaluation, and left. 
  * Do K-means clustering of employees who left the company into 3 clusters? 
  * Based on the satisfaction and evaluation factors, give your thoughts on the employee clusters. 

* Handle the left Class Imbalance using the SMOTE technique. 

  * Pre-process the data by converting categorical columns to numerical columns by: 

  * Separating categorical variables and numeric variables 

  * Applying get_dummies() to the categorical variables 

  * Combining categorical variables and numeric variables 

  * Do the stratified split of the dataset to train and test in the ratio 80:20 with random_state=123. 

  * Upsample the train dataset using the SMOTE technique from the imblearn module.

* Perform 5-fold cross-validation model training and evaluate performance. 

  * Train a logistic regression model, apply a 5-fold CV, and plot the classification report. 

  * Train a Random Forest Classifier model, apply the 5-fold CV, and plot the classification report. 

  * Train a Gradient Boosting Classifier model, apply the 5-fold CV, and plot the classification report. 

* Identify the best model and justify the evaluation metrics used. 

  * Find the ROC/AUC for each model and plot the ROC curve. 

  * Find the confusion matrix for each of the models. 

  * Explain which metric needs to be used from the confusion matrix: Recall or Precision? 

* Suggest various retention strategies for targeted employees. 

  * Using the best model, predict the probability of employee turnover in the test data. 

  * Based on the probability score range below, categorize the employees into four zones and suggest your thoughts on the retention strategies for each zone. 

  * Safe Zone (Green) (Score < 20%) 

  * Low-Risk Zone (Yellow) (20% < Score < 60%) 

  * Medium-Risk Zone (Orange) (60% < Score < 90%) 

  * High-Risk Zone (Red) (Score > 90%).

# Output: 
![OUTPUT](1.png)
![OUTPUT](2.png)
![OUTPUT](3.png)
![OUTPUT](4.png)
![OUTPUT](5.png)
![OUTPUT](6.png)
