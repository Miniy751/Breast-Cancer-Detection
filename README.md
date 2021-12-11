# Bootcamp-Project-4-Group-3
## Breast Cancer Detection

A project by Mini Yadav, Elena Dragomir and Belinda Achuo

## Project Proposal 
* Objectives
* Data Preparation
* Data Pre-processing
* Building the Model
* Randomized Search to find the best parameters (Logistic regression)
* The Final Model (Logistic regression)
* Predicting an observation - Live 

## Our Dataset

* Data was extracted from Kaggle.com, were the team agreed to go with the following Data sets:

* https://www.kaggle.com/uciml/breast-cancer-wisconsin-data 


## Data Processing

* Using a jupyter notebook.
* We then imported the libraries and dataset
* We Used Pandas function  to import, read the dataset
 
## Exploring the dataset.
* To see the shape of the data
* To get dataset information to see the numerical and categorical columns.
* To drop columns with nan Values
* To get the dataset description.( to see the  statistical summary)
### Further Exploration
* Using one  hot encoding to convert categorical values (M, B - diagnosis column) to numerical values
* Using a count plot to show the total count for Benign and Malignant values respectively.

## Correlation Matrix and Heat Map
To take out the  independent variable (diagnosis) we then dropped from the dataset.
We correlated dataset_2(independent variable) with dataset(dependent variables) by plotting a bar chart.
We then defined the correlation matrix  to see how each variable correlates with another.
We then Analyse the correlated dataset by plotting a seaborn.heatmap.


## Splitting the dataset into Train and Test set
 
* We then used the iloc function to isolate the (Independent Variable)(x)
* Then used the iloc function to isolate the (dependent Variable)(y)
* We then use a train_test_split function for splitting the dataset into (x_test and train, y_test and train) from Sklearn Library.
to specify the matrix soft features and Target variable
* We then  Assigned the data to X and y variables to get the dataset shape for each  respectively.

## Applying Feature Scaling

* We then used the standard scaler (sc.fit_transform) , to scale the x_train and x_test features. 
* This is to have all the variables on the same scale.

## Building the Models.

 1. **Logistic Regression**:

   * From sklearn.linear_model import LogisticRegression
   * We then created the logistic regression class, specifying the Random State = 0.
   * We then trained our Model using the classifier_lr.fit method(specifying the(x_train, y_train).
   * We created a variable with all predicted values by using the .predict method (x_test) and stored in a y_pred
   * Next We Analysed the performance of our  logistic regression model.

   * By Using the following Parameters.
   
     - Accuracy  classification score:     To calculate   accuracy for our sampled model       
     - F1 Score : interpreted as the weighted average  ( balanced score- 0 being   worse and 1 is better)
     - Precision score: is the ratio tp/(tp+fp) true or false positive.best value is 1 and worse is 0
     - Recall_Score: Calculates the ratio of   tp/(tp+fn), best value is 1 and worse is 0
     - Confusion Matrix: to evaluate the accuracy of a classification
     
   * Cross Validation: To evaluate our scores by cross-validation; This was to evaluate the performance of our models and compute 10 different accuracy  on the basis of               x_train and y_train. Which increased the accuracy to 97.81 %.  And standard deviation for all the 10 accuracies. 1.98%

 2. **Random Forest Model**:
      
    * from sklearn.ensemble import RandomForestClassifier
    * Specified the Classifier as  classifier_rm    then random_state = 0
    * Applying the .fit method as classifier_rm.fit(x_train, y_train) to train the model.
    * We then Analysed the performance: y_pred = classifier_rm.predict(x_test)

    * By Applying the following Parameters.

      - Accuracy  classification score:     To calculate   accuracy for our sampled model       
      - F1 Score : interpreted as the weighted average  ( balanced score- 0 being   worse and 1 is better)
      - Precision score: is the ratio tp/(tp+fp) true or false positive.best value is 1 and worse is 0
      - Recall_Score: Calculates the ratio of   tp/(tp+fn), best value is 1 and worse is 0
      - Confusion Matrix: to evaluate the accuracy of a classification
     
      
     * Cross Validation of Random Forest: Accuracy is 96.05 %  And standard deviation for all the 10 accuracies. 3.07%.
    
 3. **AdaBoost Model** 



 4.  **Comparing classification reports**
      
      - Logistic Regression: Accuracy=97.81%; Standard Deviation =1.98%
      - Random Forest: Accuracy=96.05 %; Standard Deviation =3.07
      - AdaBoost : Accuracy=96.27 %; Standard Deviation =3.09 %
      
5.   **Comparing the Models we computed a ROC metrics then did a plot**

6.   **

     
    
  

