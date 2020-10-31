'''
Basic pipeline to compare logistic regression, KNN, Randomforest, and Gradient boost
'''

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import accuracy_score, classification_report, plot_confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

import matplotlib.pyplot as plt


def train_val_test(df, features, target):
    '''
    Splits the data into train, validate and test sets
    ---
    Input: <DataFrame>
           <list> the column names of the features you want to consider
           <string> the column name of the target you want
    ---
    Output: X_train, y_train, X_val, y_val, X_test, y_test
    '''
    
    #set features and target
    X = df[features]
    y = df[target].astype(int) 
    
    #split the data into train-val-test: 80-20-20
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size = .20, random_state = 42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size = .20, random_state = 42)
    
    return X_train, y_train, X_val, y_val, X_test, y_test






def logistic_report(X_train, y_train, X_val, y_val):
    '''
    Input: X_train, y_train, X_val, y_val
    ---
    Output: train and val scores, classification report, confusion matrix
    '''
    
    #logistic regression
    logreg = LogisticRegression(penalty = 'none')
    logreg.fit(X_train, y_train)
    print('Logistic Regression Train score:', logreg.score(X_train, y_train))
    print('Logistic Regression Validate score:', logreg.score(X_val, y_val))
    
    #print classification report
    y_true, y_pred = y_val, logreg.predict(X_val)
    print('Logistic Regression Classification Report')
    print(classification_report(y_true, y_pred))
    
    #print confusion matrix
    print('Logistic Regression Confusion Matrix')
    fig, ax = plt.subplots(figsize=(7, 7))
    plt.title('Logistic Regression Confusion Matrix')
    plot_confusion_matrix(logreg, X_val, y_val, ax=ax);


    
    
    
def knn_report(X_train, y_train, X_val, y_val):
    '''
    Input: X_train, y_train, X_val, y_val
    ---
    Output: train and val scores, classification report, confusion matrix
    '''
    
    #start by scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.values)
    X_val_scaled = scaler.fit_transform(X_val.values)
    
    #fit knn model
    knn = KNeighborsClassifier(n_neighbors = 7)
    knn.fit(X_train_scaled, y_train)
    
    #print scores
    print('knn Training Score:', knn.score(X_train_scaled, y_train))    
    print('knn Test Score:', knn.score(X_val_scaled, y_val))
    
    #print classification model
    y_true_knn, y_pred_knn = y_val, knn.predict(X_val_scaled)
    print('knn Classification Report')
    print(classification_report(y_true_knn, y_pred_knn))
    
    #print confusion matrix
    print('knn Confusion Matrix')
    fig, ax = plt.subplots(figsize=(7, 7))
    plt.title('KNN Confusion Matrix')
    plot_confusion_matrix(knn, X_val, y_val, ax=ax);


    
    

def rfc_report(X_train, y_train, X_val, y_val):
    
    #fit the model
    rfc = RandomForestClassifier(n_estimators=1000)
    rfc.fit(X_train , y_train)
    
    #print classification report
    print('RandomForestClassifier Classification Report')
    y_pred = rfc.predict(X_val)
    print(classification_report(y_val, y_pred))
    
    #print confusion matrix
    print('RandomForestClassifier Confusion Matrix')
    fig, ax = plt.subplots(figsize=(7, 7))
    plt.title('RandomForestClassifier Confusion Matrix')
    plot_confusion_matrix(rfc, X_val, y_val, ax=ax);
    
    #get feature importance
    print('RandomForestClassifier feature importance')
    importance = rfc.feature_importances_
    for i,v in enumerate(importance):
        print('Feature: %0d, Score: %.5f' % (i,v))
    
    # plot feature importance
    plt.figure(figsize = (10,5))
    plt.title('RandomForestClassifier Feature Importance')
    plt.bar([x for x in range(len(importance))], importance)
    plt.xticks([i for i in range(X_val.shape[1])])
    plt.show()

    
    
    
def gb_report(X_train, y_train, X_val, y_val):
    
    #fit model
    model_GB = GradientBoostingClassifier(n_estimators=1000)
    model_GB.fit(X_train, y_train)
    
    #print classification report
    print('GradientBoodtingClassifier Classification Report')
    y_pred = model_GB.predict(X_val)
    print(classification_report(y_val, y_val))
    
    #print confusion matrix
    print('GradientBoodtingClassifier Confusion Matrix')
    fig, ax = plt.subplots(figsize=(7, 7))
    plt.title('GradientBoodtingClassifier Confusion Matrix')
    plot_confusion_matrix(model_GB, X_val, y_val, ax=ax);

    #print feature importance
    print('GradientBoodtingClassifier feature importance')
    importance_GB = model_GB.feature_importances_
    
    for i,v in enumerate(importance_GB):
        print('Feature: %0d, Score: %.5f' % (i,v))
    # plot feature importance
    plt.figure(figsize = (10,5))
    plt.title('GradientBoodtingClassifier Feature Importance')
    plt.bar([x for x in range(len(importance_GB))], importance_GB)
    plt.xticks([i for i in range(X_val.shape[1])])
    plt.show()


    
    
    
def compare(X_train, y_train, X_val, y_val):
    '''
    prints metrics for Logistic Regression, KNN, RandomForestClassifier, and GradientBoostClassifier
    ---
    Input: X_train, y_train, X_val, y_val
    
    '''
    
    print(logistic_report(X_train, y_train, X_val, y_val))
    
    print(knn_report(X_train, y_train, X_val, y_val))
    
    print(rfc_report(X_train, y_train, X_val, y_val))
    
    print(gb_report(X_train, y_train, X_val, y_val))