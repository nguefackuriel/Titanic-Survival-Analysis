#!/usr/bin/env python
# coding: utf-8

# <h1>WORKSHOP IT CLUB AIMS CAMEROON 2021/2022</h1>
# 
# ## Presented by Brenda ANAGUE and Uriel NGUEFACK

# <h1>Titanic Disaster Survival Using Logistic Regression</h1>

# ## 1) Objectives of this Workshop:
# 
# - Understand the shape of the Data (Histograms, Bar plots,..)
# - Data Exploration
# - Data Cleaning
# - Data Modeling

# ### The final Code will be given to all the students. Feel free to come and see us for any problems

# **Explaining Dataset**
# 
# survival : Survival 0 = No, 1 = Yes <br>
# pclass : Ticket class 1 = 1st, 2 = 2nd, 3 = 3rd <br>
# sex : Sex <br>
# Age : Age in years <br>
# sibsp : Number of siblings / spouses aboard the Titanic 
# <br>parch # of parents / children aboard the Titanic <br>
# ticket : Ticket number fare Passenger fare cabin Cabin number <br>
# embarked : Port of Embarkation C = Cherbourg, Q = Queenstown, S = Southampton <br>

# In[ ]:


#import libraries


# In[ ]:


from google.colab import drive
drive.mount('/content/gdrive')


# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# **Load the Data**

# In[ ]:


#load data


# In[ ]:


#titanic_data=pd.read_csv('titanic_train.csv')
titanic_data=pd.read_csv('/content/gdrive/MyDrive/IT CLUB AIMS CAMEROON WORKSHOP/Logistic-Regression-main/titanic_train.csv')


# In[ ]:


len(titanic_data)


# **View the data using head function which returns top  rows**

# In[ ]:


titanic_data.head()


# In[ ]:


titanic_data.index


# In[ ]:


titanic_data.columns


# In[ ]:


len(titanic_data.columns)


# In[ ]:


# Quick look at our data types & null counts


# In[ ]:


titanic_data.info()


# In[ ]:


titanic_data.dtypes


# In[ ]:


# Better understand the numeric data


# In[ ]:


titanic_data.describe()


# <h1>Data Analysis

# **Import Seaborn for visually analysing the data**

# **Find out how many survived vs Died using countplot method of seaboarn**

# In[ ]:


#countplot of subrvived vs not  survived


# In[ ]:


sns.countplot(x='Survived',data=titanic_data)


# **Male vs Female Survival**

# In[ ]:


#Male vs Female Survived?


# In[ ]:


# Pivot table 

print(pd.pivot_table(titanic_data, index = 'Survived', columns = 'Sex', values = 'Ticket', aggfunc = 'count'))

print()


# In[ ]:


sns.countplot(x='Survived',data=titanic_data,hue='Sex')


# **Class 1 vs Class 2 vs Class 3 Survival**

# In[ ]:


# Pivot table 

print(pd.pivot_table(titanic_data, index = 'Survived', columns = 'Pclass', values = 'Ticket', aggfunc = 'count'))

print()


# In[ ]:


sns.countplot(x='Survived',data=titanic_data,hue='Pclass')


# **Embarqued C vs Embarked Q vs Embarked S Survival**

# In[ ]:


# Pivot table 

print(pd.pivot_table(titanic_data, index = 'Survived', columns = 'Embarked', values = 'Ticket', aggfunc = 'count'))

print()


# In[ ]:


sns.countplot(x='Survived',data=titanic_data,hue='Embarked')


# **See age group of passengeres travelled **<br>
# Note: We will use displot method to see the histogram. However some records does not have age hence the method will throw an error. In order to avoid that we will use dropna method to eliminate null values from graph

# ### Correlation between Data

# In[ ]:


print(titanic_data[['Age','SibSp','Parch','Fare']].corr())

sns.heatmap(titanic_data[['Age','SibSp','Parch','Fare']].corr())


# ### Compare survival rate accross Age, SibSp, Parch, and Fare

# In[ ]:


pd.pivot_table(titanic_data, index = 'Survived', values = ['Age','SibSp','Parch','Fare'])


# ### Distribution for all numeric variables

# In[ ]:


# Let's first extract numeric values

df_num = titanic_data[['Age', 'SibSp', 'Parch', 'Fare']]

for i in df_num.columns:
    plt.hist(df_num[i])
    plt.title(i)
    plt.show()


# ### Distribution for all categorical variables

# In[ ]:


# Let's first extarct categorical data

df_cat = titanic_data[['Survived', 'Pclass', 'Sex', 'Ticket', 'Cabin', 'Embarked']]

for i in df_cat.columns:
    sns.barplot(df_cat[i].value_counts().index, df_cat[i].value_counts()).set_title(i)
    plt.show()


# In[ ]:


#Check for null


# In[ ]:


titanic_data.isna()


# In[ ]:


#Check how many values are null


# In[ ]:


titanic_data.isna().sum()


# In[ ]:


#Visualize null values


# In[ ]:


sns.heatmap(titanic_data.isna())


# In[ ]:


#find the % of null values in age column


# In[ ]:


(titanic_data['Age'].isna().sum()/len(titanic_data['Age']))*100


# In[ ]:


#find the % of null values in cabin column


# In[ ]:


(titanic_data['Cabin'].isna().sum()/len(titanic_data['Cabin']))*100


# In[ ]:


#find the distribution for the age column


# In[ ]:


sns.displot(x='Age',data=titanic_data)


# <h1>Data Cleaning

# **Fill the missing values**<br> we will fill the missing values for age. In order to fill missing values we use fillna method.<br> For now we will fill the missing age by taking average of all age 

# In[ ]:


#fill age column


# In[ ]:


titanic_data['Age'].fillna(titanic_data['Age'].mean(),inplace=True)


# **We can verify that no more null data exist** <br> we will examine data by isnull mehtod which will return nothing

# In[ ]:


#verify null value


# In[ ]:


titanic_data['Age'].isna().sum()


# **Alternatively we will visualise the null value using heatmap**<br>
# we will use heatmap method by passing only records which are null. 

# In[ ]:


#visualize null values


# In[ ]:


sns.heatmap(titanic_data.isna())


# In[ ]:





# **We can see cabin column has a number of null values, as such we can not use it for prediction. Hence we will drop it**

# In[ ]:


#Drop cabin column


# In[ ]:


titanic_data.drop('Cabin',axis=1,inplace=True)


# In[ ]:


#see the contents of the data


# In[ ]:


titanic_data.head()


# **Preaparing Data for Model**<br>
# No we will require to convert all non-numerical columns to numeric. Please note this is required for feeding data into model. Lets see which columns are non numeric info describe method

# In[ ]:


#Check for the non-numeric column


# In[ ]:


titanic_data.info()


# In[ ]:


titanic_data.dtypes


# **We can see, Name, Sex, Ticket and Embarked are non-numerical.It seems Name,Embarked and Ticket number are not useful for Machine Learning Prediction hence we will eventually drop it. For Now we would convert Sex Column to dummies numerical values******

# In[ ]:


#convert sex column to numerical values


# In[ ]:


gender=pd.get_dummies(titanic_data['Sex'],drop_first=True)


# In[ ]:


titanic_data['Gender']=gender


# In[ ]:


titanic_data.head()


# In[ ]:


#drop the columns which are not required


# In[ ]:


titanic_data.drop(['Name','Sex','Ticket','Embarked'],axis=1,inplace=True)


# In[ ]:


titanic_data.head()


# In[ ]:


#Seperate Dependent and Independent variables


# In[ ]:


x=titanic_data[['PassengerId','Pclass','Age','SibSp','Parch','Fare','Gender']]
y=titanic_data['Survived']


# In[ ]:


y


# <h1>Data Modelling

# **Building Model using Logestic Regression**

# **Build the model**

# In[ ]:


#import train test split method


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


#train test split


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)


# In[ ]:


#import Logistic  Regression


# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


#Fit  Logistic Regression 


# In[ ]:


lr=LogisticRegression()


# In[ ]:


lr.fit(x_train,y_train)


# In[ ]:


#predict


# In[ ]:


predict=lr.predict(x_test)


# <h1>Testing

# **See how our model is performing**

# In[ ]:


#print confusion matrix 


# In[ ]:


from sklearn.metrics import confusion_matrix


# In[ ]:


pd.DataFrame(confusion_matrix(y_test,predict),columns=['Predicted No','Predicted Yes'],index=['Actual No','Actual Yes'])


# 

# In[ ]:


#import classification report


# In[ ]:


from sklearn.metrics import classification_report


# In[ ]:


print(classification_report(y_test,predict))


# **Precision is fine considering Model Selected and Available Data. Accuracy can be increased by further using more features (which we dropped earlier) and/or  by using other model**
# 
# Note: <br>
# Precision : Precision is the ratio of correctly predicted positive observations to the total predicted positive observations <br>
# Recall : Recall is the ratio of correctly predicted positive observations to the all observations in actual class
# F1 score - F1 Score is the weighted average of Precision and Recall.
# 
# 
