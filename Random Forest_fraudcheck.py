#Random Forest Classification
# Use Random Forest to prepare a model on fraud data 
# treating those who have taxable_income <= 30000 as "Risky" and others are "Good"

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb

# Importing the dataset
dataset = pd.read_csv('Fraud_check.csv')

# creating dummy columns for the categorical columns 
dataset.columns
dummies = pd.get_dummies(dataset[['Undergrad', 'Marital.Status','Urban']])
# Dropping the columns for which we have created dummies
dataset.drop(['Undergrad', 'Marital.Status','Urban'],inplace=True,axis = 1)

# adding the columns to the dataset data frame 
dataset = pd.concat([dataset,dummies],axis=1)

# categorising taxable_income <= 30000 as "Risky" as 1 and others are "Good" as 0
dataset["fraud_cat"] = 0
dataset.loc[dataset['Taxable.Income']<=30000,"fraud_cat"] = 1
dataset.fraud_cat.value_counts()

df= dataset.drop(['Taxable.Income'],axis=1,inplace= False)
df.columns

# Getting the barplot for the target columns vs features
sb.countplot(x="fraud_cat",data=dataset,palette="hls")


# Data Distribution - Boxplot of continuous variables wrt to each category of categorical columns

sb.boxplot(x="fraud_cat",y="City.Population",data=dataset,palette="hls")
sb.boxplot(x="fraud_cat",y='Work.Experience',data=dataset,palette="hls")

# To get the count of null values in the data 
df.isnull().sum() #no na values
df.shape 

# spillting into X as input and Y as output variables
X=df.iloc[:, 0:9]
y=df.iloc[:, [9]]

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X.iloc[:,0:2]=sc_X.fit_transform(X.iloc[:,0:2])



from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_jobs=3,oob_score=True,n_estimators=15,criterion="entropy")
# n_estimators -> Number of trees ( you can increase for better accuracy)
# n_jobs -> Parallelization of the computing and signifies the number of jobs 
# running parallel for both fit and predict
# oob_score = True means model has done out of box sampling to make predictions

#### Attributes that comes along with RandomForest function
rf.fit(X,y) # Fitting RandomForestClassifier model from sklearn.ensemble 
rf.estimators_ # 
rf.classes_ # class labels (output)
rf.n_classes_ # Number of levels in class labels 
rf.n_features_  # Number of input features in model 8 here.

rf.n_outputs_ # Number of outputs when fit performed

rf.oob_score_  # 0.7166
y_pred=rf.predict(X)
##############################

df['rf_pred'] = rf.predict(X)
cols = ['rf_pred', 'fraud_cat']
df[cols].head()
df["fraud_cat"]


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(df['fraud_cat'],df['rf_pred']) # Confusion matrix
cm
pd.crosstab(df['fraud_cat'],df['rf_pred'])

print("Accuracy",(474+116)/(497+116+2+8)*100)   #94.70

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = rf, X = X, y = y, cv = 10)
accuracies.mean()   #0.74
accuracies.std()

# Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
parameters = [{'n_estimators': [5, 7, 10, 25], 'criterion': ["entropy"], 'oob_score': ['True']},
              {'n_estimators': [5, 7, 10, 25], 'criterion': ["gini"], 'oob_score': ['True']}]
grid_search = GridSearchCV(estimator = rf,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(X, y)
best_accuracy = grid_search.best_score_ #0.75
best_parameters = grid_search.best_params_

# best parameters are {'criterion': 'gini', 'n_estimators': 10, 'oob_score': 'True'}
# Modelling with best parameters

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_jobs=3,oob_score=True,n_estimators=10,criterion="gini")
# n_estimators -> Number of trees ( you can increase for better accuracy)
# n_jobs -> Parallelization of the computing and signifies the number of jobs 
# running parallel for both fit and predict
# oob_score = True means model has done out of box sampling to make predictions

#### Attributes that comes along with RandomForest function
model.fit(X,y) # Fitting RandomForestClassifier model from sklearn.ensemble 
model.oob_score_  # 0.72

# prediction
y_pred=model.predict(X)

pd.crosstab(df.fraud_cat, y_pred)

print("Accuracy",(476+101)/(476+101+0+23)*100)  #96.17
