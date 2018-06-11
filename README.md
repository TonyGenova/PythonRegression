### PythonRegression
Notes on regression in Python  
  
Some ideas from "Machine Learning Bootcamp" on Udemy https://www.udemy.com/python-for-data-science-and-machine-learning-bootcamp/  

need to add
 - k fold cross validation process
 - R^2 output
 - process to cycle through each X to test individually, getting R^2

``` python
#imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline #double check what this does

#read csv into Pandas DataFrame
#can utilize for any DataFrame name/file name
USAhousing  = pd.read_csv('USA_Housing.csv')  

#check how data looks
USAhousing.head()  

#info on data
USAhousing.describe()

#column names
USAhousing.columns
 
#Seaborn pairplot to view data
sns.pairplot(USAhousing)

#view distribution for a specific column:
sns.distplot(USAhousing['Price'])

#correlation
df.corr()
#correlation heatmap
sns.heatmap(USAhousing.corr(), annot=True)

#splitting X and Y, selecting only the DataFrame columns you want
X = USAhousing[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
               'Avg. Area Number of Bedrooms', 'Area Population']]
y = USAhousing['Price']

#training/testing split
fromfrom  sklearn.model_selectionsklearn.m  import train_test_split
#test_size is percentage to keep for the test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)

#creating and fitting model
#need to add K fold validation
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)

# print the intercept# print t 
print(lm.intercept_)

#list of ceofficients
coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
coeff_df

#predictions and scatterplot of predictions to actuals
predictions = lm.predict(X_test)
plt.scatter(y_test,predictions)

#histogram of residuals
sns.distplot((y_test-predictions),bins=50);

```

