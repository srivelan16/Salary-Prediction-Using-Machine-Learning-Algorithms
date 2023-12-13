#!/usr/bin/env python
# coding: utf-8

# # SALARY PREDICTION

# In[5]:


#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[6]:


#Read the data
data = pd.read_csv(r"C:/Users/tnsre/Downloads/Salary Data.csv")


# In[7]:


data.head()


# In[8]:


data.shape


# In[108]:


x = data.iloc[:, :-1].values    
y = data.iloc[:, -1].values 

x


# In[109]:


y


# In[124]:


data['Gender'] = data['Gender'].replace({'male': 0, 'female': 1})


# In[125]:


from sklearn.model_selection import train_test_split

# random_state => seed value used by random number generator
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)


# In[9]:


categorical = data.loc[:,['Gender','Age','Education Level', 'Job Title','Salary']]
categorical.head()


# In[10]:


categorical['Job Title'].value_counts()


# In[11]:


categorical['Gender'].value_counts()


# In[12]:


categorical['Age'].value_counts()


# In[13]:


categorical['Salary'].value_counts()


# In[14]:


categorical['Education Level'].value_counts()


# In[15]:


condition_Female_1 = (categorical['Gender'] == 'Female') & (categorical['Education Level'] == "Bachelor's")
len(categorical[condition_Female_1])


# In[16]:


categorical.loc[condition_Female_1, 'Job Title'].value_counts().head(10)


# In[17]:


condition_male_2 = (categorical['Gender'] == 'Male') & (categorical['Education Level'] == "Master's")
len(categorical[condition_male_2])


# In[18]:


categorical.loc[condition_male_2, 'Job Title'].value_counts().head(10)


# In[19]:


condition_female_3 = (categorical['Gender'] == 'Female') & (categorical['Education Level'] == "PhD")
len(categorical[condition_female_3])


# In[20]:


categorical.loc[condition_female_3, 'Job Title'].value_counts().head(5)


# In[21]:


Numerical = data.loc[:, ['Age', 'Years of Experience', 'Salary']]
Numerical.head(10)


# In[22]:


Numerical.plot(kind = 'scatter', x = 'Years of Experience', y = 'Salary')


# In[23]:


Numerical.plot(kind = 'scatter', x = 'Age', y = 'Salary')


# In[24]:


#checking null values
data.isnull()


# In[25]:


data.sum()


# In[26]:


data.isnull().sum()


# In[27]:


condition_age = data['Age'].isnull()
data[condition_age]


# In[28]:


condition_gender = data['Gender'].isnull()
data[condition_gender]


# In[29]:


data[data['Education Level'].isnull()]


# In[30]:


data[data['Job Title'].isnull()]


# In[31]:


data[data['Years of Experience'].isnull()]


# In[32]:


data[data['Salary'].isnull()]


# In[33]:


Fresh_data = data.copy()


# In[34]:


Fresh_data.dropna(axis = 0, inplace = True)


# In[35]:


Fresh_data.isnull().sum()


# In[36]:


gender = pd.get_dummies(Fresh_data['Gender'])
Fresh_data = pd.concat([Fresh_data, gender], axis = 1)
Fresh_data.drop(columns = ['Gender'], inplace = True)


# In[37]:


education = pd.get_dummies(Fresh_data['Education Level'])
Fresh_data = pd.concat([Fresh_data, education], axis = 1)
Fresh_data.drop(columns = ['Education Level'], inplace = True)


# In[38]:


from sklearn.preprocessing import LabelEncoder
Fresh_data['Job Title'] = LabelEncoder().fit(Fresh_data['Job Title']).transform(Fresh_data['Job Title'])


# In[39]:


Fresh_data.head(10)


# In[40]:


Fresh_data['Age'] = Fresh_data['Age'] / Fresh_data['Age'].max()
Fresh_data['Job Title'] = Fresh_data['Job Title'] / Fresh_data['Job Title'].max()
Fresh_data['Years of Experience'] = Fresh_data['Years of Experience'] / Fresh_data['Years of Experience'].max()
Fresh_data['Salary'] = Fresh_data['Salary'] / Fresh_data['Salary'].max()


# In[56]:


x=Fresh_data['Salary'] = Fresh_data['Salary'] / Fresh_data['Salary'].max()


# In[88]:


x = data.iloc[:, :-1].values


# In[70]:


y=Fresh_data['Years of Experience'] = Fresh_data['Years of Experience'] / Fresh_data['Years of Experience'].max()


# In[90]:


y = data.iloc[:, -1].values


# In[91]:


from sklearn.model_selection import train_test_split
x_train, x_, y_train, y_ = train_test_split(Fresh_data.drop(columns = ['Salary']), Fresh_data['Salary'], test_size = 0.60, random_state = 1)
x_cv, x_test, y_cv, y_test = train_test_split(x_, y_, test_size = 0.75, random_state = 1)


# In[92]:


print('x_train', x_train.shape, 'y_train', y_train.shape)
print('x_cv', x_cv.shape, 'y_cv', y_cv.shape)
print('x_test', x_test.shape, 'y_test', y_test.shape)


# # importing libraries and fitting model

# In[93]:


from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error


# In[94]:


models = []
scalers = []
train_mses = []
cv_mses = []
train_r2s = []
cv_r2s = []
train_maes = []
cv_maes = []

degree = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
for n in degree:
    poly = PolynomialFeatures(degree = n, include_bias = False)
    x_train_poly = poly.fit_transform(x_train)
    model = LinearRegression()
    model.fit(x_train_poly, y_train)
    models.append(model)
    
    train_prediction = model.predict(x_train_poly)
    train_mse = mean_squared_error(y_train, train_prediction)
    train_mses.append(train_mse)
    train_r2 = r2_score(y_train, train_prediction)
    train_r2s.append(train_r2)
    train_mae = mean_absolute_error(y_train, train_prediction)
    train_maes.append(train_mae)
    
    poly = PolynomialFeatures(degree = n, include_bias = False)
    x_cv_poly = poly.fit_transform(x_cv)
    cv_prediction = model.predict(x_cv_poly)
    cv_mse = mean_squared_error(y_cv, cv_prediction)
    cv_mses.append(cv_mse)
    cv_r2 = r2_score(y_cv, cv_prediction)
    cv_r2s.append(cv_r2)
    cv_mae = mean_absolute_error(y_cv, cv_prediction)
    cv_maes.append(cv_mae)


# In[95]:


#Evaluating model
cv_degree = np.argmin(cv_mses) + 1
print(f'Lowest CV_MSE at degree = {cv_degree}')


# In[96]:


cv_degree = np.argmin(cv_maes) + 1
print(f'Lowest CV_MAE at degree = {cv_degree}')


# In[97]:


print(f'w = {models[0].coef_}, b = {models[0].intercept_}')


# In[98]:


poly = PolynomialFeatures(degree = cv_degree, include_bias = False)
x_test_poly = poly.fit_transform(x_test)
test_prediction = models[0].predict(x_test_poly)
test_mse = mean_squared_error(y_test, test_prediction)
test_r2 = r2_score(y_test, test_prediction)
test_mae = mean_absolute_error(y_test, test_prediction)


# In[99]:


from sklearn.metrics import mean_squared_error
print(f'Train_MSE = {train_mses[0]}')
print(f'CV_MSE = {cv_mses[0]}')
print(f'Test_MSE = {test_mse}')


# In[100]:


from sklearn.metrics import r2_score
print(f'Train_R2 = {train_r2s[0]}')
print(f'CV_R2 = {cv_r2s[0]}')
print(f'Test_R2 = {test_r2}')


# In[101]:


from sklearn.metrics import mean_absolute_error
print(f'Train_MAE = {train_maes[0]}')
print(f'CV_MAE = {cv_maes[0]}')
print(f'Test_MAE = {test_mae}')


# In[ ]:




