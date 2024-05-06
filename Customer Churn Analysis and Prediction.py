#!/usr/bin/env python
# coding: utf-8

# # CUSTOMER CHURN ANALYSIS AND PREDICTION ON
# # TELECOMMUNICATION CASE STUDY
# 

# # DataSet Info: Sample Data Set containing Telco customer data and showing customers left within the last month

# In[1]:


#import the required libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
#to enable the inline plotting, where the plots/graphs 
#will be displayed just below the cell where your plotting commands are written.


# In[4]:


#Load the data file
telco_base_data=pd.read_csv(r"C:\Users\Divya sree\Downloads\telco-churn.csv")


# In[5]:


#Load the top 5 records of data
telco_base_data.head()


# In[6]:


telco_base_data.tail()


# #check the various attributes of data like shape(rows,columns), Columns and datatype

# In[7]:


telco_base_data.shape


# In[8]:


telco_base_data.columns.values


# In[9]:


telco_base_data.dtypes


# In[10]:


#check the descriptive statistics of numeric values
telco_base_data.describe()


# Senior citizen is actually a categorical data hence 25%-50%-75% distribution is not proper
# 
# 
# 75% customers have tenure less than 55 months
# 
# 
# Average Monthly charges are USD 64.76 whereas 25% customers pay more than USD 89.85 per month

# In[11]:


telco_base_data["Churn"].value_counts().plot(kind="barh",figsize=(8,6))
plt.xlabel("Count",labelpad=14)
plt.ylabel("Target Variable",labelpad=14)
plt.title("Count of TARGET Variable per category",y=1.02)


# In[12]:


telco_base_data["Churn"].value_counts()


# In[13]:


100*telco_base_data["Churn"].value_counts()/len(telco_base_data["Churn"])


# Data is highly imbalanced, ratio= 73:27
# 
# So we analyse the data with other features while taking the target values separately to get some insights

# In[14]:


#Concise summary of the dataframe, as we have too many columns, we are using the verbose=True made
telco_base_data.info(verbose=True)


# In[15]:


telco_base_data.isna().sum()


# #There are no missing values in the dataset

# # Data Cleaning

# 1.Create a copy of base data for manipulation and processing

# In[16]:


telco_data=telco_base_data.copy() #creates a copy of telco_base_data in telco_data


# 2.Total Charges should be numeric amount, so convert that column to numeric datatype

# In[17]:


telco_data.TotalCharges=pd.to_numeric(telco_data.TotalCharges,errors='coerce')


# In[18]:


telco_data.info()


# In[19]:


telco_data.isnull().sum()


# 3.As there are 11 null values in Total charges coulmns, we can check these results

# In[20]:


telco_data.loc[telco_data["TotalCharges"].isnull()==True]


# 4.Missing values treatment
# 
# Since the % of these records compared to total dataset is very low i,e 0.15%, it is safe to remove them for further processing

# In[21]:


#Removing missing values
telco_data.dropna(how='any',inplace=True)


# 5.Divide customers into bins based on their tenure ex: for tenure<12 months: assigns a tenure group if 1-12, for tenure between 0 to 2 yrs, tenure group of 13-24; and so on

# In[22]:


#get maximum tenure 
print(telco_data['tenure'].max())


# In[23]:


#group the tenure in bins of 12 months
labels=["{0}-{1}".format(i, i+11) for i in range(1,72,12)]
telco_data["tenure_group"]=pd.cut(telco_data.tenure,range(1,80,12),right=False, labels=labels) #pd.cut needs to segment and sort data values into bins


# In[24]:


telco_data["tenure_group"].value_counts()


# In[25]:


#drop column customerID and tenure
telco_data.drop(columns=["customerID","tenure"],axis=1,inplace=True)
telco_data.head()


# # Data Exploration

# 1.Plot distribution of individual predictors by churn

# In[26]:


telco_data["PaymentMethod"].unique()


# # Univariate analysis

# In[27]:


for i, predictor in enumerate(telco_data.drop(columns=["Churn","TotalCharges","MonthlyCharges"])):
    plt.figure(i)
    sns.countplot(data=telco_data,x=predictor,hue="Churn")


# 2. Convert the target variable "Churn" in a binary numberic variable i,e Yes=1, No=0

# In[28]:


telco_data["Churn"]=np.where(telco_data.Churn=="Yes",1,0)


# In[29]:


telco_data.head()


# In[30]:


telco_data["PaymentMethod"].unique()
telco_data.columns


# 3.Convert all categorical variables into continuous variables using label encoding

# In[31]:


from sklearn import preprocessing
label_encoder=preprocessing.LabelEncoder()
en=['gender', 'Partner', 'Dependents', 'PhoneService',
       'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
       'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
       'Contract', 'PaperlessBilling', 'PaymentMethod','tenure_group']
for i in en:
    telco_data[i]=label_encoder.fit_transform(telco_data[i])
telco_data.head()


# 4.Relationship between monthly charges and total charges

# In[32]:


sns.lmplot(data=telco_data,x="MonthlyCharges",y="TotalCharges",fit_reg=False)
#lmplot(lineplot) is used to visualize a linear fit


# Total Charges increases as monthly charges increases- as expected

# 5. Churn by monthly charges and total charges

# In[33]:


Mth=sns.kdeplot(telco_data.MonthlyCharges[(telco_data["Churn"]==0)],color="Red",shade="True")
Mth=sns.kdeplot(telco_data.MonthlyCharges[(telco_data["Churn"]==1)],color="Blue",shade="True")
Mth.legend(["No Churn","Churn"],loc="upper right")
Mth.set_ylabel("Density")
Mth.set_xlabel("Monthly Charges")
Mth.set_title("Monthly Charges by Churn")
# Kernal density estimate plot is used to visualize the distribution of observations in a dataset


# Insight: When monthly charges are high, Churn is high

# In[34]:


Tot=sns.kdeplot(telco_data.TotalCharges[(telco_data["Churn"]==0)],color="Red",shade="True")
Tot=sns.kdeplot(telco_data.TotalCharges[(telco_data["Churn"]==1)],ax=Tot,color="Blue",shade="True")
Tot.legend(["No Churn","Churn"],loc="upper right")
Tot.set_ylabel("Density")
Tot.set_xlabel("Total Charges")
Tot.set_title("Total Charges by Churn")


# Insight:High churn at lower total charges
# If we combine the insights of 3 parameters i,e: Tenure,Monthly Charges& Total Charges,then the insights are:
# Higher Monthly Charge at lower tenure results into lower Total Charges. Hence all the 3 factors viz Higher monthly charge, lower tenure, lower total charge are linked to high churn

# 6.Build a correlation of all predictors with Churn

# In[35]:


plt.figure(figsize=(20,8))
telco_data.corr()["Churn"].sort_values(ascending=False).plot(kind="bar")


# Derived Insights:
# High Churn is seen in the case of Month to month contracts, no online security, no tech support, First year of subscription and fibre optics internet
# 
# 
# Low churn is seen in the case of Long term contracts, subscriptions without internet service  and the customers engaged for 5+ years
# 
# Factors like Gender, Availability of Phone Service and multiple lines have almost no impact on churn
# 

# In[36]:


telco_data.corr()


# In[37]:


#This is also evident from the heatmap below:
plt.figure(figsize=(12,12))
sns.heatmap(telco_data.corr(),cmap="Paired")


# CONCLUSION:
# 
# 
# There are some of the quick insights:
# 
# 
# 1.Electronic check medium are the highest churners
# 
# 
# 2.Contract-type-Monthly customers are more likely to churn because of no contract terms,as they are free to go customers
# 
# 
# 3.No online security, no tech support category are high churners
# 
# 
# 4.Non senior Citizens are high churners

# In[38]:


telco_data.to_csv("tel_churn.csv")


# # Model Building

# In[39]:


import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from imblearn.combine import SMOTEENN


# In[40]:


df=pd.read_csv('tel_churn.csv')


# In[41]:


df.head(5)


# In[42]:


df=df.drop('Unnamed: 0',axis=1)


# In[43]:


df.head(5)


# In[44]:


x=df[['SeniorCitizen','Partner','Dependents','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','Contract','PaperlessBilling','PaymentMethod','MonthlyCharges','TotalCharges','tenure_group']]


# In[45]:


x


# In[46]:


y=df["Churn"]
y


# In[47]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# # Decision Tree Classifier

# In[48]:


model_dt=DecisionTreeClassifier(criterion='gini',random_state=100,max_depth=6,min_samples_leaf=8)


# In[49]:


model_dt.fit(x_train,y_train)


# In[50]:


y_pred=model_dt.predict(x_test)


# In[51]:


y_pred


# In[52]:


print(classification_report(y_test, y_pred,labels=[0,1]))


# In[53]:


print(confusion_matrix(y_test,y_pred))


# In[54]:


sm=SMOTEENN()
x_resampled,y_resampled=sm.fit_sample(x,y)


# In[55]:


xr_train,xr_test,yr_train,yr_test=train_test_split(x_resampled,y_resampled,test_size=0.2,random_state=100)


# In[56]:


model_dt_smote=DecisionTreeClassifier(criterion='gini',random_state=100,max_depth=6,min_samples_leaf=8)


# In[57]:


model_dt_smote.fit(xr_train,yr_train)


# In[58]:


y_pred_smote=model_dt_smote.predict(xr_test)


# In[59]:


y_pred_smote


# In[60]:


print(classification_report(yr_test, y_pred_smote,labels=[0,1]))


# In[61]:


print(confusion_matrix(yr_test,y_pred_smote))


# # Random Forest Classifier 

# In[62]:


from sklearn.ensemble import RandomForestClassifier


# In[63]:


model_rf=RandomForestClassifier(criterion='gini',random_state=100,max_depth=6,min_samples_leaf=8)
model_rf.fit(x_train,y_train)
y_pred=model_rf.predict(x_test)


# In[64]:


y_pred


# In[65]:


print(classification_report(y_test, y_pred,labels=[0,1]))


# In[66]:


sm=SMOTEENN()
x_resampled,y_resampled=sm.fit_sample(x,y)


# In[67]:


xr_train,xr_test,yr_train,yr_test=train_test_split(x_resampled,y_resampled,test_size=0.2,random_state=100)


# In[68]:


model_rf_smote=RandomForestClassifier(criterion='gini',random_state=100,max_depth=6,min_samples_leaf=8)


# In[69]:


model_rf_smote.fit(xr_train,yr_train)


# In[70]:


y_pred_smote_rf=model_rf_smote.predict(xr_test)


# In[71]:


print(classification_report(yr_test, y_pred_smote_rf,labels=[0,1]))


# In[72]:


print(confusion_matrix(yr_test,y_pred_smote_rf))


# In[73]:


x.columns


# In[ ]:


print("For Gender MALE-1, FEMALE-0")
print("For senior citizen, partner, dependents, phoneservice, paperless billing YES-1 NO-0")
print("For MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies  NO-0 NO INTERNET SERVICE-1 YES-2")
print("For PaymentMethod Bank transfer-0, Credit card-1, Electronic check-2, Mailed check-3")
print("For tenure_group 1-12->0  13-24->1  25-36->2 37-48->3 49-60->4 61-72->5")
l=[]
for i in range(0,13):
    if(x.columns[i]=="TotalCharges"):
        print("Enter "+"TotalCharges")
        z=float(input())
    elif(x.columns[i]=="MonthlyCharges"):
        print("Enter "+"MonthlyCharges")
        z=float(input())
    else:
        print("Enter "+x.columns[i])
        z=int(input())
    l.append(z)
    if(x.columns[i]=="TotalCharges"):
        if(z<18 or z>9000):
            print("Invalid input")
            break
    elif(x.columns[i]=="MonthlyCharges"):
        if(z<18 or z>120):
            print("Invalid input")
            break
    elif(x.columns[i]=="gender" or x.columns[i]=="SeniorCitizen" or x.columns[i]=="Partner" or x.columns[i]=="Dependents" or x.columns[i]=="PhoneService" or x.columns[i]=="PaperlessBilling"):
        if(z<0 or z>1):
            print("Invalid input")
            break
    elif(x.columns[i]=="PaymentMethod"):
         if(z<0 or z>3):
            print("Invalid input")
            break
    elif(x.columns[i]=="tenure_group"):
        if(z<0 or z>5):
            print("Invalid input")
            break
    else:
        if(z<0 or z>2):
            print("Invalid input")
            break
l=np.array(l).reshape(1,-1)
solution=model_rf_smote.predict(l)
if(solution==1):
    print("Customer is likely to churn!")
else:
    print("Customer is not likely to churn")
            


# In[ ]:





# In[ ]:




