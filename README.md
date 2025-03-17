NAME : Ramya P
REGISTER NO : 212223240137
EX. NO : 1
DATE : 10/03/2025
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
```
import pandas as pd                                                
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
df=pd.read_csv("Churn_Modelling.csv",index_col="RowNumber")         
df.head()
```

```
df.isnull().sum()                                                   
```
```
df.duplicated().sum()                                               
```
```
df=df.drop(['Surname', 'Geography','Gender'], axis=1)               
scaler=StandardScaler()                                             
df=pd.DataFrame(scaler.fit_transform(df))
df.head()
```
```
X,Y=df.iloc[:,:-1].values ,df.iloc[:,-1].values                     
print('Input:\n',X,'\nOutput:\n',Y) 
Xtrain,Xtest,Ytrain,Ytest = train_test_split(X, Y, test_size=0.2)   
print("Xtrain:\n" ,Xtrain, "\nXtest:\n", Xtest)                     
print("\nYtrain:\n" ,Ytrain, "\nYtest:\n", Ytest)                   
```
## OUTPUT:
## DATASET:
![image](https://github.com/user-attachments/assets/a7ecbfc3-2363-4fc3-be76-1d142540ec41)
## NULL VALUES:
![image](https://github.com/user-attachments/assets/9e7c2a04-a6b5-422d-8867-9f1fc03daddd)
## NORMALIZED DATA:
![image](https://github.com/user-attachments/assets/be85853e-db10-474d-a959-5d254027b78e)
## DATA SPLITTING:
![Screenshot 2025-03-16 162745](https://github.com/user-attachments/assets/222eb899-c6e4-4702-98d8-74ac006638d1)
![Screenshot 2025-03-16 163117](https://github.com/user-attachments/assets/310d4097-36d2-4e8f-a124-e412ecb08633)
## TRAIN AND TEST DATA:
![Screenshot 2025-03-16 163245](https://github.com/user-attachments/assets/4f4e09e8-1175-40fc-90dd-2a35234616b6)
![Screenshot 2025-03-17 044509](https://github.com/user-attachments/assets/3496eea3-1534-40e0-b019-6211e09828d9)
![Screenshot 2025-03-17 080044](https://github.com/user-attachments/assets/907b3fcf-0cf0-4350-8d83-df0f03cd52c0)
![Screenshot 2025-03-17 080138](https://github.com/user-attachments/assets/7923c70e-cfd8-4bba-8180-13ce29b140b2)
## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


