# MLtower
upload the code of machine learning

1. import all the lib file


```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn import metrics
```

2. check the data and read the first five lines


```python
# check the data
filepath="C:\\Users\\towermalta\\MLcode\\tcd ml 2019-20 income prediction training (with labels).csv"
data = pd.read_csv(filepath)
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Instance</th>
      <th>Year of Record</th>
      <th>Gender</th>
      <th>Age</th>
      <th>Country</th>
      <th>Size of City</th>
      <th>Profession</th>
      <th>University Degree</th>
      <th>Wears Glasses</th>
      <th>Hair Color</th>
      <th>Body Height [cm]</th>
      <th>Income in EUR</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1</td>
      <td>1997.0</td>
      <td>0</td>
      <td>41.0</td>
      <td>Belarus</td>
      <td>1239930</td>
      <td>steel workers</td>
      <td>Bachelor</td>
      <td>0</td>
      <td>Blond</td>
      <td>193</td>
      <td>61031.94416</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2</td>
      <td>1996.0</td>
      <td>other</td>
      <td>41.0</td>
      <td>Singapore</td>
      <td>1603504</td>
      <td>safe event coordinator</td>
      <td>Master</td>
      <td>0</td>
      <td>Black</td>
      <td>186</td>
      <td>91001.32764</td>
    </tr>
    <tr>
      <td>2</td>
      <td>3</td>
      <td>2018.0</td>
      <td>other</td>
      <td>28.0</td>
      <td>Norway</td>
      <td>1298017</td>
      <td>receivables/payables analyst</td>
      <td>PhD</td>
      <td>1</td>
      <td>Brown</td>
      <td>170</td>
      <td>157982.17670</td>
    </tr>
    <tr>
      <td>3</td>
      <td>4</td>
      <td>2006.0</td>
      <td>other</td>
      <td>33.0</td>
      <td>Cuba</td>
      <td>751903</td>
      <td>fleet assistant</td>
      <td>No</td>
      <td>1</td>
      <td>Black</td>
      <td>171</td>
      <td>45993.75793</td>
    </tr>
    <tr>
      <td>4</td>
      <td>5</td>
      <td>2010.0</td>
      <td>female</td>
      <td>46.0</td>
      <td>United Arab Emirates</td>
      <td>95389</td>
      <td>lead trainer</td>
      <td>0</td>
      <td>0</td>
      <td>Blond</td>
      <td>188</td>
      <td>38022.16217</td>
    </tr>
  </tbody>
</table>
</div>



3. get the feature of the dataset


```python
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 111993 entries, 0 to 111992
    Data columns (total 12 columns):
    Instance             111993 non-null int64
    Year of Record       111552 non-null float64
    Gender               104561 non-null object
    Age                  111499 non-null float64
    Country              111993 non-null object
    Size of City         111993 non-null int64
    Profession           111671 non-null object
    University Degree    104623 non-null object
    Wears Glasses        111993 non-null int64
    Hair Color           104751 non-null object
    Body Height [cm]     111993 non-null int64
    Income in EUR        111993 non-null float64
    dtypes: float64(3), int64(4), object(5)
    memory usage: 10.3+ MB
    

4. check the situation of N/A value


```python
data.isnull().sum()
```




    Instance                0
    Year of Record        441
    Gender               7432
    Age                   494
    Country                 0
    Size of City            0
    Profession            322
    University Degree    7370
    Wears Glasses           0
    Hair Color           7242
    Body Height [cm]        0
    Income in EUR           0
    dtype: int64



5. check the information of the dataset: count,mean,std,min,max.......


```python
data.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Instance</th>
      <th>Year of Record</th>
      <th>Age</th>
      <th>Size of City</th>
      <th>Wears Glasses</th>
      <th>Body Height [cm]</th>
      <th>Income in EUR</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>count</td>
      <td>111993.000000</td>
      <td>111552.000000</td>
      <td>111499.000000</td>
      <td>1.119930e+05</td>
      <td>111993.000000</td>
      <td>111993.000000</td>
      <td>1.119930e+05</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>55997.000000</td>
      <td>1999.421274</td>
      <td>37.345304</td>
      <td>8.388538e+05</td>
      <td>0.500531</td>
      <td>175.220192</td>
      <td>1.092138e+05</td>
    </tr>
    <tr>
      <td>std</td>
      <td>32329.738686</td>
      <td>11.576382</td>
      <td>16.036694</td>
      <td>2.196879e+06</td>
      <td>0.500002</td>
      <td>19.913889</td>
      <td>1.498024e+05</td>
    </tr>
    <tr>
      <td>min</td>
      <td>1.000000</td>
      <td>1980.000000</td>
      <td>14.000000</td>
      <td>7.700000e+01</td>
      <td>0.000000</td>
      <td>94.000000</td>
      <td>-5.696906e+03</td>
    </tr>
    <tr>
      <td>25%</td>
      <td>27999.000000</td>
      <td>1989.000000</td>
      <td>24.000000</td>
      <td>7.273400e+04</td>
      <td>0.000000</td>
      <td>160.000000</td>
      <td>3.077169e+04</td>
    </tr>
    <tr>
      <td>50%</td>
      <td>55997.000000</td>
      <td>1999.000000</td>
      <td>35.000000</td>
      <td>5.060920e+05</td>
      <td>1.000000</td>
      <td>174.000000</td>
      <td>5.733917e+04</td>
    </tr>
    <tr>
      <td>75%</td>
      <td>83995.000000</td>
      <td>2009.000000</td>
      <td>48.000000</td>
      <td>1.184501e+06</td>
      <td>1.000000</td>
      <td>190.000000</td>
      <td>1.260936e+05</td>
    </tr>
    <tr>
      <td>max</td>
      <td>111993.000000</td>
      <td>2019.000000</td>
      <td>115.000000</td>
      <td>4.999251e+07</td>
      <td>1.000000</td>
      <td>265.000000</td>
      <td>5.285252e+06</td>
    </tr>
  </tbody>
</table>
</div>



6. show the trend of each feature


```python
data.hist(bins=50,figsize=(15,10))
plt.show
```




    <function matplotlib.pyplot.show(*args, **kw)>




![png](output_11_1.png)


7. show the trend of income specially


```python
data.plot('Year of Record','Income in EUR',kind = 'scatter')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1852efcfa08>




![png](output_13_1.png)


8. Pearson correlation coefficient is used according to the correlation between features and income.
   at this step, I choose to drop the feature of "wear glasses"


```python
#按各个特征和income的相关性，使用皮尔逊相关系数Pearson
corr_matrix = data.corr()
print(corr_matrix['Income in EUR'].sort_values(ascending=False))
```

    Income in EUR       1.000000
    Age                 0.186160
    Year of Record      0.165116
    Body Height [cm]    0.072889
    Size of City        0.014993
    Wears Glasses       0.005718
    Instance            0.002897
    Name: Income in EUR, dtype: float64
    


```python

```

1. read the train dataset into datatrain[]


```python
filepath="C:\\Users\\towermalta\\MLcode\\tcd ml 2019-20 income prediction training (with labels).csv"
datatrain = pd.read_csv(filepath)
datatrain.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Instance</th>
      <th>Year of Record</th>
      <th>Gender</th>
      <th>Age</th>
      <th>Country</th>
      <th>Size of City</th>
      <th>Profession</th>
      <th>University Degree</th>
      <th>Wears Glasses</th>
      <th>Hair Color</th>
      <th>Body Height [cm]</th>
      <th>Income in EUR</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1</td>
      <td>1997.0</td>
      <td>0</td>
      <td>41.0</td>
      <td>Belarus</td>
      <td>1239930</td>
      <td>steel workers</td>
      <td>Bachelor</td>
      <td>0</td>
      <td>Blond</td>
      <td>193</td>
      <td>61031.94416</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2</td>
      <td>1996.0</td>
      <td>other</td>
      <td>41.0</td>
      <td>Singapore</td>
      <td>1603504</td>
      <td>safe event coordinator</td>
      <td>Master</td>
      <td>0</td>
      <td>Black</td>
      <td>186</td>
      <td>91001.32764</td>
    </tr>
    <tr>
      <td>2</td>
      <td>3</td>
      <td>2018.0</td>
      <td>other</td>
      <td>28.0</td>
      <td>Norway</td>
      <td>1298017</td>
      <td>receivables/payables analyst</td>
      <td>PhD</td>
      <td>1</td>
      <td>Brown</td>
      <td>170</td>
      <td>157982.17670</td>
    </tr>
    <tr>
      <td>3</td>
      <td>4</td>
      <td>2006.0</td>
      <td>other</td>
      <td>33.0</td>
      <td>Cuba</td>
      <td>751903</td>
      <td>fleet assistant</td>
      <td>No</td>
      <td>1</td>
      <td>Black</td>
      <td>171</td>
      <td>45993.75793</td>
    </tr>
    <tr>
      <td>4</td>
      <td>5</td>
      <td>2010.0</td>
      <td>female</td>
      <td>46.0</td>
      <td>United Arab Emirates</td>
      <td>95389</td>
      <td>lead trainer</td>
      <td>0</td>
      <td>0</td>
      <td>Blond</td>
      <td>188</td>
      <td>38022.16217</td>
    </tr>
  </tbody>
</table>
</div>



2. read the test dataset into the datatest[]


```python
filepath="C:\\Users\\towermalta\\MLcode\\tcd ml 2019-20 income prediction test (without labels).csv"
# read_csv里面的参数是csv在你电脑上的路径
datatest = pd.read_csv(filepath)
#读取前五行数据，如果是最后五行，用data.tail()
datatest.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Instance</th>
      <th>Year of Record</th>
      <th>Gender</th>
      <th>Age</th>
      <th>Country</th>
      <th>Size of City</th>
      <th>Profession</th>
      <th>University Degree</th>
      <th>Wears Glasses</th>
      <th>Hair Color</th>
      <th>Body Height [cm]</th>
      <th>Income</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>111994</td>
      <td>1992.0</td>
      <td>other</td>
      <td>21.0</td>
      <td>Honduras</td>
      <td>391652</td>
      <td>senior project analyst</td>
      <td>Master</td>
      <td>1</td>
      <td>Brown</td>
      <td>153</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>1</td>
      <td>111995</td>
      <td>1986.0</td>
      <td>other</td>
      <td>34.0</td>
      <td>Kyrgyzstan</td>
      <td>33653</td>
      <td>greeter</td>
      <td>Bachelor</td>
      <td>0</td>
      <td>Black</td>
      <td>163</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>2</td>
      <td>111996</td>
      <td>1994.0</td>
      <td>unknown</td>
      <td>53.0</td>
      <td>Portugal</td>
      <td>34765</td>
      <td>liaison</td>
      <td>Bachelor</td>
      <td>1</td>
      <td>Blond</td>
      <td>153</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>3</td>
      <td>111997</td>
      <td>1984.0</td>
      <td>0</td>
      <td>29.0</td>
      <td>Uruguay</td>
      <td>1494132</td>
      <td>occupational therapist</td>
      <td>No</td>
      <td>0</td>
      <td>Black</td>
      <td>154</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>4</td>
      <td>111998</td>
      <td>2007.0</td>
      <td>other</td>
      <td>17.0</td>
      <td>Serbia</td>
      <td>120661</td>
      <td>portfolio manager</td>
      <td>No</td>
      <td>0</td>
      <td>Red</td>
      <td>191</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



3. according to the analysis before, drop the faeture of "wears glasses, hair color", process the feature in both datatrain and datatest


```python
# 选取数据集中有用的特征
datatrain = datatrain.drop(labels=[ 'Wears Glasses', 'Hair Color'], axis=1)
datatrain.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Instance</th>
      <th>Year of Record</th>
      <th>Gender</th>
      <th>Age</th>
      <th>Country</th>
      <th>Size of City</th>
      <th>Profession</th>
      <th>University Degree</th>
      <th>Body Height [cm]</th>
      <th>Income in EUR</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1</td>
      <td>1997.0</td>
      <td>0</td>
      <td>41.0</td>
      <td>Belarus</td>
      <td>1239930</td>
      <td>steel workers</td>
      <td>Bachelor</td>
      <td>193</td>
      <td>61031.94416</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2</td>
      <td>1996.0</td>
      <td>other</td>
      <td>41.0</td>
      <td>Singapore</td>
      <td>1603504</td>
      <td>safe event coordinator</td>
      <td>Master</td>
      <td>186</td>
      <td>91001.32764</td>
    </tr>
    <tr>
      <td>2</td>
      <td>3</td>
      <td>2018.0</td>
      <td>other</td>
      <td>28.0</td>
      <td>Norway</td>
      <td>1298017</td>
      <td>receivables/payables analyst</td>
      <td>PhD</td>
      <td>170</td>
      <td>157982.17670</td>
    </tr>
    <tr>
      <td>3</td>
      <td>4</td>
      <td>2006.0</td>
      <td>other</td>
      <td>33.0</td>
      <td>Cuba</td>
      <td>751903</td>
      <td>fleet assistant</td>
      <td>No</td>
      <td>171</td>
      <td>45993.75793</td>
    </tr>
    <tr>
      <td>4</td>
      <td>5</td>
      <td>2010.0</td>
      <td>female</td>
      <td>46.0</td>
      <td>United Arab Emirates</td>
      <td>95389</td>
      <td>lead trainer</td>
      <td>0</td>
      <td>188</td>
      <td>38022.16217</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 选取数据集中有用的特征
datatest = datatest.drop(labels=[ 'Wears Glasses', 'Hair Color'], axis=1)
datatest.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Instance</th>
      <th>Year of Record</th>
      <th>Gender</th>
      <th>Age</th>
      <th>Country</th>
      <th>Size of City</th>
      <th>Profession</th>
      <th>University Degree</th>
      <th>Body Height [cm]</th>
      <th>Income</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>111994</td>
      <td>1992.0</td>
      <td>other</td>
      <td>21.0</td>
      <td>Honduras</td>
      <td>391652</td>
      <td>senior project analyst</td>
      <td>Master</td>
      <td>153</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>1</td>
      <td>111995</td>
      <td>1986.0</td>
      <td>other</td>
      <td>34.0</td>
      <td>Kyrgyzstan</td>
      <td>33653</td>
      <td>greeter</td>
      <td>Bachelor</td>
      <td>163</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>2</td>
      <td>111996</td>
      <td>1994.0</td>
      <td>unknown</td>
      <td>53.0</td>
      <td>Portugal</td>
      <td>34765</td>
      <td>liaison</td>
      <td>Bachelor</td>
      <td>153</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>3</td>
      <td>111997</td>
      <td>1984.0</td>
      <td>0</td>
      <td>29.0</td>
      <td>Uruguay</td>
      <td>1494132</td>
      <td>occupational therapist</td>
      <td>No</td>
      <td>154</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>4</td>
      <td>111998</td>
      <td>2007.0</td>
      <td>other</td>
      <td>17.0</td>
      <td>Serbia</td>
      <td>120661</td>
      <td>portfolio manager</td>
      <td>No</td>
      <td>191</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



4. divide the data(datatrain and datatest) into two parts
   first part:"gender, country, profession, university degree"
   second part:"year of record, age, body height, size of city"


```python
data_object1 = pd.DataFrame(datatrain, columns=['Gender', 'Country', 'Profession', 'University Degree'], index=datatrain.index)
data_num1 = pd.DataFrame(datatrain, columns=['Year of Record', 'Age', 'Body Height [cm]','Size of City'], index=datatrain.index)

data_object2 = pd.DataFrame(datatest, columns=['Gender', 'Country', 'Profession', 'University Degree'], index=datatest.index)
data_num2 = pd.DataFrame(datatest, columns=['Year of Record', 'Age', 'Body Height [cm]','Size of City'], index=datatest.index)

```

5. to fill the missing value with most frequent value
   use simpleimputer() in sklearn


```python
#填充有缺失值的行
from sklearn.impute import SimpleImputer
#SimpleImputer中输入的至少是二维矩阵  
simple = SimpleImputer(missing_values = np.nan,strategy="most_frequent")
data_object1 = simple.fit_transform(data_object1.values)
data_num1 = simple.fit_transform(data_num1.values)

data_object2 = simple.fit_transform(data_object2.values)
data_num2 = simple.fit_transform(data_num2.values)
```

6. use ordinalEncoder() to encode the object value


```python
#对object类型的数据进行编码
from sklearn import preprocessing
encoder=preprocessing.OrdinalEncoder()
data_object1=encoder.fit_transform(data_object1)
X = np.c_[data_num1,data_object1]

print(X)
```

    [[1.997e+03 4.100e+01 1.930e+02 ... 1.200e+01 1.209e+03 1.000e+00]
     [1.996e+03 4.100e+01 1.860e+02 ... 1.230e+02 1.048e+03 2.000e+00]
     [2.018e+03 2.800e+01 1.700e+02 ... 1.060e+02 1.008e+03 4.000e+00]
     ...
     [1.993e+03 3.600e+01 1.530e+02 ... 1.320e+02 7.890e+02 2.000e+00]
     [2.019e+03 5.400e+01 1.900e+02 ... 4.100e+01 1.085e+03 3.000e+00]
     [2.017e+03 2.700e+01 1.740e+02 ... 1.330e+02 9.620e+02 3.000e+00]]
    

7. got the value of target


```python
#获得y值
train_income = pd.DataFrame(datatrain, columns=['Income in EUR'], index=datatrain.index)
#y = data_conti.values
y = train_income.values
print(y)
```

    [[ 61031.94416]
     [ 91001.32764]
     [157982.1767 ]
     ...
     [289951.3294 ]
     [100046.5278 ]
     [145886.2885 ]]
    

8. divide the train dataset into X_train, X_test, y_train, y_test


```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
print (X_train.shape)
print (y_train.shape)
print (X_test.shape)
print (y_test.shape)
```

    (89594, 8)
    (89594, 1)
    (22399, 8)
    (22399, 1)
    

9. use minmaxscaler() to scale the feature of X_train


```python
#对特征进行缩放
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
#特征很多的时候使用MinMaxScaler().partial_fit(data)来代替fit否则会报错
#scaler.fit(X)  #在这里本质是生成min(x)和max(x)
X_train = scaler.fit_transform(X_train)  #通过接口导出结果
print(X_train)
```

    [[0.94871795 0.18811881 0.59064327 ... 0.95597484 0.69678865 0.25      ]
     [0.94871795 0.0990099  0.59649123 ... 0.83018868 0.82374907 0.75      ]
     [0.02564103 0.41584158 0.64912281 ... 0.89937107 0.7356236  0.5       ]
     ...
     [0.41025641 0.21782178 0.28654971 ... 0.50314465 0.52352502 0.5       ]
     [0.1025641  0.16831683 0.48538012 ... 0.7672956  0.5690814  0.75      ]
     [0.92307692 0.01980198 0.39766082 ... 0.28301887 0.86407767 0.25      ]]
    

10. use LinearRegression to train the train dataset


```python
#训练模型
from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(X_train, y_train)

#from sklearn.ensemble import RandomForestRegressor
#forest_reg = RandomForestRegressor()
#forest_reg.fit(X_train, y_train)
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)



11. encode the X_test


```python
data_object2=encoder.fit_transform(data_object2)
X_test1 = np.c_[data_num2,data_object2]
print(X_test1)
```

    [[1.992e+03 2.100e+01 1.530e+02 ... 6.000e+01 1.115e+03 2.000e+00]
     [1.986e+03 3.400e+01 1.630e+02 ... 7.200e+01 5.780e+02 1.000e+00]
     [1.994e+03 5.300e+01 1.530e+02 ... 1.110e+02 7.150e+02 1.000e+00]
     ...
     [2.019e+03 5.000e+01 1.620e+02 ... 1.350e+02 7.890e+02 2.000e+00]
     [1.980e+03 5.400e+01 1.620e+02 ... 9.800e+01 6.730e+02 2.000e+00]
     [2.017e+03 4.100e+01 1.970e+02 ... 1.430e+02 7.560e+02 2.000e+00]]
    

12. combine the dataset of test before with the new X_test
    this is bigger than the original X_test
    but will perform better


```python
X_test2 = np.concatenate((X_test,X_test1),axis=0)
print(X_test2.shape)
print(X_test2)
```

    (95629, 8)
    [[2.011e+03 2.400e+01 1.720e+02 ... 1.360e+02 1.072e+03 1.000e+00]
     [1.995e+03 2.300e+01 1.780e+02 ... 7.300e+01 7.000e+01 1.000e+00]
     [1.995e+03 4.800e+01 2.100e+02 ... 1.130e+02 9.740e+02 1.000e+00]
     ...
     [2.019e+03 5.000e+01 1.620e+02 ... 1.350e+02 7.890e+02 2.000e+00]
     [1.980e+03 5.400e+01 1.620e+02 ... 9.800e+01 6.730e+02 2.000e+00]
     [2.017e+03 4.100e+01 1.970e+02 ... 1.430e+02 7.560e+02 2.000e+00]]
    

13. scale the testX dataset


```python
X_test = scaler.fit_transform(X_test2)  #通过接口导出结果
print(X_test)
```

    [[0.79487179 0.08928571 0.45637584 ... 0.85534591 0.80059746 0.25      ]
     [0.38461538 0.08035714 0.4966443  ... 0.4591195  0.05227782 0.25      ]
     [0.38461538 0.30357143 0.7114094  ... 0.71069182 0.72740851 0.25      ]
     ...
     [1.         0.32142857 0.38926174 ... 0.8490566  0.58924571 0.5       ]
     [0.         0.35714286 0.38926174 ... 0.6163522  0.50261389 0.5       ]
     [0.94871795 0.24107143 0.62416107 ... 0.89937107 0.56460045 0.5       ]]
    

14. use the testX to predict the income 


```python
#y_pred = forest_reg.predict(X_test2)
y_pred = linreg.predict(X_test)
print(y_pred)
```

    [[108353.0345478 ]
     [ 16930.05210477]
     [134827.46341319]
     ...
     [140670.48634148]
     [ 61700.29470528]
     [140642.4863915 ]]
    


```python
income = pd.DataFrame(y_pred, columns=['Income'])
print(income)
```

                  Income
    0      108353.034548
    1       16930.052105
    2      134827.463413
    3      147523.209325
    4      109886.356222
    ...              ...
    95624  140883.081288
    95625  149524.636216
    95626  140670.486341
    95627   61700.294705
    95628  140642.486391
    
    [95629 rows x 1 columns]
    

15. put the result into a file


```python
income.to_csv('linearimpro_pred.csv') 
```


```python

```


```python

```


```python

```

# tree model

#训练模型
from sklearn.tree import DecisionTreeRegressor 
tree_reg = DecisionTreeRegressor()
tree_reg.fit(X_train,y_train)

y_pred = tree_reg.predict(X_test)
print(y_pred)

# random forest model

from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor()
forest_reg.fit(X_train, y_train)

y_pred = forest_reg.predict(X_test)
print(y_pred)

