import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('C:\\Users\\student\\PSU\\LV4\\cars_processed.csv')
df1 = df.copy()
df2 = df.copy()
df1 = df1.sort_values(by=['selling_price'])
print(df1.tail())
print(df1.head())

godina2012 = 0
godina = df1.sort_values(by='year')
for godina in godina.year:
    if(godina == 2012):
        godina2012 += 1

print('godina 2012:' , godina2012)

df1 = df1.sort_values(by=['km_driven'])
print(df1.tail())
print(df1.head())

prosjek= df2[(df2.fuel == 'Diesel')] 
prosjek = prosjek.iloc[:,3:4]
print('km')
print(prosjek.mean())

prosjek2= df2[(df2.fuel == 'Petrol')] 
prosjek2 = prosjek2.iloc[:,3:4]
print('km')
print(prosjek2.mean())

print(df.info())

sns.pairplot(df, hue='fuel')
sns.relplot(data=df, x='km_driven', y='selling_price', hue='fuel')
df = df.drop(['name','mileage'], axis=1)
obj_cols = df.select_dtypes(object).columns.values.tolist()
num_cols = df.select_dtypes(np.number).columns.values.tolist()
fig = plt.figure(figsize=[15,8])
for col in range(len(obj_cols)):
 plt.subplot(2,2,col+1)
 sns.countplot(x=obj_cols[col], data=df)
df.boxplot(by ='fuel', column =['selling_price'], grid = False)
df.hist(['selling_price'], grid = False)
tabcorr = df.corr()
sns.heatmap(df.corr(), annot=True, linewidths=2, cmap= 'coolwarm')
plt.show()