import pandas as pd
import numpy as np
import openpyxl
from pandas import Series, DataFrame
list_data=[1,2,3,4,5]
list_name=["a","b","c","d","e"]
example_obj=Series(data=list_data,index=list_name)
print(example_obj)

df=pd.read_excel('D:/공부/수업 자료/4-1/USG공유대학/빅데이터응용/실습자료/excel-comp-data.xlsx')
print(df)
print(df.head(5))
print(df.head(3).T)