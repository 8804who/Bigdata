import pandas as pd
import numpy as np
from pandas import Series, DataFrame
list_data=[1,2,3,4,5]
list_name=["a","b","c","d","e"]
example_obj=Series(data=list_data,index=list_name)
print(example_obj)
