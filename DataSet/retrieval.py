#reads the property value data from the avm csv
import pandas as pd
df = pd.read_csv('PropertyValueData/AVM.csv')
print(df.head())

#reads the property deed data from the PropertyDeedData csv
import pandas as pd
df = pd.read_csv('PropertyDeedData/PropertyDeedData.csv')
print(df.head())

#Now extraction and prediction algorithms
