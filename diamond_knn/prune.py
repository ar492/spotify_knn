# The data is scrapped from Australian Diamond Importers website on 24th Feb 2022.
# all of it is from feb 24

import pandas as pd
from sklearn.utils import shuffle
f=pd.read_csv("diamonds.csv")
f = shuffle(f)

# keep_col = ['shape','size', 'clarity', 'cut', 'symmetry', 'polish', 'depth_percent', 
# 'table_percent', 'meas_length', 'meas_width', 'meas_depth', 'fluor_intensity',
# 'lab', 'total_sales_price']

keep_col = ['size', 'depth_percent', 
'table_percent', 'meas_length', 'meas_width', 'meas_depth', 'total_sales_price']

new_f = f[keep_col]

for j in keep_col:
	new_f.dropna(subset=[j], inplace=True) # delete all rows with a hole somewhere
drop=len(new_f)-100000
new_f=new_f.iloc[:-drop]

new_f.to_csv("pruned.csv", index=False)