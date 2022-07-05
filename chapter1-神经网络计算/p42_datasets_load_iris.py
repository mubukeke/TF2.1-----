from sklearn.datasets import load_iris
from pandas import DataFrame
import pandas as pd

x_data = load_iris().data
y_data = load_iris().target

print("x_data from iris: \n", x_data)
print("y_data from iris: \n", y_data)

x_data = DataFrame(x_data, columns=['花萼长', '花萼宽', '花瓣长', '花瓣宽'])
pd.set_option('display.unicode.east_asian_width', True)
print(x_data)

x_data['类别'] = y_data
print("x_data add a column:\n", x_data)
