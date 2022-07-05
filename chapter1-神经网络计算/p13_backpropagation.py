from sklearn import datasets
from pandas import DataFrame
import pandas as pd

x_data = datasets.load_iris().data   # [150, 4]
y_data = datasets.load_iris().target  # 标签：0，1，2
print("x_data from iris datasets: \n", x_data)
print("y_data from iris datasets: \n", y_data)

# print(len(x_data))
# 表格形式展现
x_data = DataFrame(x_data, index=range(len(x_data)), columns=['花萼长', '花萼宽', '花瓣长', '花瓣宽'])
pd.set_option('display.unicode.east_asian_width', True)
print('x_data add index: \n', x_data)

x_data['类别'] = y_data
print("x_data add label: \n", x_data)   # [150, 5]
