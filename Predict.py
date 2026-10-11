import pickle
import pandas as pd

#Đọc dữ liệu từ file
data = pd.read_csv('./data/Test_USA_Housing.csv')
X=data

#Load model: linear_reg_model.sav
filename='Models/linear_reg_model.sav'
load_model = pickle.load(open(filename, 'rb'))

#Dự báo nhãn của tập X
y_pred = load_model.predict(X)
print('y_pred:',y_pred)