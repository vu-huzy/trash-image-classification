#Khai báo các thư viện/hàm cần dùng
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error as MAE, mean_squared_error as RMSE, r2_score
import pickle

#Đọc dữ liệu từ file csv
data = pd.read_csv('./data/USA_Housing.csv')

#Tách tập dữ liệu data thành 2 tập dt_Train và dt_Test, với tỉ lệ dt_Test=30%, dt_Train=70%
dt_Train, dt_Test = train_test_split(data, test_size=0.3, shuffle=False)

#Tách dt_Train thành 2 phần: tập mẫu X_train và tập nhãn y_train
X_train = dt_Train.iloc[:, :5]
y_train= dt_Train.iloc[:, 5]

#Tách dt_Train thành 2 phần: tập mẫu X_train và tập nhãn y_train
X_test = dt_Test.iloc[:, :5]
y_test = dt_Test.iloc[:, 5]

#Khai báo mô hình (tên của phương pháp học máy và các tham số tương ứng)
model = LinearRegression()
#model2 = Lasso()

#Gọi hàm fit để huấn luyện mô hình trên tập (X_train, y_train)
model.fit(X_train, y_train)

#In trọng số của mô hình hồi quy tuyến tính
print('w=', model.coef_)
print('w0=', model.intercept_)

#Sử dụng model đã được huấn luyện để dự đoán nhãn của các mẫu trong tập X_test
y_pred = model.predict(X_test)

#Đánh giá chất lượng dự báo của model
print("R2: %.2f" % r2_score(y_test, y_pred))
print('MAE:', MAE(y_test, y_pred))
print('RMSE:', RMSE(y_test,y_pred))

#Ghi mô hình đã được huấn luyện
file_model = 'Models\linear_reg_model.sav'
pickle.dump(model, open(file_model, 'wb'))





