import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


data=pd.read_csv("data.csv",sep=";")
data=data.melt(id_vars=["H03","H05","H16"],var_name="Date",value_name="sales")
pattern=r'(MO)(\d{2})(\d{4})'
data.loc[:,"Date"]=data.loc[:,"Date"].astype(str)
data[["Prefix","Month","Year"]]=data.loc[:,"Date"].str.extract(pattern)
data["date"]=pd.to_datetime(data.loc[:,"Year"].astype(str)+'-'+data.loc[:,"Month"].astype(str)+'-'+'01',format='%Y-%m-%d')
data.drop(columns=["Prefix","Month","Year","Date"],inplace=True)
data=pd.get_dummies(data,columns=["H03","H05","H16","date"])
x=data.iloc[:,1:]
y=data.loc[:,"sales"]
print(x.head())
print(y.head())
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
model=LinearRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
r2score=metrics.r2_score(y_test,y_pred)
print(r2score)
with open('results.txt','w') as f:
    f.write(f"R2-score is :{r2score}")
plt.scatter(y_test,y_pred,c="blue",label="Actual vs Predicted")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.show()
plt.savefig("Actual vs Predicted.png")