
import pandas as pd
import numpy as np
import plotly.express as px
#importing data
#creating a dataframe
data = pd.read_csv("credit card.csv")
#cleaning
#checking the data
#to work with ease the data is already cleaned
#Representation of data 
typ = data["type"].value_counts()
transactions = typ.index
quantity = typ.values
figure = px.pie(data,values=quantity,names=transactions,hole=0.5,title="Distribution of Transaction types")
a = int(input("Press 1 to see the distribution of data or 0 not to see : "))
if a == 1:
    figure.show()
else:
    print("")
#calculating correlation
correlation = data.corr(numeric_only = True)
#calculating correlation with respect to "isFraud column"
b = int(input("""Press 1 to see the correlation of data with "isFraud" column or 0 : """ ))
if b == 1:
    print(correlation["isFraud"].sort_values(ascending=False))
else:
    pass
#Transforming the categorial features into numericals
data["type"] = data["type"].map({"CASH_OUT":1,"PAYMENT":2,"CASH_IN":3,"TRANSFER":4,"DEBIT":5})
data["isFraud"] = data["isFraud"].map({0:"No Fraud",1:"Fraud"})
#splitting data
from sklearn.model_selection import train_test_split
x = np.array(data[["type","amount","oldbalanceOrg","newbalanceOrig"]]
            )
y = np.array(data[["isFraud"]])
#Training a machine learning model
from sklearn.tree import DecisionTreeClassifier
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.10,random_state=42)
#we selected test_size factor as 0.10 which means we are alloting 10 percent of data for testing
model = DecisionTreeClassifier()
model.fit(xtrain,ytrain)
#prediction
#features = [type,amount,oldbalanceOrg,newbalanceOrig]
c = 4
l = []
print("Types of Transactions...")
print("""Enter 
1 for CASH_OUT
2 for PAYMENT
3 for CASH_IN
4 for TRANSFER
5 for DEBIT
""")
d = ["type of online transaction ","Amount","balance before the transaction ", "balance after the transaction "]
for i in range(0,4):
    l.append(float(input(f'Enter {d[i]} : ')))
features = np.array([l])
print(model.predict(features))
