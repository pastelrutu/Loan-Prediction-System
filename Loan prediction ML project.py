import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
df=pd.read_csv("train_u6lujuX_CVtuZ9i.csv")
# Text columns -> Mode
df["Gender"].fillna(df["Gender"].mode()[0],inplace=True)
df["Self_Employed"].fillna(df["Self_Employed"].mode()[0],inplace=True)
df["Credit_History"].fillna(df["Credit_History"].mode()[0],inplace=True)
# Numerical columns
df["LoanAmount"].fillna(df["LoanAmount"].median(), inplace=True)
df["Loan_Amount_Term"].fillna(df["Loan_Amount_Term"].mode()[0],inplace=True)

le = LabelEncoder()
df["Gender"]=le.fit_transform(df["Gender"])
df["Married"]=le.fit_transform(df["Married"])
df["Education"]=le.fit_transform(df["Education"])
df["Self_Employed"]=le.fit_transform(df["Self_Employed"])
df["Property_Area"]=le.fit_transform(df["Property_Area"])
df["Loan_Status"]=le.fit_transform(df["Loan_Status"])
df["Dependents"]=le.fit_transform(df["Dependents"])
x=df.iloc[:,1:12]
y=df.iloc[:,12]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

models={
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Naive Bayes": GaussianNB(),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC()
    }
best_accuracy=0
best_model=None
best_model_name=""
for name,models in models.items():
    models.fit(x_train,y_train)
    predict_result=models.predict(x_test)
    acc=accuracy_score(y_test,predict_result)
    if acc>best_accuracy:
        best_accuracy=acc
        best_model=models
        best_model_name=name
    precision=precision_score(y_test,predict_result)
    recall=recall_score(y_test,predict_result)
    f1=f1_score(y_test,predict_result)
    cm=confusion_matrix(y_test,predict_result)

    print("-------------")
    print(name)
    print("Accuracy: ",acc)
    print("Precision: ",precision)
    print("Recall: ",recall)
    print("F1: ",f1)
    print("Confusion matrix:")
    print(cm)
print("----------------------------")
print("Best Model :",best_model_name)
print("Best Accuracy :",best_accuracy)
print("----------------------------")

gender=int(input("Enter Gender (1:Male,0:Female): "))
married=int(input("Enter marriage Status (1:Yes,0:No): "))
dependents=int(input("Enter Number of Dependents(0,1,2,3).(Enter 3 if u have more than 3 or 3): "))
education=int(input("Education (1:Graduate,0:Not Graduate): "))
self_employed=int(input("Are you Self-Employed (1:Yes,0:No): "))
applicant_income=int(input("Applicant Income: "))
coapplicant_income=int(input("Enter Co-applicant Income: "))
loan_amt=int(input("Enter Loan Amount: "))
loan_amt_term=int(input("Enter Loan Amount Term: "))
credit_history=int(input("Enter Credit History(0,1): "))
property_area=int(input("Entter Your Property area as (0:Urban,1:Semiurban,2:Rural): "))

y_pred=best_model.predict([[
    gender,married,dependents,education,self_employed,applicant_income,coapplicant_income,
    loan_amt,loan_amt_term,credit_history,property_area]])
if y_pred[0]==1:
    print("Loan Approved!")
else:
    print("Loan Rejected!")




   
    
    


