import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle as pickle


def get_clean_data():
    df= pd.read_csv("C:/Users/HP/Downloads/Cancer/Cancerdataset.csv")
    df=df.drop(['Unnamed: 32','id'], axis=1)
    df['diagnosis'].replace({'M':1,'B':0},inplace=True)
    return df

def create_model(df):
    x= df.drop(['diagnosis'], axis=1)
    y= df['diagnosis']
    
    scaler=StandardScaler()
    x= scaler.fit_transform(x)
    
    x_train , x_test , y_train , y_test =train_test_split(
        x, y, test_size=0.2 , random_state=42
    )
    
    model=LogisticRegression()
    model.fit(x_train , y_train)
    
    #testing the model
    y_pred = model.predict(x_test)
    print('Accuracy of our model: ', accuracy_score(y_test, y_pred))
    print("Classification report: \n", classification_report(y_test, y_pred))
    return model, scaler


def main():
    df=get_clean_data()
    
    model, scalar =create_model(df)
    
    
    with open('model/model.pkl', 'wb') as f:
       pickle.dump(model, f)
    with open('model/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f) 

    
if __name__ =='__main__':
    main()
