from flask import Flask,render_template,redirect,request,session,flash,make_response,url_for
from flask_bcrypt import bcrypt
from flask_sqlalchemy import SQLAlchemy
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score

app=Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI']='sqlite:///parkinson.db'

db=SQLAlchemy(app)

app.secret_key = 'WORK_IT_OUT'

class employee(db.Model):
    id=db.Column(db.String(50),primary_key=True)
    password=db.Column(db.String(100))

    def __init__(self,id,password):
        self.id=id
        self.password=bcrypt.hashpw(password.encode('utf-8'),bcrypt.gensalt()).decode('utf-8')
    
    def check_password(self,password):
        return bcrypt.checkpw(password.encode('utf-8'),self.password.encode('utf-8'))
    

class patient_prediction(db.Model):
    id=db.Column(db.Integer,primary_key=True)
    patient_id=db.Column(db.Integer,nullable=False)
    patient_name=db.Column(db.String,nullable=False)
    MDVP_Fo_Hz=db.Column(db.Integer)
    MDVP_Fhi_Hz=db.Column(db.Integer)
    MDVP_Flo_Hz=db.Column(db.Integer)
    MDVP_Jitter_percent_Hz=db.Column(db.Integer)
    MDVP_Jitter_abs_Hz=db.Column(db.Integer)
    MDVP_RAP=db.Column(db.Integer)
    MDVP_PPQ=db.Column(db.Integer)
    Jitter_DDP=db.Column(db.Integer)
    MDVP_Shimmer=db.Column(db.Integer)
    MDVP_Shimmer_dB=db.Column(db.Integer)
    Shimmer_APQ3=db.Column(db.Integer)
    Shimmer_APQ5=db.Column(db.Integer)
    MDVP_APQ=db.Column(db.Integer)
    Shimmer_DDA=db.Column(db.Integer)
    NHR=db.Column(db.Integer)
    HNR=db.Column(db.Integer)
    RPDE=db.Column(db.Integer)
    DFA=db.Column(db.Integer)
    spread1=db.Column(db.Integer)
    spread2=db.Column(db.Integer)
    D2=db.Column(db.Integer)
    PPE=db.Column(db.Integer)
    status=db.Column(db.Integer)

    def __init__(self,patient_id,patient_name, MDVP_Fo_Hz,MDVP_Fhi_Hz,MDVP_Flo_Hz,MDVP_Jitter_percent_Hz,MDVP_Jitter_abs_Hz,MDVP_RAP,MDVP_PPQ,Jitter_DDP,MDVP_Shimmer,MDVP_Shimmer_dB,Shimmer_APQ3,Shimmer_APQ5,MDVP_APQ,Shimmer_DDA,NHR,HNR,RPDE,DFA,spread1,spread2,D2,PPE,status):
        self.patient_id=patient_id
        self.patient_name=patient_name
        self.MDVP_Fo_Hz=MDVP_Fo_Hz
        self.MDVP_Fhi_Hz=MDVP_Fhi_Hz
        self.MDVP_Flo_Hz=MDVP_Flo_Hz
        self.MDVP_Jitter_percent_Hz=MDVP_Jitter_percent_Hz
        self.MDVP_Jitter_abs_Hz=MDVP_Jitter_abs_Hz
        self.MDVP_RAP=MDVP_RAP
        self.MDVP_PPQ=MDVP_PPQ
        self.Jitter_DDP=Jitter_DDP
        self.MDVP_Shimmer=MDVP_Shimmer
        self.MDVP_Shimmer_dB=MDVP_Shimmer_dB
        self.Shimmer_APQ3=Shimmer_APQ3
        self.Shimmer_APQ5=Shimmer_APQ5
        self.MDVP_APQ=MDVP_APQ
        self.Shimmer_DDA=Shimmer_DDA
        self.NHR=NHR
        self.HNR=HNR
        self.RPDE=RPDE
        self.DFA=DFA
        self.spread1=spread1
        self.spread2=spread2
        self.D2=D2
        self.PPE=PPE
        self.status=status

    
    
with app.app_context():
    #db.drop_all()
    db.create_all()

@app.route('/')
def index():
    return render_template('signup.html')

@app.route('/signup',methods=['GET','POST'])
def signup():
    if request.method=='POST':
        id=request.form['id']
        password=request.form['password']


        if employee.query.filter_by(id=id).first():
            flash('user already exists')
            return render_template('signup.html')

        else:
            new_user=employee(id=id,password=password)
            db.session.add(new_user)
            db.session.commit()
            flash('Your account is created successfully.')
            return redirect('/login')
        
    
@app.route('/login',methods=['GET','POST'])
def login():
    if request.method=='POST':
        id=request.form['id']
        password=request.form['password']

        user=employee.query.filter_by(id=id).first()

        if user and user.check_password(password):
            session['id']=user.id
            session['password']=user.password
            return render_template('predict.html')
        else:
            flash('Invalid credentials')
            return render_template('login.html')
        
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('name',None)
    return redirect('/login')

@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method=='POST':
        patient_id=request.form['patient_id']
        patient_name=request.form['patient_name']
        c1=request.form['c1']
        c2=request.form['c2']
        c3=request.form['c3']
        c4=request.form['c4']
        c5=request.form['c5']
        c6=request.form['c6']
        c7=request.form['c7']
        c8=request.form['c8']
        c9=request.form['c9']
        c10=request.form['c10']
        c11=request.form['c11']
        c12=request.form['c12']
        c13=request.form['c13']
        c14=request.form['c14']
        c15=request.form['c15']
        c16=request.form['c16']
        c17=request.form['c17']
        c18=request.form['c18']
        c19=request.form['c19']
        c20=request.form['c20']
        c21=request.form['c21']
        c22=request.form['c22']

    
        parkinsons_data = pd.read_csv('parkinsons.csv')
        parkinsons_data.head()

# number of rows and columns in the dataframe
        parkinsons_data.shape

# getting more information about the dataset
        parkinsons_data.info()

# checking for missing values in each column
        parkinsons_data.isnull().sum()

# getting some statistical measures about the data
        parkinsons_data.describe()

# distribution of target Variable
        parkinsons_data['status'].value_counts()

# grouping the data bas3ed on the target variable
        parkinsons_data_numeric = parkinsons_data.drop(columns=['name'], axis=1)
        grouped_data = parkinsons_data_numeric.groupby('status').mean()

        X = parkinsons_data.drop(columns=['name','status'], axis=1)
        Y = parkinsons_data['status']


        #print(Y)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

        #print(X.shape, X_train.shape, X_test.shape)
        scaler = StandardScaler()

        scaler.fit(X_train)

        X_train = scaler.transform(X_train)

        X_test = scaler.transform(X_test)

        #print(X_train)

        model = svm.SVC(kernel='rbf')

# training the SVM model with training data
        model.fit(X_train, Y_train)

# accuracy score on training data
        X_train_prediction = model.predict(X_train)
        training_data_accuracy = accuracy_score(Y_train, X_train_prediction)

        #print('Accuracy score of training data : ', training_data_accuracy)

# accuracy score on training data
        X_test_prediction = model.predict(X_test)
        test_data_accuracy = accuracy_score(Y_test, X_test_prediction)

        #print('Accuracy score of test data : ', test_data_accuracy)
        input_data=(c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15,c16,c17,c18,c19,c20,c21,c22)
    #input_data = (197.07600,206.89600,192.05500,0.00289,0.00001,0.00166,0.00168,0.00498,0.01098,0.09700,0.00563,0.00680,0.00802,0.01689,0.00339,26.77500,0.422229,0.741367,-7.348300,0.177551,1.743867,0.085569)

# changing input data to a numpy array
        input_data_as_numpy_array = np.asarray(input_data)

# reshape the numpy array
        input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# standardize the data
        std_data = scaler.transform(input_data_reshaped)

        prediction = model.predict(std_data)
        #print(prediction)

        new_test=patient_prediction(patient_id=patient_id,patient_name=patient_name, MDVP_Fo_Hz=c1,MDVP_Fhi_Hz=c2,MDVP_Flo_Hz=c3,MDVP_Jitter_percent_Hz=c4,MDVP_Jitter_abs_Hz=c5,MDVP_RAP=c6,MDVP_PPQ=c7,Jitter_DDP=c8,MDVP_Shimmer=c9,MDVP_Shimmer_dB=c10,Shimmer_APQ3=c11,Shimmer_APQ5=c12,MDVP_APQ=c13,Shimmer_DDA=c14,NHR=c15,HNR=c16,RPDE=c17,DFA=c18,spread1=c19,spread2=c20,D2=c21,PPE=c22,status=prediction[0])
        db.session.add(new_test)
        db.session.commit()

        if (prediction[0] == 0):
            return "The Person does not have Parkinsons Disease"

        else:
            return "The Person has Parkinsons"
    
        
if __name__=='__main__':
  
    app.run(debug=True)