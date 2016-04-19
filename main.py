import pandas as pd;
import numpy as np;

training_data=pd.read_csv("data/train.csv");#Store training data into a pandas dataframe
#print training_data.tail(10);
training_data=training_data.drop(['Name','Ticket','Cabin'],axis=1); # dropping the Name Ticket and Cabin data because it is irrelevant for the context
training_data=training_data.dropna(); # Drop the rows with NA values


training_data['Gender'] = training_data['Sex'].map({'female': 0, 'male':1}).astype(int); #Change gender info from char to int

training_data['Port'] = training_data['Embarked'].map({'C':1, 'S':2, 'Q':3}).astype(int) # Change embarked info from char to int

training_data=training_data.drop(['Sex','Embarked'],axis=1); #drop the previous rows with char data

#print training_data.info();

cols=training_data.columns.tolist();
cols = cols[1:2] + cols[0:1]+cols[2:];
#print cols

training_data=training_data[cols]; # Rearranging the columns so that the survived column ( target) appears first
#print type(training_data);
#print training_data.head(10);
#print training_data.info();

train_data=training_data.values; # Transform the training data into a numpy array for applying to scikit learn
#print train_data[:10];


from sklearn.ensemble import RandomForestClassifier #design a Random Forest Classifier

model = RandomForestClassifier(n_estimators = 100)
#print train_data[0:,2:];

model = model.fit(train_data[0:,2:], train_data[0:,0]) # The model is trained

# The next step is to test the trained model using the testing data
#Similar steps for filtering the testing data as the training data.


test=pd.read_csv("data/test.csv");#Store training data into a pandas dataframe
#print test.tail(10);
test=test.drop(['Name','Ticket','Cabin'],axis=1); # dropping the Name Ticket and Cabin data because it is irrelevant for the context
test=test.dropna(); # Drop the rows with NA values


test['Gender'] = test['Sex'].map({'female': 0, 'male':1}).astype(int); #Change gender info from char to int

test['Port'] = test['Embarked'].map({'C':1, 'S':2, 'Q':3}).astype(int) # Change embarked info from char to int

test=test.drop(['Sex','Embarked'],axis=1); #drop the previous rows with char data

#print test.info();
#print test.head(10);


test_data=test.values; # Transform the test data into a numpy array for applying to scikit learn
output = model.predict(test_data[:,1:]) #predicting the output of the test data

#print output
result = np.c_[test_data[:,0].astype(int), output.astype(int)];
#print result
df_result = pd.DataFrame(result[:,0:2], columns=['PassengerId', 'Survived'])

#print df_result.head(10);
df_result.to_csv('results/titanic_1-0.csv', index=False)


#print df_result.shape;