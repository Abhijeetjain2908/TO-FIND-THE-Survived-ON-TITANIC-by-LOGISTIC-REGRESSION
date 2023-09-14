import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

titanic_train=pd.read_csv(r"C:\Users\HP\Desktop\machine learning\dataset\titanic.csv",header=0)
print(titanic_train)

# print(titanic_train.info())
# print(titanic_train.isnull())

# fill the NaN or Change null values to the median of Age

Age_median=titanic_train.Age.median(axis='index', skipna=True)
print(Age_median)
titanic_train.Age.fillna(Age_median, inplace=True)

print(titanic_train['Age'])
print(titanic_train.Age.isnull().sum())


# fill the NaN or Change null values to the median of Fare

# Fare_median=titanic_train.Fare.median(axis='index', skipna=True,)
Fare_mean=titanic_train.Fare.mean(axis='index', skipna=True)
print(Fare_mean)
titanic_train.Fare.fillna(Fare_mean, inplace=True)


print(titanic_train['Fare'])
print(titanic_train.Fare.isnull().sum())


char_cabin = titanic_train["Cabin"].astype(str)     # Convert cabin to str

new_Cabin = np.array([cabin[0] for cabin in char_cabin]) # Take first letter

titanic_train["Cabin"] = pd.Categorical(new_Cabin)  # Save the new cabin var
print(titanic_train["Cabin"])

from sklearn import linear_model

from sklearn import preprocessing

# Initialize label encoder

label_encoder = preprocessing.LabelEncoder()

# Convert Sex variable to numeric

encoded_sex = label_encoder.fit_transform(titanic_train["Sex"])


# Initialize logistic regression model

log_model = linear_model.LogisticRegression(solver = 'lbfgs') #lbfgs is method of Solver

# Train the model

log_model.fit(X = pd.DataFrame(encoded_sex), 

              y = titanic_train["Survived"])

# Check trained model intercept

print(log_model.intercept_)

# Check trained model coefficients

print(log_model.coef_)

# Make predictions
preds = log_model.predict_proba(X= pd.DataFrame(encoded_sex))

preds = pd.DataFrame(preds)
preds.columns = ["Death_prob", "Survival_prob"]

# Generate table of predictions vs Sex
print(pd.crosstab(titanic_train["Sex"], preds.loc[:, "Survival_prob"]))



# Convert more variables to numeric
encoded_class = label_encoder.fit_transform(titanic_train["Pclass"])
encoded_cabin = label_encoder.fit_transform(titanic_train["Cabin"])

train_features = pd.DataFrame([encoded_class,
                              encoded_cabin,
                              encoded_sex,
                              titanic_train["Age"]]).T

# print(train_features)

# Initialize logistic regression model
log_model = linear_model.LogisticRegression(solver = 'lbfgs')

# Train the model
log_model.fit(X = train_features ,
              y = titanic_train["Survived"])

# Check trained model intercept
print(log_model.intercept_)

# Check trained model coefficients
print(log_model.coef_)

# Make predictions
preds = log_model.predict(X= train_features)

# Generate table of predictions vs actual
pd.crosstab(preds,titanic_train["Survived"])

#to find accuracy
score=log_model.score(X = train_features ,
                y = titanic_train["Survived"])
print(score)


from sklearn import metrics

# View confusion matrix
metrics.confusion_matrix(y_true=titanic_train["Survived"],  # True labels
                         y_pred=preds) # Predicted labels

# View summary of common classification metrics
print(metrics.classification_report(y_true=titanic_train["Survived"],
                                    y_pred=preds) ) 


titanic_test=pd.read_csv(r"C:\Users\HP\Desktop\machine learning\dataset\test data for titanic.csv",header=0)

# fill the NaN or Change null values to the median of Age

Age_median=titanic_test.Age.median(axis='index', skipna=True)
print(Age_median)
titanic_test.Age.fillna(Age_median, inplace=True)

# Fare_median=titanic_train.Fare.median(axis='index', skipna=True,)
# Fare_mean=titanic_test.Fare.mean(axis='index', skipna=True)
# print(Fare_mean)
# titanic_test.Fare.fillna(Fare_mean, inplace=True)


char_cabin = titanic_test["Cabin"].astype(str)     # Convert cabin to str
new_Cabin = np.array([cabin[0] for cabin in char_cabin]) # Take first letter
titanic_test["Cabin"] = pd.Categorical(new_Cabin)  # Save the new cabin var
print(titanic_test["Cabin"])

# Read and prepare test data

# titanic_test = pd.read_csv("test.csv")    # Read the data

# char_cabin = titanic_test["Cabin"].astype(str)     # Convert cabin to str

# new_Cabin = np.array([cabin[0] for cabin in char_cabin]) # Take first letter

# titanic_test["Cabin"] = pd.Categorical(new_Cabin)  # Save the new cabin var

# # Impute median Age for NA Age values

# new_age_var = np.where(titanic_test["Age"].isnull(), # Logical check

#                        28,                       # Value if check is true

#                        titanic_test["Age"])      # Value if check is false

# titanic_test["Age"] = new_age_var

encoded_sex = label_encoder.fit_transform(titanic_test["Sex"])
encoded_class = label_encoder.fit_transform(titanic_test["Pclass"])
encoded_cabin = label_encoder.fit_transform(titanic_test["Cabin"])

test_features = pd.DataFrame([encoded_class,
                              encoded_cabin,
                              encoded_sex,
                              titanic_test["Age"]]).T

print(test_features)
# log_model = linear_model.LogisticRegression(solver = 'lbfgs')

# Train the model
# log_model.fit(X = test_features ,
#               y = titanic_test["Survived"])

# Check trained model intercept
# print(log_model.intercept_)

# # Check trained model coefficients
# print(log_model.coef_)

# Make predictions
test_preds = log_model.predict(X= test_features)
print(test_preds)
print(titanic_test)

# Create a submission for Kaggle
submission = pd.DataFrame({"PassengerId":titanic_test["PassengerId"],
                           "Survived":test_preds})

# Save submission to CSV
submission.to_csv(r"C:\Users\HP\Desktop\machine learning\data after prediction of titanic\tutorial_logreg_submission.csv", 
                  index=False)       # Do not save index values

