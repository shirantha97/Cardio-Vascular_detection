import pandas as pd
import numpy as np
import cloudpickle
from flask import Flask
from statistics import mode
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import GridSearchCV
import sklearn.metrics as metrics
from sklearn.pipeline import Pipeline

cardioVascularDF = pd.read_csv('Dataset1 - cardio-vascular dataset.csv', index_col=False)
cardioVascularDF.info()
cardioVascularDF.describe().transpose()

print(cardioVascularDF.describe(include="all"))
cardioVascularDF["education"] = cardioVascularDF["education"].fillna(mode(cardioVascularDF["education"]))
cardioVascularDF["cigsPerDay"] = cardioVascularDF["cigsPerDay"].fillna(cardioVascularDF["cigsPerDay"].mean())
cardioVascularDF["BPMeds"] = cardioVascularDF["BPMeds"].fillna(mode(cardioVascularDF["BPMeds"]))
cardioVascularDF["totChol"] = cardioVascularDF["totChol"].fillna(cardioVascularDF["totChol"].mean())
cardioVascularDF["glucose"] = cardioVascularDF["glucose"].fillna(cardioVascularDF["glucose"].mean())
cardioVascularDF["BMI"] = cardioVascularDF["BMI"].fillna(cardioVascularDF["BMI"].mean())
cardioVascularDF["heartRate"] = cardioVascularDF["heartRate"].fillna(cardioVascularDF["heartRate"].mean())

data = pd.DataFrame(cardioVascularDF.values)
X_features = cardioVascularDF.drop(["TenYearCHD"], axis=1)
Y_features = cardioVascularDF["TenYearCHD"]
bestFeatures = SelectKBest(score_func=chi2, k=10)
fit = bestFeatures.fit(X_features, Y_features)
dataScore = pd.DataFrame(fit.scores_)
dataColumns = pd.DataFrame(X_features.columns)
featureScore = pd.concat([dataColumns, dataScore], axis=1)
featureScore.columns = ["feature", "score"]
print(featureScore)

X = cardioVascularDF[
    ["sysBP", "glucose", "age", "totChol", "cigsPerDay", "diaBP", "prevalentHyp", "diabetes", "BPMeds", "male", "BMI",
     "prevalentStroke"]]
Y = cardioVascularDF["TenYearCHD"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

# scaler = MinMaxScaler(feature_range=(0, 1))
# scaler.fit(X_train)
# X_train = scaler.transform(X_train)
# X_test = scaler.transform(X_test)

# logreg = LogisticRegression(max_iter=1000)
# grid_values = {'penalty': ['l2'], 'C': [0.001, .009, 0.01, .09, 1, 5, 10, 25, 50]}
# grid_clf_acc = GridSearchCV(logreg, param_grid=grid_values, scoring='recall')
# grid_clf_acc.fit(X_train, Y_train)
#
# y_pred = grid_clf_acc.predict(X_test)
# print('Accuracy Score : ' + str(metrics.accuracy_score(Y_test, y_pred)))

sc = StandardScaler()
pca = decomposition.PCA()
logistic = LogisticRegression()
pipe = Pipeline(steps=[('sc', sc),
                       ('pca', pca),
                       ('logistic', logistic)])
n_components = list(range(1, X.shape[1] + 1, 1))

C = np.logspace(-4, 4, 50)
penalty = ['l2']
parameters = dict(pca__n_components=n_components,
                  logistic__C=C,
                  logistic__penalty=penalty)
clf = GridSearchCV(pipe, parameters)
clf.fit(X_train, Y_train)
fileName = 'finalModel.sav'
pickleModel = cloudpickle.dump(clf, open(fileName, 'wb'))
print('Best C:', clf.best_estimator_.get_params()['logistic__C'])
print('Best Number Of Components:', clf.best_estimator_.get_params()['pca__n_components'])
model = cloudpickle.load(open(fileName, 'rb'))
y_pred = model.predict(X_test)
print('Accuracy Score : ' + str(metrics.accuracy_score(Y_test, y_pred)))
