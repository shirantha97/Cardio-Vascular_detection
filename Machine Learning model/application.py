import numpy as np
import pandas as pd
import pickle
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

data_raw = pd.read_csv('cardio_train.csv', sep=";")
data_raw.drop("id", axis=1, inplace=True)

dup_data = data_raw[data_raw.duplicated(keep=False)]
dup_data = dup_data.sort_values(by=['age', "gender", "height"], ascending=True)
data_raw.drop_duplicates(inplace=True)

x = data_raw.copy(deep=True)

col_list = ["age", "height", "weight", "ap_hi", "ap_lo"]


def data_standartization(x):
    std_data = x.copy(deep=True)
    for col in col_list:
        std_data[col] = (std_data[col] - std_data[col].mean()) / std_data[col].std()
    return std_data


standard_x = data_standartization(x)

blood_pressure = ['ap_hi', 'ap_lo']
data_boundary = pd.DataFrame(index=["lower", "upper"])

for each in blood_pressure:
    Q1 = x[each].quantile(0.25)
    Q3 = x[each].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    data_boundary[each] = [lower_bound, upper_bound]

# remove outliers from the glucose levels
ap_hi_filter = (x["ap_hi"] > data_boundary["ap_hi"][1])
ap_lo_filter = (x["ap_lo"] > data_boundary["ap_lo"][1])
outlier_filter = (ap_hi_filter | ap_lo_filter)
x_outliers = x[outlier_filter]

out_filter = ((x["ap_hi"] > 250) | (x["ap_lo"] > 200))

x = x[~out_filter]

x["bmi"] = x["weight"] / (x["height"] / 100) ** 2

x["gender"] = x["gender"] % 2

y = x["cardio"]
x.drop("cardio", axis=1, inplace=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

log_reg = LogisticRegression(solver="liblinear", max_iter=200)
grid = {"penalty": ["l2"],
        "C": np.arange(60, 80, 2)}
log_reg_cv = GridSearchCV(log_reg, grid, cv=3)
log_reg_cv.fit(x_train, y_train)

fileName = 'finalModel.sav'
pickle.dump(log_reg_cv, open(fileName, 'wb'))

print("Tuned hyperparameter n_estimators: {}".format(log_reg_cv.best_params_))
print("Best score: {}".format(log_reg_cv.best_score_))

# y_pred = log_reg_cv.predict(x_test)
# print('Test data accuracy Score : ' + str(metrics.accuracy_score(y_test, y_pred) * 100))

features = [9176, 1, 178, 68.0, 110, 75, 3, 1, 0, 1, 1, 24.95784]
final = [np.array(features)]
model = pickle.load(open(fileName, 'rb'))
print(model.predict(final))
print(model.predict_proba(final))

