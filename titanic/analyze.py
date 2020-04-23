import numpy as np
import sklearn
import pandas
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegressionCV
from sklearn.svm import SVC
from sklearn.impute import KNNImputer

trainset = pandas.read_csv("./train.csv")
testset = pandas.read_csv("./test.csv")

def transform(dataset):
    # Note: Still don't understand the semantics of the Parch column, going to assume it's integer
    d = pandas.get_dummies(dataset, columns=['Embarked', 'Pclass', 'SibSp', 'Sex'])
    d = d.drop(columns=['Cabin', 'Name', 'Ticket'])
    return d

def for_model_input(dataset, test=False):
    d = dataset.drop(columns=['PassengerId'])
    if test:
        y = None
        X = d.values
    else:
        y = np.ravel(d[['Survived']].values) # * 2 - 1
        X = d.drop(columns=['Survived']).values
    X = preprocessing.scale(X)
    return (X, y)

trainset = transform(trainset)
testset = transform(testset)

print(trainset.columns)
print(testset.columns)

(Xtrain, ytrain) = for_model_input(trainset)
ytrain_svc = ytrain * 2 - 1

knn_imputer = KNNImputer()
Xtrain = knn_imputer.fit_transform(Xtrain)

svc_model = SVC()
svc_model.fit(Xtrain, ytrain_svc)
svc_scores = cross_val_score(svc_model, Xtrain, ytrain_svc, cv=5)

logit_model = LogisticRegressionCV(fit_intercept=True)
logit_model.fit(Xtrain, ytrain)
logit_scores = cross_val_score(logit_model, Xtrain, ytrain, cv=5)

selector_logit = RFE(logit_model)
selector_logit.fit(Xtrain, ytrain)
selector_logit_scores = cross_val_score(selector_logit, Xtrain, ytrain_svc, cv=5)

print("SVC CV scores:\n", svc_scores, np.mean(svc_scores))
print("Logit CV scores:\n", logit_scores, np.mean(logit_scores))
print("logit (feature-selected) CV scores:\n", selector_logit_scores,
        np.mean(selector_logit_scores))

(Xtest, _) = for_model_input(testset, test=True)
Xtest = knn_imputer.fit_transform(Xtest)

predictions_svc = (svc_model.predict(Xtest) + 1)/2
predictions_svc = predictions_svc.astype('int64')

predictions_logit = logit_model.predict(Xtest) # + 1) / 2
predictions_logit = predictions_svc.astype('int64')

predictions_logit_rfe = selector_logit.predict(Xtest) # + 1) / 2
predictions_logit_rfe = predictions_logit_rfe.astype('int64')

pred_svc_df = pandas.DataFrame(predictions_svc, columns=['Survived'])
fin_ans_svc = pandas.DataFrame(testset['PassengerId']).join(pred_svc_df)
with open('predictions_svc.csv', 'w') as f:
    f.write((fin_ans_svc.to_csv(index=False)))

pred_logit_df = pandas.DataFrame(predictions_logit, columns=['Survived'])
fin_ans_logit = pandas.DataFrame(testset['PassengerId']).join(pred_logit_df)
with open('predictions_logit.csv', 'w') as f:
    f.write((fin_ans_logit.to_csv(index=False)))

pred_logit_rfe_df = pandas.DataFrame(
                        predictions_logit_rfe, columns=['Survived'])
fin_ans_logit_rfe = pandas.DataFrame(testset['PassengerId']).join(
                        pred_logit_rfe_df)
with open('predictions_logit_rfe.csv', 'w') as f:
    f.write((fin_ans_logit_rfe.to_csv(index=False)))
