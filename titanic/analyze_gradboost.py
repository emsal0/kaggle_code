import numpy as np
import sklearn
import pandas

from sklearn import preprocessing
from sklearn.impute import KNNImputer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier, XGBRFClassifier

trainset = pandas.read_csv("./train.csv")
testset = pandas.read_csv("./test.csv")

def transform(dataset):
    # Note: Still don't understand the semantics of the Parch column, going to assume it's integer
    d = pandas.get_dummies(dataset, columns=['Embarked', 'Pclass', 'SibSp', 'Sex'])
    d = d.drop(columns=['Name', 'Cabin', 'Ticket'])
    return d

trainset = transform(trainset)
testset = transform(testset)

def for_model_input(dataset, test=False):
    d = dataset.drop(columns=['PassengerId'])
    if test:
        y = None
        X = d.values
    else:
        y = np.ravel(d[['Survived']].values)
        X = d.drop(columns=['Survived']).values
    X = preprocessing.scale(X)
    return (X, y)

(Xtrain, ytrain) = for_model_input(trainset)
knn_imputer = KNNImputer()
Xtrain = knn_imputer.fit_transform(Xtrain)

boosted_model = XGBRFClassifier()
boosted_model.fit(Xtrain, ytrain)
boosted_scores = cross_val_score(boosted_model, Xtrain, ytrain, cv=5)

print("Gradient-Boosting Model CV scores:\n", boosted_scores,
        np.mean(boosted_scores))

(Xtest, _) = for_model_input(testset, test=True)
Xtest = knn_imputer.fit_transform(Xtest)
predictions_boosted = boosted_model.predict(Xtest) # + 1) / 2
predictions_boosted = predictions_boosted.astype('int64')
pred_boosted_df = pandas.DataFrame(
                        predictions_boosted, columns=['Survived'])
fin_ans_boosted = pandas.DataFrame(testset['PassengerId']).join(
                        pred_boosted_df)
with open('predictions_xgboost_rf.csv', 'w') as f:
    f.write((fin_ans_boosted.to_csv(index=False)))
