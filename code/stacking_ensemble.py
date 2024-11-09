from sklearn.ensemble import StackingClassifier
from xgboost import XGBClassifier

class EnsembleStacker:
    def __init__(self, estimators, X_train, y_train, X_test, y_test):
        self.stacking_clf = StackingClassifier(
            estimators=estimators,
            final_estimator=XGBClassifier(),
            cv=5
        )
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def fit(self):
        self.stacking_clf.fit(self.X_train, self.y_train)

    def evaluate(self):
        y_pred = self.stacking_clf.predict(self.X_test)
        y_pred_proba = self.stacking_clf.predict_proba(self.X_test)[:, 1]
        return {
            'Accuracy': accuracy_score(self.y_test, y_pred),
            'Precision': precision_score(self.y_test, y_pred),
            'Recall': recall_score(self.y_test, y_pred),
            'F1 Score': f1_score(self.y_test, y_pred),
            'ROC AUC': roc_auc_score(self.y_test, y_pred_proba),
            'FPR': confusion_matrix(self.y_test, y_pred)[0][1] / sum(confusion_matrix(self.y_test, y_pred)[0])
        }
