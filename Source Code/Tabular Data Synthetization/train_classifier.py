from sklearn.ensemble import RandomForestClassifier
from dataset import Adult
import pickle
import numpy as np

d = Adult()
data, labels = d.to_svm(d.dataset)

rf = RandomForestClassifier(
    n_estimators=100, criterion='gini', max_depth=None, min_samples_split=2,
    min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto',
    max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None,
    bootstrap=True, oob_score=False, n_jobs=None, random_state=None,
    verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None
)
rf.fit(data, labels)

pickle.dump(rf, open('./rf.pkl', 'wb'))

