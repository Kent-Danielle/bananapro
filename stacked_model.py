import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier


class StackedModel:
    def __init__(self):
        self.models = list()
        self.initialize_models()
        self.predictions_df = None
        self.base_model = RidgeClassifier(alpha=1.0, class_weight=None, copy_X=True, fit_intercept=True,
                                          max_iter=None, positive=False, random_state=123, solver='auto',
                                          tol=0.0001)

    def __getstate__(self):
        return self.models, self.predictions_df, self.base_model

    def __setstate__(self, state):
        self.models, self.predictions_df, self.base_model = state

    def initialize_models(self):
        et = ExtraTreesClassifier(bootstrap=False, ccp_alpha=0.0, class_weight=None,
                                  criterion='gini', max_depth=None, max_features='sqrt',
                                  max_leaf_nodes=None, max_samples=None,
                                  min_impurity_decrease=0.0, min_samples_leaf=1,
                                  min_samples_split=2, min_weight_fraction_leaf=0.0,
                                  monotonic_cst=None, n_estimators=100, n_jobs=-1,
                                  oob_score=False, random_state=123, verbose=0,
                                  warm_start=False)
        self.models.append(et)
        knn = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                                   metric_params=None, n_jobs=-1, n_neighbors=9, p=2,
                                   weights='distance')
        self.models.append(knn)
        lightgbm = LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,
                                  importance_type='split', learning_rate=0.1, max_depth=-1,
                                  min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,
                                  n_estimators=100, n_jobs=-1, num_leaves=31, objective=None,
                                  random_state=123, reg_alpha=0.0, reg_lambda=0.0, subsample=1.0,
                                  subsample_for_bin=200000, subsample_freq=0)
        self.models.append(lightgbm)

    def fit(self, x_train, x_test, y_train, y_test):
        df_predictions = pd.DataFrame()
        for i, model in enumerate(self.models):
            model.fit(x_train, y_train)
            predictions = self.models[i].predict(x_test)
            col_name = str(model.__class__.__name__)
            df_predictions[col_name] = predictions
        self.predictions_df = df_predictions
        # df_predictions.to_csv(FILE_DIR + 'stacked_model_predictions.csv', index=False)

        self.base_model.fit(df_predictions, y_test)
        # pkl.dump(self.base_model, open(BIN_DIR + 'stacked_model_pkl', 'wb'))

    def predict(self, x):
        df_predictions = pd.DataFrame()
        for i, model in enumerate(self.models):
            predictions = self.models[i].predict(x)
            col_name = str(model.__class__.__name__)
            df_predictions[col_name] = predictions

        return self.base_model.predict(df_predictions)
