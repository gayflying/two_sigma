
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn import preprocessing
import pandas as pd


class manager_skill(BaseEstimator, TransformerMixin):
    """
    Adds the column "manager_skill" to the dataset, based on the Kaggle kernel
    "Improve Perfomances using Manager features" by den3b. The function should
    be usable in scikit-learn pipelines.

    Parameters
    ----------
    threshold : Minimum count of rental listings a manager must have in order
                to get his "own" score, otherwise the mean is assigned.

    Attributes
    ----------
    mapping : pandas dataframe
        contains the manager_skill per manager id.

    mean_skill : float
        The mean skill of managers with at least as many listings as the
        threshold.
    """

    def __init__(self, threshold=5):
        self.threshold = threshold

    def _reset(self):
        """Reset internal data-dependent state of the scaler, if necessary.

        __init__ parameters are not touched.
        """
        # Checking one attribute is enough, becase they are all set together
        # in fit
        if hasattr(self, 'mapping_'):
            self.mapping_ = {}
            self.mean_skill_ = 0.0

    def fit(self, X, y):
        """Compute the skill values per manager for later use.

        Parameters
        ----------
        X : pandas dataframe, shape [n_samples, n_features]
            The rental data. It has to contain a column named "manager_id".

        y : pandas series or numpy array, shape [n_samples]
            The corresponding target values with encoding:
            low: 0.0
            medium: 1.0
            high: 2.0
        """
        self._reset()

        temp = pd.concat([X.manager_id, pd.get_dummies(y)], axis=1).groupby('manager_id').mean()
        temp.columns = ['low_frac', 'medium_frac', 'high_frac']
        temp['count'] = X.groupby('manager_id').count().iloc[:, 1]

        print(temp.head())

        temp['manager_skill'] = temp['high_frac'] * 2 + temp['medium_frac']

        mean = temp.loc[temp['count'] >= self.threshold, 'manager_skill'].mean()

        temp.loc[temp['count'] < self.threshold, 'manager_skill'] = mean

        self.mapping_ = temp[['manager_skill']]
        self.mean_skill_ = mean

        return self

    def transform(self, X):
        """Add manager skill to a new matrix.

        Parameters
        ----------
        X : pandas dataframe, shape [n_samples, n_features]
            Input data, has to contain "manager_id".
        """
        X = pd.merge(left=X, right=self.mapping_, how='left', left_on='manager_id', right_index=True)
        X['manager_skill'].fillna(self.mean_skill_, inplace=True)

        return X

