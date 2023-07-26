
from sklearn.feature_selection import SelectKBest



### Inspired by: https://stackoverflow.com/questions/29412348/selectkbest-based-on-estimated-amount-of-features/29412485
class SelectAtMostKBest(SelectKBest):
    def _check_params(self, X, y):
        if not (self.k == "all" or 0 <= self.k <= X.shape[1]):
            # set k to "all" (skip feature selection), if less than k features are available
            self.k = "all"
