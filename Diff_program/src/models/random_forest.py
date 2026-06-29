from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV


class RandomForestModel:
    def __init__(self, param_grid=None, random_state=42):
        self.random_state = random_state
        self.param_grid = param_grid or {
            'n_estimators': [10, 50, 100, 200, 300],
            'max_depth': [10, 20, 30, None]
        }
        self.model = None
        self.best_params = None

    def fit(self, X, y, grid_search=True):
        if grid_search:
            print('Grid Search for optimal parameters...')
            grid = GridSearchCV(
                estimator=RandomForestRegressor(random_state=self.random_state),
                param_grid=self.param_grid,
                cv=5,
                scoring='r2',
                n_jobs=-1,
                verbose=1
            )
            grid.fit(X, y)
            self.best_params = grid.best_params_
            self.model = grid.best_estimator_
            print(f"Best params: {self.best_params}")
            print(f"Best CV R2: {grid.best_score_:.4f}")
        else:
            self.model = RandomForestRegressor(random_state=self.random_state)
            self.model.fit(X, y)
        return self.model

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        return self.model.score(X, y)