from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline

X, y = make_classification(
    n_samples=1000, n_features=20, random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="mean")),
        ("classifier", RandomForestClassifier(random_state=42)),
    ]
)

param_grid = {
    "classifier__n_estimators": [50, 100, 200],
    "classifier__max_depth": [None, 10, 20],
}

grid_search = GridSearchCV(
    pipeline, param_grid, cv=5, scoring="accuracy", n_jobs=-1
)
grid_search.fit(X_train, y_train)

y_pred = grid_search.predict(X_test)
print("Best Parameters found:", grid_search.best_params_)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
