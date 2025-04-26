import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import make_scorer, balanced_accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.exceptions import ConvergenceWarning
import warnings
import random

random.seed(42)
np.random.seed(42)

warnings.filterwarnings("ignore", category=ConvergenceWarning)

df = pd.read_csv("data.csv")

df.dropna(axis=1, how='all', inplace=True)
df.dropna(axis=0, how='any', inplace=True)

df.drop(columns=["id"], inplace=True)
df['diagnosis'] = df['diagnosis'].map({'M': 0, 'B': 1})

print("Wartości NaN przed skalowaniem:")
print(df.isna().sum())

scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(df.iloc[:, 1:])

print("Wartości NaN po skalowaniu:")
print(pd.DataFrame(scaled_features).isna().sum())

imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(scaled_features)

print("Wartości NaN po imputacji:")
print(pd.DataFrame(X).isna().sum())

y = df['diagnosis'].to_numpy()

def create_mlp(hidden_layers, activation, solver):
    return MLPClassifier(hidden_layer_sizes=hidden_layers, activation=activation, solver=solver, random_state=42, max_iter=1000, tol=1e-4)

def cross_val(hidden_layers, activation, solver):
    clf = create_mlp(hidden_layers, activation, solver)
    return cross_val_score(clf, X, y, cv=10, scoring=make_scorer(balanced_accuracy_score)).mean()

hidden_layer_configs = [(30,), (30, 30), (60,), (60, 60), (100,), (50, 100), (100, 50)]
results_a = []

for config in hidden_layer_configs:
    score = cross_val(config, 'relu', 'adam')
    results_a.append(score)

activations = ['identity', 'logistic', 'tanh', 'relu']
best_hidden_layer = (60,)
results_b = []

for activation in activations:
    score = cross_val(best_hidden_layer, activation, 'adam')
    results_b.append(score)

solvers = ['lbfgs', 'sgd', 'adam']
results_c = []

for solver in solvers:
    score = cross_val(best_hidden_layer, 'relu', solver)
    results_c.append(score)

plt.figure(figsize=(18, 5))

plt.subplot(131)
plt.bar(range(len(hidden_layer_configs)), results_a, tick_label=[str(config) for config in hidden_layer_configs], color='skyblue')
plt.title("Wpływ ilości neuronów w warstwie ukrytej")
plt.xlabel("Warstwa ukryta")
plt.ylabel("Średnia balanced_accuracy")

plt.subplot(132)
plt.bar(range(len(activations)), results_b, tick_label=activations, color='lightgreen')
plt.title("Funkcja aktywacja")
plt.xlabel("Funkcja aktywacja")
plt.ylabel("Średnia balanced_accuracy")

plt.subplot(133)
plt.bar(range(len(solvers)), results_c, tick_label=solvers, color='salmon')
plt.title("Solver")
plt.xlabel("Solver")
plt.ylabel("Średnia balanced_accuracy")

plt.tight_layout()
plt.show()