import pandas as pd
import numpy as np
import time
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.svm import SVC
import matplotlib.pyplot as plt

# Załadowanie danych
df = pd.read_csv('normalized_cleaned_data.csv')

# Wybór cech (wszystko oprócz 'quality')
X = df.drop('quality', axis=1)
y = df['quality']

# Stratyfikowana walidacja krzyżowa
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Parametry do sprawdzenia
param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'gamma': [0.01, 0.1, 1]
}

results = []

# Przechodzenie przez wszystkie kombinacje parametrów
for C in param_grid['C']:
    for kernel in param_grid['kernel']:
        for gamma in param_grid['gamma']:
            # Inicjalizacja modelu
            model = SVC(C=C, kernel=kernel, gamma=gamma, random_state=42)
            
            # Mierzenie czasu rozpoczęcia
            start_time = time.time()
            
            # Walidacja krzyżowa
            cv_results = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
            
            # Mierzenie czasu zakończenia
            elapsed_time = time.time() - start_time
            
            # Zbieranie wyników
            results.append({
                'C': C,
                'Kernel': kernel,
                'Gamma': gamma,
                'Mean Accuracy': np.round(cv_results.mean(), 2),
                'Execution Time': np.round(elapsed_time, 2)
            })

# Konwersja wyników do DataFrame
results_df = pd.DataFrame(results)

# Zapis do pliku CSV
results_df.to_csv('svm_results.csv', index=False)

# Wykresy zależności średniej dokładności od parametrów
plt.figure(figsize=(12, 8))

# Zależność od C
for kernel in param_grid['kernel']:
    subset = results_df[results_df['Kernel'] == kernel]
    plt.plot(subset['C'], subset['Mean Accuracy'], marker='o', label=f'Kernel: {kernel}')

plt.title("Średnia dokładność w zależności od parametru C")
plt.xlabel("C")
plt.ylabel("Średnia dokładność")
plt.xscale('log')
plt.legend()
plt.grid(True)
plt.show()

# Zależność od Gamma
plt.figure(figsize=(12, 8))
for kernel in param_grid['kernel']:
    subset = results_df[results_df['Kernel'] == kernel]
    plt.plot(subset['Gamma'], subset['Mean Accuracy'], marker='o', label=f'Kernel: {kernel}')

plt.title("Średnia dokładność w zależności od parametru Gamma")
plt.xlabel("Gamma")
plt.ylabel("Średnia dokładność")
plt.xscale('log')
plt.legend()
plt.grid(True)
plt.show()

# Zależność od Kernel
kernel_avg = results_df.groupby('Kernel')['Mean Accuracy'].mean()
plt.figure(figsize=(8, 6))
kernel_avg.plot(kind='bar', color='skyblue')
plt.title("Średnia dokładność w zależności od Kernel")
plt.xlabel("Kernel")
plt.ylabel("Średnia dokładność")
plt.grid(axis='y')
plt.show()

print("Zapisano wyniki do pliku 'svm_results.csv'.")
