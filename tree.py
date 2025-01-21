import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import time  

# Załadowanie danych
df = pd.read_csv('normalized_cleaned_data.csv')

# Wybór cech (wszystko oprócz 'quality')
X = df.drop('quality', axis=1)
y = df['quality']

# Stratyfikowana walidacja krzyżowa (zachowanie proporcji klas)
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Definiowanie różnych parametrów do testowania
max_depth_values = [3, 4, 5, 6, 7]
min_samples_split_values = [2, 5, 10]
min_samples_leaf_values = [1, 5, 10]
criterion_values = ['gini', 'entropy']

# Lista do przechowywania wyników
results = []

# Iteracja przez wszystkie kombinacje parametrów
for max_depth in max_depth_values:
    for min_samples_split in min_samples_split_values:
        for min_samples_leaf in min_samples_leaf_values:
            for criterion in criterion_values:
                # Model drzewa decyzyjnego z różnymi parametrami
                model = DecisionTreeClassifier(
                    random_state=42,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    criterion=criterion
                )

                # Mierzenie czasu rozpoczęcia walidacji
                start_time = time.time()

                # Przeprowadzenie walidacji krzyżowej
                cv_results = cross_val_score(model, X, y, cv=kf, scoring='accuracy')

                # Mierzenie czasu zakończenia walidacji
                end_time = time.time()

                # Obliczanie czasu trwania walidacji
                elapsed_time = end_time - start_time

                # Wyniki walidacji krzyżowej
                accuracy_validation = cv_results.mean()

                # Dodanie wyników do listy
                results.append({
                    'max_depth': max_depth,
                    'min_samples_split': min_samples_split,
                    'min_samples_leaf': min_samples_leaf,
                    'criterion': criterion,
                    'accuracy_validation': accuracy_validation,
                    'elapsed_time': elapsed_time
                })

# Konwersja wyników na DataFrame
results_df = pd.DataFrame(results)

# Zaokrąglenie wszystkich liczb do 2 miejsc po przecinku w DataFrame
results_df = results_df.round(2)

# Zapisanie wyników do pliku CSV
results_df.to_csv('model_comparison_results1.csv', index=False)

# Wypisanie najlepszych parametrów z najwyższą dokładnością
best_params = results_df.loc[results_df['accuracy_validation'].idxmax()]
print("Najlepsze parametry:")
print(best_params)

# Wyświetlenie tabeli z wynikami
print("\nTabela wyników:")
print(results_df)

# Aby pokazać wyniki graficznie (np. wykres dokładności dla różnych parametrów)
# Wykres zależności dokładności od max_depth
plt.figure(figsize=(10, 6))
for criterion in criterion_values:
    subset = results_df[results_df['criterion'] == criterion]
    plt.plot(subset['max_depth'], subset['accuracy_validation'], marker='o', label=f'Criterion: {criterion}')

plt.title("Dokładność walidacji dla różnych parametrów (max_depth vs accuracy)")
plt.xlabel('max_depth')
plt.ylabel('Średnia dokładność')
plt.legend(title='Criterion')
plt.grid(True)
plt.show()

# Wykres zależności dokładności od min_samples_split
plt.figure(figsize=(10, 6))
for criterion in criterion_values:
    subset = results_df[results_df['criterion'] == criterion]
    plt.plot(subset['min_samples_split'], subset['accuracy_validation'], marker='o', label=f'Criterion: {criterion}')

plt.title("Dokładność walidacji dla różnych parametrów (min_samples_split vs accuracy)")
plt.xlabel('min_samples_split')
plt.ylabel('Średnia dokładność')
plt.legend(title='Criterion')
plt.grid(True)
plt.show()

# Wykres zależności dokładności od min_samples_leaf
plt.figure(figsize=(10, 6))
for criterion in criterion_values:
    subset = results_df[results_df['criterion'] == criterion]
    plt.plot(subset['min_samples_leaf'], subset['accuracy_validation'], marker='o', label=f'Criterion: {criterion}')

plt.title("Dokładność walidacji dla różnych parametrów (min_samples_leaf vs accuracy)")
plt.xlabel('min_samples_leaf')
plt.ylabel('Średnia dokładność')
plt.legend(title='Criterion')
plt.grid(True)
plt.show()
