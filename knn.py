import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import time

# Załadowanie danych
df = pd.read_csv('normalized_cleaned_data.csv')

# Wybór cech (wszystko oprócz 'quality')
X = df.drop('quality', axis=1)
y = df['quality']

# Stratyfikowana walidacja krzyżowa (zachowanie proporcji klas)
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Definiowanie zakresów parametrów do testowania
n_neighbors_values = [3, 5, 7, 9, 11]  # różne liczby sąsiadów
weights_values = ['uniform', 'distance']  # różne wagi (jednakowe lub odwrotność odległości)
leaf_size_values = [20, 30, 40, 50]  # różne rozmiary liści

# Lista do przechowywania wyników
results = []

# Iteracja przez wszystkie kombinacje parametrów
for n_neighbors in n_neighbors_values:
    for weights in weights_values:
        for leaf_size in leaf_size_values:
            # Tworzenie modelu KNN z określonymi parametrami
            model = KNeighborsClassifier(
                n_neighbors=n_neighbors,
                weights=weights,
                leaf_size=leaf_size
            )

            # Mierzenie czasu rozpoczęcia walidacji
            start_time = time.time()

            # Przeprowadzenie walidacji krzyżowej
            cv_results = cross_val_score(model, X, y, cv=kf, scoring='accuracy')

            # Mierzenie czasu zakończenia walidacji
            end_time = time.time()

            # Obliczanie czasu trwania walidacji
            elapsed_time = end_time - start_time

            # Obliczanie średniej dokładności
            accuracy_validation = cv_results.mean()

            # Dodanie wyników do listy
            results.append({
                'n_neighbors': n_neighbors,
                'weights': weights,
                'leaf_size': leaf_size,
                'accuracy_validation': accuracy_validation,
                'elapsed_time': elapsed_time
            })

# Konwersja wyników na DataFrame
results_df = pd.DataFrame(results)

# Zaokrąglenie wszystkich liczb do 2 miejsc po przecinku w DataFrame
results_df = results_df.round(2)

# Zapisanie wyników do pliku CSV
results_df.to_csv('knn_model_comparison_results.csv', index=False)

# Wypisanie najlepszych parametrów z najwyższą dokładnością
best_params = results_df.loc[results_df['accuracy_validation'].idxmax()]
print("Najlepsze parametry:")
print(best_params)

# Wyświetlenie tabeli z wynikami
print("\nTabela wyników:")
print(results_df)

# Aby pokazać wyniki graficznie, np. wykres zależności dokładności w różnych parametrach
import matplotlib.pyplot as plt

# Wykres zależności dokładności od liczby sąsiadów
plt.figure(figsize=(10, 6))
for weights in weights_values:
    subset = results_df[results_df['weights'] == weights]
    plt.plot(subset['n_neighbors'], subset['accuracy_validation'], marker='o', label=f'Weights: {weights}')

plt.title("Dokładność w zależności od liczby sąsiadów i wag")
plt.xlabel('n_neighbors')
plt.ylabel('Średnia dokładność')
plt.legend(title='Weights')
plt.grid(True)
plt.show()

# Wykres zależności dokładności od leaf_size
plt.figure(figsize=(10, 6))
for weights in weights_values:
    subset = results_df[results_df['weights'] == weights]
    plt.plot(subset['leaf_size'], subset['accuracy_validation'], marker='o', label=f'Weights: {weights}')

plt.title("Dokładność w zależności od rozmiaru liścia i wag")
plt.xlabel('leaf_size')
plt.ylabel('Średnia dokładność')
plt.legend(title='Weights')
plt.grid(True)
plt.show()
