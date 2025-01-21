import pandas as pd

# Załadowanie danych
df = pd.read_csv('winequality-red.csv')

# Sprawdzanie, czy dane zawierają duplikaty
if df.duplicated().any():
    print("Znaleziono duplikaty w danych.")
    # Wyświetlenie liczby duplikatów
    print(f"Liczba duplikatów: {df.duplicated().sum()}")
    
    # Usunięcie duplikatów
    df = df.drop_duplicates()
    print("Duplikaty zostały usunięte.")
else:
    print("Brak duplikatów w danych.")

# Funkcja do usuwania odstających wartości na podstawie IQR
def remove_outliers_iqr(df, column):
    q1 = df[column].quantile(0.25)  # 1. kwartyl
    q3 = df[column].quantile(0.75)  # 3. kwartyl
    iqr = q3 - q1  # rozstęp międzykwartylowy
    lower_bound = q1 - 1.5 * iqr  # dolna granica
    upper_bound = q3 + 1.5 * iqr  # górna granica
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Usuwanie wartości odstających dla wybranych kolumn
columns_to_check = [
    'total sulfur dioxide', 
    'residual sugar', 
    'chlorides', 
    'fixed acidity', 
    'volatile acidity', 
    'citric acid', 
    'sulphates'
]

for column in columns_to_check:
    df = remove_outliers_iqr(df, column)

# Sprawdzenie efektów
print("Podsumowanie danych po usunięciu odstających wartości:")
print(df.describe())

# Wybór tylko kolumn liczbowych, z wyjątkiem 'quality'
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
numeric_columns = [col for col in numeric_columns if col != 'quality']

# Normalizacja Min-Max za pomocą Pandas (bez normalizacji kolumny 'quality')
df[numeric_columns] = df[numeric_columns].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

# Sprawdzenie wyników
print("Podsumowanie danych po normalizacji:")
print(df.describe())

# Ostateczne dane są już w zmiennej df, bez zapisywania do pliku
# Można je teraz używać w dalszych analizach, np. w treningu modelu
df.to_csv('normalized_cleaned_data.csv', index=False)