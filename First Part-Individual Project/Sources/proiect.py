# Costache Ovidiu-Stefan 313CC
# Partea I
# Problema de clasificare - Diabet

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Citesc datele
df = pd.read_csv("diabetes_prediction_dataset.csv")

print("Antetul este:")
print(df.head())
print()

print("Informatii despre dataframe:")
print(df.info())
print()

print("Dimensiunile sunt:", df.shape)
print()

print("Tipurile datelor folosite:")
print(df.dtypes)
print()

print("Ce valori se pot afisa pentru fiecare coloana in parte:")
for col in df.columns:
    print(f"{col}: {df[col].unique()[:5]}")
print()

# EDA

# Analiza valorilor lipsa
print("Analiza valorilor lipsa:")
print()

print("Numarul valorilor lipsa pe coloana:")
print(df.isnull().sum())
print()

print("Procentul valorilor lipsa pe coloana:")
print((df.isnull().mean() * 100).round(2))
print()

print("Tratarea valorilor lipsa pentru BMI:")
if df["bmi"].isnull().sum() > 0:
    medie_bmi = df["bmi"].mean()
    df["bmi"].fillna(medie_bmi, inplace = True)
    print("Valorile lipsa sunt inlocuite cu media ", medie_bmi)
else:
    print("Nu exista valori lipsa pentru BMI")
print()

# Statistici descriptive
print("Statistici descriptive:")
print(df.describe(include = "all"))
print()

# Analiza distributiei variabilelor
# Histograme pentru caracteristici numerice
caract_num = df.select_dtypes(include = [np.number]).columns.tolist()
for col in caract_num:
    plt.figure()
    df[col].hist()
    plt.title(f"Histograma pentru {col}")
    plt.xlabel(col)
    plt.savefig(f"histograma_{col}.png")
    plt.close()

# Countplot pentru variabile categorice
var_categ = df.select_dtypes(include = ["object"]).columns.tolist()
for col in var_categ:
    plt.figure()
    sns.countplot(x=col, data=df)
    plt.title(f"Countplot {col}")
    # Ca sa salvez in fisiere diferite fiecare grafic
    plt.savefig(f"countplot_{col}.png")
    plt.close()

# Detectarea outlierilor
caract_num = df.select_dtypes(include=[np.number]).columns.tolist()
for col in caract_num:
    plt.figure()
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot pentru {col}")
    plt.xlabel(col)
    plt.savefig(f"boxplot_{col}.png")
    plt.close()

# Analiza corelatiilor
print("Matricea de corelatii:")
print(df[caract_num].corr())
plt.figure()
sns.heatmap(df[caract_num].corr(), annot=True)
plt.title("Heatmap")
plt.savefig("corelatie_numerica.png")
plt.close()

# Analiza relatiilor cu variabila tinta
caract_num = df.select_dtypes(include = [np.number]).columns.tolist()
for col in caract_num:
    if col != "diabetes":
        plt.figure()
        sns.violinplot(x = df["diabetes"], y = df[col])
        plt.title(f"Violin plot {col} vs diabetes")
        plt.xlabel("diabetes")
        plt.ylabel(col)
        plt.savefig(f"violin_{col}_vs_diabetes.png")
        plt.close()
print()

# Codificare pentru datele de tip obiect
cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
print("Coloanele categorice pentru codificare:", cat_cols)
for col in cat_cols:
    df[col] = LabelEncoder().fit_transform(df[col])

print("Cum arata datele dupa codificare:")
print(df.head())
print()

# Split in train.csv si test.csv
X = df.drop("diabetes", axis = 1)
y = df["diabetes"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
print("Dimensiunea setului train ", X_train.shape)
print("Dimensiunea setului test ", X_test.shape)
print()

# Salvarea seturilor de train si test in fisiere .csv in afara directorului Surse
train_path = "../train.csv"
test_path = "../test.csv"

# Combin datele de antrenament
train_data = pd.concat([X_train, y_train], axis = 1)
# Combin datele de test
test_data = pd.concat([X_test, y_test], axis = 1)

# Salvez datele de antrenament fara indexul de la Pandas
train_data.to_csv(train_path, index = False)
# Salvez datele de test fara indexul de la Pandas
test_data.to_csv(test_path, index = False)

# Aici am standardizat datele numerice si am calculat media si deviatia std
# M-am folosit de codul din Laboratorul 5
num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

# Antrenez si evaluez modelul

# Creez modelul
model = LogisticRegression(max_iter = 10000)
model.fit(X_train, y_train)

# Predictiile pe setul de test
y_pred = model.predict(X_test)

# Metrice de acuratete
acuratete = accuracy_score(y_test, y_pred)
print("Acuratetea pe setul de test este ", acuratete)
print()
print("Raportul de clasificare:")
print(classification_report(y_test, y_pred))

# Matricea de confuzie
conf_mat = confusion_matrix(y_test, y_pred)
plt.figure()
sns.heatmap(conf_mat, annot = True, fmt = "d")
plt.xlabel("Etichete prezise")
plt.ylabel("Etichete reale")
plt.title("Matrice de confuzie")
plt.savefig("matrice_confuzie.png")
plt.close()

