# Proiect PCLP

## Mihai Nan∗, Andrei-Daniel Voicu, George Alexandru Tudor

## Departamentul de Calculatoare

## 30 aprilie 2025

## Notă!

Proiectul poate fi realizat individual sau în echipă de 2 student,i. În cazul în care aleget,i să realizat,i
proiectul individual, atunci vet,i realiza doar prima parte. Dacă aleget,i să realizat,i proiectul în
echipă, atunci fiecare student îs,i va alege o parte.

## Cuprins

1 Introducere 1

2 Partea I – Construirea s,i explorarea unui dataset tabelar 1
2.1 Cerint,e detaliate.................................... 1

3 Partea a II-a – Prelucrarea avansată a datelor s,i compararea modelelor 2
3.1 Cerint,e detaliate.................................... 3

4 Bonus – Interfat,ă grafică 4

5 Punctaj 5
∗mihai.nan@upb.ro


## 1 Introducere

Python a devenit indispensabil în domeniul Data Science s,i Inteligent,a Artificială, mai ales în
explorarea datelor s,i dezvoltarea de modele, datorită ecosistemului său bogat de biblioteci spe-
cializate. Cu biblioteci precumnumpy,pandas,matplotlib,seaborn, Python oferă instrumente
puternice pentru manipularea, vizualizarea s,i analiza datelor statistice. Aceste biblioteci permit
cercetătorilor s,i analis,tilor să exploreze seturile de date, să identifice modele s,i tendint,e, s,i să
obt,ină înt,elegeri profunde din datele brute.

## 2 Partea I – Construirea s,i explorarea unui dataset tabelar

În prima parte a proiectului, vet,i construi un set de date tabelar propriu, folosind una dintre
următoarele metode:

1. Extragerea de informat,ii de pe internet prin tehnici de web scraping sau utilizarea de
    API-uri publice (ex: OpenWeatherMap, SpaceX Launch Data etc.);
2. Generarea sintetică a unui dataset care să aibă sens contextual (ex: simularea datelor medi-
    cale despre pacient,i, date financiare despre companii, date despre specii de plante/animale);
3. Pornirea de la un dataset simplu de pe platforme precum Kaggle s,i extinderea acestuia prin
    adăugarea de noi coloane calculate, simularea de date lipsă, generarea de etichete pentru
    clasificare sau introducerea de zgomot.

### 2.1 Cerint,e detaliate

1. Tipul problemei:setul de date propus trebuie să fie destinat fie unei probleme de regresie,
    fie unei probleme de clasificare. Tipul problemei trebuie specificat clar în documentat,ie.
2. Structura setului de date:Setul de date final trebuie să fie împărt,it în două subseturi:
    - Subset de antrenare: cel put,in 500 de instant,e (rânduri);
    - Subset de testare: cel put,in 200 de instant,e (rânduri).
    Împărt,irea poate fi realizată prin extragere randomizată (ex: folosind funct,ii dinscikit-learn
    saupandas).
3. Numărul minim de caracteristici:fiecare instant,ă trebuie să aibă minimum 8 coloane
    relevante, inclusiv coloana t,intă. Pentru aceste coloane vet,i selecta cel put,in 3 tipuri diferite
    de date (spre exemplu: numere întregi, numere reale, valori categoriale, s,iruri de caractere
    etc.).
4. Salvare dataseturilor: Exportul subsetului de antrenare s,i al subsetului de testare în
    fis,iere CSV separate.
5. Documentare:Explicarea clară a modului de construct,ie a setului de date: surse, metode
    de generare, eventuale ipoteze sau procesări suplimentare.


6. Analiza exploratorie a datelor (EDA complex): Se va realiza o explorare detaliată
    pentru fiecare dintre cele două subseturi propuse (cel de antrenare s,i cel de testare) care
    să includă obligatoriu:
       a)Analiza valorilor lipsă: număr s,i procent de valori lipsă pe coloană; strategii de
          tratare a acestora (ex: imputare, s,tergere).
b) Statistici descriptive:utilizareadescribe()s,i interpretarea principalelor statistici
pentru variabile numerice s,i categorice.
c)Analiza distribut,iei variabilelor:
- Histogramă pentru fiecare caracteristică numerică;
- Grafice de tip countplot/barplot pentru variabilele categorice.
d) Detectarea outlierilor: Utilizarea boxplot-urilor sau altor tehnici (ex: IQR rule)
pentru identificarea valoriloraberante.
e)Analiza corelat,iilor:Matrice de corelat,ii (heatmap) pentru variabilele numerice.
f) Analiza relat,iilor cu variabila t,intă: Scatter plots sau violin plots pentru relat,ia
dintre caracteristici s,i variabila t,intă (în funct,ie de tipul problemei).
       g)Comentarii s,i interpretări personale: Fiecare grafic trebuie însot,it de o scurtă
          interpretare textuală care să răspundă la următoarele întrebări:
             - Ce observăm?
             - Ce suspiciuni/idei putem formula?
             - Ce preprocesări ar trebui să aplicăm?
7. Antrenarea s,i evaluarea unui model de bază: Se va antrena un model simplu din
    bibliotecascikit-learn, potrivit pentru problema aleasă:
       - Exemplu: regresie liniară pentru probleme de regresie, logistic regression sau random
          forest pentru probleme de clasificare.
       - Modelul va fi antrenat pe subsetul de antrenare s,i evaluat pe subsetul de testare.
       - Se vor raporta s,i interpreta rezultatele utilizând metrici adecvate:
          a) Pentru regresie: RMSE, MAE sauR^2.
          b) Pentru clasificare: acuratet,e, precizie, recall, F1-score.
       - Se vor include grafice relevante pentru performant,a modelului (ex: matrice de confu-
          zie, grafice de erori etc.).

## 3 Partea a II-a – Prelucrarea avansată a datelor s,i compararea

## modelelor

În această parte a proiectului, se va continua lucrul pornind de la datasetul realizat de colegul
de echipă la Partea I (2).


### 3.1 Cerint,e detaliate

1. Prelucrarea datelor:Se vor aplica tehnici de prelucrare a datelor, după caz:
    - Normalizare sau standardizare pentru variabilele numerice;
    - Encodare pentru variabilele categorice (ex:OneHotEncoder,LabelEncoder);
    - Înlocuirea valorilor lipsă prin metode adecvate (ex: medie, mediană, modă, imputare
       avansată).
Se va explica pentru fiecare tehnică aleasă de ce a fost necesară s,i ce impact are asupra
modelelor de machine learning (pe baza rezultatelor obt,inute pentru rularea modelului cu
sau fără aplicarea tehnicii).
2. Analiza exploratorie a datelor (EDA complex) după aplicarea prelucrărilor:Se
    va realiza o explorare detaliată pentru fiecare dintre cele două subseturi propuse (cel de
    antrenare s,i cel de testare) care să includă obligatoriu:
       a)Analiza valorilor lipsă: număr s,i procent de valori lipsă pe coloană; strategii de
          tratare a acestora (ex: imputare, s,tergere).
b) Statistici descriptive:utilizareadescribe()s,i interpretarea principalelor statistici
pentru variabile numerice s,i categorice.
c)Analiza distribut,iei variabilelor:
- Histogramă pentru fiecare caracteristică numerică;
- Grafice de tip countplot/barplot pentru variabilele categorice.
d) Detectarea outlierilor: Utilizarea boxplot-urilor sau altor tehnici (ex: IQR rule)
pentru identificarea valoriloraberante.
e)Analiza corelat,iilor:Matrice de corelat,ii (heatmap) pentru variabilele numerice.
f) Analiza relat,iilor cu variabila t,intă: Scatter plots sau violin plots pentru relat,ia
dintre caracteristici s,i variabila t,intă (în funct,ie de tipul problemei).
       g)Comentarii s,i interpretări personale: Fiecare grafic trebuie însot,it de o scurtă
          interpretare textuală care să răspundă la următoarele întrebări:
             - Ce observăm?
             - Ce suspiciuni/idei putem formula?
             - Ce preprocesări ar trebui să aplicăm?
    Putet,i colabora cu colegul de echipă s,i să preluat,i codul realizat de el pe care să-l adaptat,i
    pentru a putea fi aplicat pe seturile de date procesate. Identificat,i s,i documentat,i eventu-
    alele limitări / puncte tari ale dataset-ului produs de colegul de echipă.
3. Antrenarea s,i compararea a cel put,in 3 algoritmi diferit,i:Se vor alege minimum
    3 modele diferite din bibliotecascikit-learn, potrivite tipului de problemă (regresie sau
    clasificare). Exemple:
       - Pentru regresie: Linear Regression, Ridge Regression, Decision Tree Regressor, Ran-
          dom Forest Regressor, SVR etc.


- Pentru clasificare: Logistic Regression, Decision Tree Classifier, Random Forest Cla-
    ssifier, SVM, KNN etc.
Fiecare model va fi antrenat pe datele prelucrate (subsetul de antrenare) s,i evaluat pe
subsetul de testare.
4. Evaluarea performant,ei:Performant,ele modelelor se vor compara utilizând aceeas,i me-
trică (aleasă de voi) pentru toate modelele:
- Regresie: RMSE, MAE,R^2 etc.
- Clasificare: acuratet,e, F1-score, ROC AUC etc.
Se va construi un tabel comparativ care să includă: numele algoritmului, valorile obt,inute
pentru fiecare metrică relevantă.
În plus, pentru o analiză mai detaliată:
- În cazul problemelor de clasificare, vet,i reprezenta grafic matricea de confuzie pentru
fiecare model s,i, dacă este relevant, curbele ROC.
- În cazul problemelor de regresie, vet,i include diagrame de tipscatter plot (valoare
reală vs. valoare prezisă) s,i/sau distribut,ia reziduurilor.
Aceste vizualizări trebuie să ajute la interpretarea performant,ei s,i la identificarea eventu-
alelor puncte slabe ale fiecărui model.

## 4 Bonus – Interfat,ă grafică

Indiferent de ce parte aleget,i să rezolvat,i, se poate dezvolta o interfat,ă grafică pentru acest
proiect, care să permită utilizatorului să introducă valori pentru variabilele de intrare s,i să
vizualizeze predict,iile s,i performant,ele modelelor de învăt,are automată. Iată câteva sugestii
pentru implementarea acesteia:

- Input pentru date:
    - Crează câmpuri de introducere a valorilor pentru variabilele numerice s,i categorice
       (de exemplu, câmpuri de text pentru valori numerice s,i dropdown pentru variabilele
       categorice).
    - Include un buton de "Predict,ie" care să preia datele introduse s,i să le paseze unui
       model antrenat.
- Predict,ie s,i vizualizare:
    - După ce utilizatorul apasă butonul de predict,ie, aplicat,ia poate rula modelele selectate
       (de exemplu, regresie sau clasificare) s,i să afis,eze predict,iile pentru fiecare algoritm.
    - În cazul clasificării, ar putea fi afis,ate probabilităt,ile fiecărei clase.
    - În cazul regresiei, ar putea fi afis,ată valoarea prezisă s,i o comparat,ie cu valoarea reală,
       dacă există.
Vizualizări grafice:


- Clasificare: Afis,ează matricea de confuzie pentru fiecare model, utilizând o funct,ie de
    plotare. Dacă este cazul, adaugă curbele ROC pentru fiecare model.
- Regresie: Creează un scatter plot cu valorile reale fat,ă de valorile prezise s,i/sau
    distribut,ia reziduurilor pentru fiecare model.

## 5 Punctaj

Proiectul va fi încărcat pe Moodle, de fiecare membru al echipei (fiecare student îs,i încarcă partea
lui), sub forma unei arhive.zipcu următorul cont,inut:

- un director cu numeleParteaI(pentru cei care au ales Partea I– 2) ce cont,ine
    următoarele subdirectoare:
       - Surse- toate fis,ierele cu cod folosite în realizarea temei (.py/.ipynb)
       - README.pdf - fis,ierul care cont,ine toate histogramele rezultate, toate graficele
          create s,i documentat,ia.
       - train.csvs,itest.csv- datele care compun cele două subseturi ale setului de date
          realizat
    În fis,ierul fis,ierul README vet,i descrie modul de rezolvare pentru fiecare cerint,ă din
    Partea I, răspunsurile la întrebările din cerint,ă s,i alte observat,ii; prima linie a fis,ierului va
    cont,ine numele complet, seria s,i grupa studentului care a rezolvat partea I.
- un director cu numeleParteaII(pentru cei care au ales Partea a II-a– 3) ce cont,ine
    următoarele subdirectoare:
       - Surse- toate fis,ierele cu cod folosite în realizarea temei (.py/.ipynb);
       - Date- toate fis,ierele rezultate din modificări / prelucrări ale setului de date;
       - README.pdf- fis,ierul care cont,ine toate graficele create s,i documentat,ia.
    În fis,ierul README vet,i descrie modul de rezolvare pentru fiecare cerint,ă din Partea a II-
    a, răspunsurile la întrebările din cerint,ă s,i alte observat,ii; prima linie a fis,ierului va cont,ine
    numele complet, seria s,i grupa studentului care a rezolvat partea a II-a.
Punctajul pentru fiecare parte este împărt,it după cum urmează:
Partea I Punctaj
Cerint,e 50 puncte (40% cod + 40% rezultat + 20% documentat,ie)
BONUS folosire Git 20 puncte
Realizare interfat,ă grafică 20 puncte
TOTAL MAXIM ACORDAT 70 puncte

```
Partea a II-a Punctaj
Cerint,e 50 puncte (40% cod + 40% rezultat + 20% documentat,ie)
BONUS folosire Git 20 puncte
Realizare interfat,ă grafică 20 puncte
TOTAL MAXIM ACORDAT 70 puncte
```

Atent,ie!

- Pentru a primi punctaj, trebuie săprezentat,iproiectul în ultima săptămână a
    semestrului.
- Toate soluţiile trimise vor fi verificate, folosind o unealtă pentru detecta-
    rea plagiatului. În cazul depistării codului copiat (de pe Internet, colegi,
    din surse generate cu tool-uri tip ChatGPT), întregul punctaj pentru
    proiect este anulat.
- Pentru orice întrebare putet,i folosi forumul.
- Punctajul bonus pentru folosirea utilitaruluigiteste acordat raportat la numărul
    de cerint,e realizate s,i la complexitatea funct,ionalităt,ilor utilizate.
- Pentru bonus aleget,i dacă îl primit,i pentru utilizareagitsau pentru realizarea unei
    interfet,e grafice.


