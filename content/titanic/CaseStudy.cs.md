# Případová studie – Titanic

## Obsah


[**Krok 1: Obchodní porozumění**](#Step-1:-Obchodní-porozumění)

[**Krok 2: Porozumění údajům**](#Step-2:-Porozumění-údajům)

- [**Načíst data**](#Načíst-data)
- [**Kontrola kvality dat**](#Kontrola-kvality-dat)
- [**Průzkumná analýza dat-EDA**](#Průzkumná-analýza-dat---EDA)
 
[**Krok 3: Příprava dat**](#Krok-3:-Příprava dat)
- [**Vypořádejte se s chybějícími daty**](#Vypořádejte-se-s-chybějícími-daty)
- [**Funkce inženýrství**](#funkce-inženýrství)

[**Krok 4: Modelování**](#Krok-4:-Modelování)

[Zpět na začátek](#Obsah)

## Krok 1: Obchodní porozumění
Tato počáteční fáze se zaměřuje na pochopení cílů a požadavků projektu z obchodní perspektivy a poté na převedení těchto znalostí do definice problému dolování dat a předběžného plánu navrženého k dosažení cílů.
#### Příběh Titaniku
Potopení RMS Titanic je jedním z nejneslavnějších vraků v historii. 15. dubna 1912, během své první plavby, se Titanic potopil po srážce s ledovcem a zabil 1502 z 2224 cestujících a posádky. Tato senzační tragédie šokovala mezinárodní společenství a vedla k lepším bezpečnostním předpisům pro lodě.

Jedním z důvodů, proč ztroskotání vedlo k takovým ztrátám na životech, byl nedostatek záchranných člunů pro cestující a posádku. I když přežití potopení zahrnovalo určitý prvek štěstí, některé skupiny lidí měly větší šanci na přežití než jiné, jako jsou ženy, děti a cestující z vyšší třídy.

#### Cíl
V této výzvě dokončíme analýzu toho, jaké druhy lidí pravděpodobně přežijí.

Kromě toho vytvoříme regresní model pro predikci ceny letenky (jízdného).


[Zpět na začátek](#Obsah)

## Krok 2: Porozumění datům
Fáze porozumění datům začíná počátečním sběrem dat a pokračuje činnostmi s cílem seznámit se s daty, identifikovat problémy s kvalitou dat, objevit první pohledy na data nebo odhalit zajímavé podmnožiny za účelem vytvoření hypotéz o skrytých informacích. Tento krok se často kombinuje s dalším krokem, přípravou dat.
### Datový slovník
Data jsou v souboru csv ```titanic.csv```.

| Proměnná | Definice | Klíč |
| --- | --- | --- |
| survival | Přežití | 0 = ne, 1 = ano |
| pclass | Třída vstupenek | 1 = 1., 2 = 2., 3 = 3. |
| sex | Sex | muž/žena |
| Age | Věk | v letech |
| sibsp | Počet sourozenců / manželů na palubě Titaniku | |
| parch | Počet rodičů / dětí na palubě Titaniku | |
| ticket | Číslo lístku | |
| fare | Jízdné pro cestující | |
| cabin | Číslo kabiny | |
| embarked | Přístav nalodění | C = Cherbourg, Q = Queenstown, S = Southampton |

**Poznámky proměnných**
- pclass: proxy pro socioekonomický status (SES)
  - 1. = horní
  - 2. = střední
  - 3. = nižší

- věk: Věk je zlomek, pokud je menší než 1. Pokud je věk odhadován, je ve tvaru xx.5

- sibsp: Dataset definuje rodinné vztahy tímto způsobem...
- Sibling (Sourozenec) = bratr, sestra, nevlastní bratr, nevlastní sestra
- Spouse (Manžel) = manžel, manželka (milenky a snoubenci byli ignorováni)

- parch: Dataset definuje rodinné vztahy tímto způsobem...
  - Rodič = matka, otec
  - Dítě = dcera, syn, nevlastní dcera, nevlastní syn
  - Některé děti cestovaly pouze s chůvou, proto pro ně parch=0.


### Načíst data

Tato datová sada je v titanic.csv. Ujistěte se, že je soubor v aktuální složce.

import pandas as pd
import matplotlib.pyplot as plt
import piplite
await piplite.install('seaborn')
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
df_titanic = pd.read_csv('titanic.csv')
df_titanic.head()

### Zkontrolujte kvalitu dat
Zkontrolujte kvalitu dat. Nejběžnější kontrolou je kontrola chybějících hodnot. Můžeme provést základní čištění dat, jako je čištění pole měn.
- Zkontrolujte nulové hodnoty
- Pole měny je třeba převést na plovoucí, odstranit '$' nebo ',', někdy je záporná hodnota uzavřena v ()

##### Úkol 1: Podívejte se na základní informace o datovém rámci

Nápověda: funkce info().
Diskutujte o chybějících hodnotách v datovém rámci.
df_titanic.info()
#Další způsob, jak zobrazit počet chybějících hodnot v každém sloupci
df_titanic.isnull().sum()
##### Úkol 2: Vyčistit jízdné, převést na plovoucí
Odstraňte "$" z Fare, převeďte datový typ na float.
# uklidit Fare, převést na float
df_titanic.Fare = df_titanic.Fare.str.replace('$','')
df_titanic['Fare'] = df_titanic.Fare.astype(float)
df_titanic.head()
##### Úkol 3: Podívejte se na statistiky Numeric Columns

Funkce Hint:describe().

Diskutujte:
- Věk, SibSp, Parch, statistiky jízdného
- Co znamená Přežil?
df_titanic.describe()
### Průzkumná analýza dat - EDA
EDA je přístup k analýze souborů dat za účelem shrnutí jejich hlavních charakteristik, často pomocí vizuálních metod.

#### Typy funkcí
##### Kategorické vlastnosti:
Kategorická proměnná je taková, která má dvě nebo více kategorií a každá hodnota v tomto rysu může být podle nich kategorizována. Například pohlaví je kategorická proměnná, která má dvě kategorie (muž a žena). Nyní nemůžeme třídit ani dávat žádné resp
vzhledem k takovým proměnným. Jsou také známé jako nominální proměnné.

Kategorické funkce v datové sadě: Sex,Embarked.

##### Souvislá funkce:
Prvek se nazývá spojitý, pokud může nabývat hodnot mezi libovolnými dvěma body nebo mezi minimálními či maximálními hodnotami ve sloupci prvků.

Souvislé funkce v datové sadě: Jízdné
### Kategorické vlastnosti
Budeme analyzovat Survived jako univariantní. Vztah mezi sexem a přežitím, embarkovaným a přežitím.

#### Kolik přežilo
Sloupcový graf ve sloupci Přežil. Existuje několik způsobů, jak vytvořit sloupcový graf. Ukážeme si zde 2 způsoby, seaborn count plot a pandas series bar.
##### Úkol 4: Vykreslete sloupcový graf pro Zahynulé vs. Přežilé
Vykreslit sloupcový graf pro sloupec Přežil. Survived=0 znamená zahynulo, Survived=1 znamená přežilo.
#Kolik jich přežilo
f,ax=plt.subplots(figsize=(5,5))
sns.countplot('Survived',data=df_titanic, ax = ax)
ax.set_title('Zahynulí vs. přežili')
#Není nutné, jen k odstranění jakéhokoli výstupu
plt.show()
#počet přeživších
f,ax=plt.subplots(figsize=(5,5))
survivd_counts = df_titanic.Survived.value_counts()
survivd_counts.plot.bar(ax=ax)
ax.set_title('Zahynulí vs. přežili')
plt.show()
#Procento přeživších
f,ax=plt.subplots(figsize=(5,5))
přeživší_počet = df_titanic.Survived.value_counts(normalize=True)
přežil_počet.plot.bar(ax=ax)
ax.set_title('Zahynulí vs. přežili')
ax.set_xticklabels( ['Zhynulo', 'Přežilo'], rotace=0)
plt.show()
#### Vztah mezi sexem a přežitím
Můžeme použít agregační funkci nebo graf.

Další 2 buňky demonstrují agregační funkci.

Následující buňka znázorňuje sloupcový graf a graf počtu.

##### Úkol 5: Vykreslete sloupcový graf počtu cestujících mužů a žen

Tip: Použijte seaborn countplot().
#Muž vs. Žena
f,ax=plt.subplots(figsize=(5,5))
sns.countplot('Sex',data=df_titanic,ax=ax)
ax.set_title('Muž vs. Žena')
plt.show()
##### Úkol 6: Skupinové pohlaví, abyste zjistili míru přežití mužů a žen
#míra přežití žen/mužů
df_titanic.groupby(['Sex'], as_index=False).agg({'Survived':'mean'})
##### Úkol 7: Vykreslení zániku vs. bar přežití pro muže a ženy
Znovu použijeme seaborn countplot(), ale nastavíme argument `hue` na 'Survived'.
#Zhynul vs. přežil pro muže/ženu
f,ax=plt.subplots(figsize=(5,5))
sns.countplot('Sex',hue='Survived',data=df_titanic,ax=ax)
ax.set_title('Pohlaví: Zahynulý vs. Přežil')
plt.show()
Počet mužů na lodi je mnohem vyšší než počet žen. Stále je počet přeživších žen téměř dvojnásobný než počet přeživších mužů. Většina žen přežila, zatímco velká většina mužů zahynula.
#### Pclass a přežití
##### Úkol 8: Uveďte míru přežití každé třídy P
df_titanic.groupby(['Pclass'], as_index=False).agg({'Survived':'mean'})
##### Úkol 9: Vykreslení zániku vs. přežití pro každou třídu P
# barový pozemek a seaborn počítat spiknutí
f,ax=plt.subplots(figsize=(5,5))
sns.countplot('Pclass',hue='Survived',data=df_titanic,ax=ax)
ax.set_title('Pclass:Perished vs. Survived')
plt.show()
### Průběžné funkce

#### Jednorozměrný distribuční graf
Histogram lze udělat několika způsoby. Ukážeme si 3 způsoby.
- ax.hist(): nemůže zpracovat hodnotu NnN
- seaborn.distplot(): nezvládne NaN. Ve výchozím nastavení má KDE (odhad hustoty jádra).
- pd.Sereis.hist(): nejjednodušší a standardně zvládne NaN
##### Úkol 10: Vykreslete histogram pro věk
Použijte funkci pandas Series hist(), která zpracovává chybějící hodnotu.
# použít dataframe hist(), který bude standardně zpracovávat NaN
obr, ax = plt.subplots()
df_titanic.Age.hist(ax=ax, bins=20, edgecolor='black', alpha=0.5)
##### Úkol 11: Naskládejte věkový histogram přežitých na vrchol celkového věkového histogramu
Vykreslete histogram pro věk, poté odfiltrujte přeživší cestující a vykreslete histogram pro věk na stejné ose. Nastavte jinou barvu a štítek.
#use dataframe hist(), který bude standardně zpracovávat NaN
obr, ax = plt.subplots()
df_titanic.Age.hist(ax=ax, label='all', bins=20, edgecolor='black', alpha=0.5)
#stack přežil
df_titanic[df_titanic.Survived==1].Age.hist(ax=ax, bins=20, color='g', label='survived', edgecolor='black', alpha=0.5)
ax.set_title('Věková distribuce')
ax.legend()
Děti mají vyšší míru přežití.
[Zpět na začátek](#Obsah)

## Krok 3: Příprava dat
Vytvářejte nové funkce prostřednictvím inženýrství funkcí; Vypořádat se s chybějícími hodnotami; Vyčistit data, tzn. odstranit nadbytečné bílé mezery v hodnotách řetězce. Zaměříme se na řešení chybějících údajů v tomto slovním spojení.
#zkontrolovat všechna chybějící data
df_titanic.isnull().sum()
### Vypořádejte se s chybějícími daty
Předvedeme si plnění průměrem/režimem a odhadem z dalších sloupců.

#### Vyplňte průměrem/režimem
Embarked má pouze 2 chybějící hodnoty a neexistuje žádný zřejmý způsob, jak odhadnout chybějící hodnotu, jednoduše ji doplníme režimem sloupce nebo 'S'
##### Úkol 12: Doplňte chybějící Nastoupili jste s režimem
# fill NaN v režimu Embarked with
df_titanic['Embarked'].fillna(df_titanic.Embarked.mode()[0],inplace=True)
df_titanic.info()
#### Vyplňte odhadovanou hodnotou

Titul je slovo používané ve jménu osoby v určitých kontextech. Může to znamenat buď úctu, oficiální postavení nebo profesionální neboakademická kvalifikace. Je to dobrý údaj o věku, například Mr je pro dospělého muže, Master je pro mladé chlapce.

Pokud se podíváme na všechna jména cestujících Titaniku, vidíme, že jméno je ve formátu Last, Title. První. Tyto informace můžeme použít k odhadu chybějících věků.

- Nejprve použijeme regulární výraz k extrahování názvu z názvu.
- Poté převedeme název na velká písmena.
- Chybějící věk pak doplníme průměrným věkem konkrétního titulu.
# extrahujte předponu z názvu
df_titanic['Title']=df_titanic.Name.str.extract('([A-Za-z]+\.)')
df_titanic.head()
##### Úkol 13: převeďte počáteční písmena na velká písmena.
Abychom zajistili přesný průměrný věk každé iniciály, převedeme iniciály na všechna velká písmena.
df_titanic.Title = df_titanic.Title.str.upper()
df_titanic.head()
##### Úkol 14: Doplňte chybějící věk průměrným věkem iniciály
df_titanic.Title.value_counts()
df_titanic.Age.fillna(df_titanic.groupby('Title').Age.transform('mean'), inplace=True)
df_titanic.info()
### Funkce inženýrství
Vytvoříme nový sloupec FamilySize. Existují 2 sloupce týkající se velikosti rodiny, parch označuje číslo rodiče nebo dětí, Sibsp označuje číslo sourozence a manžela.

Vezměte si jako příklad jedno jméno 'Asplund', můžeme vidět, že celková velikost rodiny je 7 (Parch + SibSp + 1) a každý člen rodiny má stejné jízdné, což znamená, že jízdné je pro celou skupinu. Velikost rodiny tedy bude důležitou vlastností pro předpovídání Fare. V datové sadě jsou pouze 4 Asplundy ze 7, protože datová sada je pouze podmnožinou všech cestujících.
df_titanic[df_titanic.Name.str.contains('Asplund')]
##### Úkol 15: Vytvořte sloupec Rodinná velikost 'FamilySize'
Rodinná velikost FamilySize = Parch + SibSp + 1
df_titanic['FamilySize'] = df_titanic.Parch + df_titanic.SibSp + 1
df_titanic.sample(5)

[Zpět na začátek](#Obsah)

## Krok 4: Modelování

Nyní máme relativně čistou datovou sadu (kromě sloupce **Cabin**, který má mnoho chybějících hodnot). Můžeme provést klasifikaci na Survived, abychom předpověděli, zda cestující katastrofu přežije, nebo regresi na Fare, abychom předpověděli jízdné. Tato datová sada není vhodná pro regresi. Ale protože v tomto workshopu nehovoříme o klasifikaci, zkonstruujeme v tomto cvičení lineární regresi na Fare.

import statsmodels.formula.api as smf
result = smf.ols("Fare ~ C(Pclass) + C(Embarked) + FamilySize", data=df_titanic).fit()
result.summary()

