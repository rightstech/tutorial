# Case Study - Titanic

## Table of Contents


[**Step 1: Business Understanding**](#Step-1:-Business-Understanding)

[**Step 2: Data Understanding**](#Step-2:-Data-Understanding)

- [**Load Data**](#Load-Data)
- [**Check Data Quality**](#Check-Data-Quality)
- [**Exploratory Data Analysis-EDA**](#Exploratory-Data-Analysis---EDA)
 
[**Step 3: Data Preparation**](#Step-3:-Data-Preparation)
- [**Deal with Missing Data**](#Deal-with-Missing-Data)
- [**Feature Engineering**](#Feature-Engineering)

[**Step 4: Modeling**](#Step-4:-Modeling)

[Back to Top](#Table-of-Contents)

## Step 1: Business Understanding
This initial phase focuses on understanding the project objectives and requirements from a business perspective, and then converting this knowledge into a data mining problem definition, and a preliminary plan designed to achieve the objectives.
#### Titanic Story
The sinking of the RMS Titanic is one of the most infamous shipwrecks in history.  On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. This sensational tragedy shocked the international community and led to better safety regulations for ships.

One of the reasons that the shipwreck led to such loss of life was that there were not enough lifeboats for the passengers and crew. Although there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others, such as women, children, and the upper-class passengers.

#### Objective
In this challenge, we will complete the analysis of what sorts of people were likely to survive. 

In addition, we will build a regression model to predict ticket price(Fare).


[Back to Top](#Table-of-Contents)

## Step 2: Data Understanding
The data understanding phase starts with an initial data collection and proceeds with activities in order to get familiar with the data, to identify data quality problems, to discover first insights into the data, or to detect interesting subsets to form hypotheses for hidden information. This step is often mixed with the next step, Data Preparation.
### Data Dictionary
The data is in a csv file titanic.csv. 

| Variable | Definition | Key |
| --- | --- | --- |
| survival | Survival | 0 = No, 1 = Yes |
| pclass | Ticket class	| 1 = 1st, 2 = 2nd, 3 = 3rd |
| sex | Sex | male/female |	
| Age | Age | in years |
| sibsp | # of siblings / spouses aboard the Titanic | |
| parch | # of parents / children aboard the Titanic | |
| ticket | Ticket number | |
| fare | Passenger fare | |
| cabin | Cabin number | |
| embarked | Port of Embarkation | C = Cherbourg, Q = Queenstown, S = Southampton |

**Variable Notes**
- pclass: A proxy for socio-economic status (SES)
 - 1st = Upper
 - 2nd = Middle
 - 3rd = Lower

- age: Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5

- sibsp: The dataset defines family relations in this way...
- Sibling = brother, sister, stepbrother, stepsister
- Spouse = husband, wife (mistresses and fiancés were ignored)

- parch: The dataset defines family relations in this way...
 - Parent = mother, father
 - Child = daughter, son, stepdaughter, stepson
 - Some children travelled only with a nanny, therefore parch=0 for them.


### Load Data

This dataset is in titanic.csv. Make sure the file is in current folder.
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

### Check Data Quality
Check data quality. Most common check is to check missing values. We can do some basic data cleaning like cleaning up currency field.
- Check null values
- Currency field need to be converted to float, remove '$' or ',', sometimes negative value is enclosed in ()

##### Task1: Check out Basic Dataframe Info

Hint: info() function.
Discuss missing values in the dataframe.
df_titanic.info()
#Another way to see number of missing values in each column
df_titanic.isnull().sum()
##### Task2: Clean up Fare, Convert to Float
Strip "$" from Fare, convert datatype to float.
#clean up Fare, convert to float
df_titanic.Fare = df_titanic.Fare.str.replace('$','')
df_titanic['Fare'] = df_titanic.Fare.astype(float)
df_titanic.head()
##### Task3: Check out statistics of Numeric Columns

Hint:describe() function.

Discuss:
- Age, SibSp, Parch, Fare statistics
- What does mean Survived mean?
df_titanic.describe()
### Exploratory Data Analysis - EDA
EDA is an approach to analyzing data sets to summarize their main characteristics, often with visual methods.

#### Types Of Features
##### Categorical Features:
A categorical variable is one that has two or more categories and each value in that feature can be categorised by them.For example, gender is a categorical variable having two categories (male and female). Now we cannot sort or give any ordering to such variables. They are also known as Nominal Variables.

Categorical Features in the dataset: Sex,Embarked.

##### Continous Feature:
A feature is said to be continous if it can take values between any two points or between the minimum or maximum values in the features column.

Continous Features in the dataset: Fare
### Categorical Features
We will analysis Survived as univariant. Relationship between Sex and Survival, Embarked and Survivval.

#### How many survived
Bar chart on Survived column. There are multiple ways to do the bar chart. We will demonstrate 2 ways here, seaborn countplot and pandas series bar.
##### Task4: Plot bar chart for Perished vs. Survived
Plot bar chart for Survived column. Survived=0 means perished, Survived=1 means Survived.
#How many survived
f,ax=plt.subplots(figsize=(5,5))
sns.countplot('Survived',data=df_titanic, ax = ax)
ax.set_title('Perished vs. Survived')
#Not necessary, just to eliminate any output
plt.show()
#counts of survived
f,ax=plt.subplots(figsize=(5,5))
survived_counts = df_titanic.Survived.value_counts()
survived_counts.plot.bar(ax=ax)
ax.set_title('Perished vs. Survived')
plt.show()
#Percent of survived
f,ax=plt.subplots(figsize=(5,5))
survived_counts = df_titanic.Survived.value_counts(normalize=True)
survived_counts.plot.bar(ax=ax)
ax.set_title('Perished vs. Survived')
ax.set_xticklabels( ['Perished', 'Survived'], rotation=0)
plt.show()
#### Relationship between Sex and Survival
We may use aggregation function or plot.

Next 2 cells demonstate aggregate function.

The following cell demonstrates bar plot and countplot.

##### Task5: Plot Bar Chart on Number of Male and Femal Passengers

Hint: Use seaborn countplot().
#Male vs. Female
f,ax=plt.subplots(figsize=(5,5))
sns.countplot('Sex',data=df_titanic,ax=ax)
ax.set_title('Male vs. Female')
plt.show()
##### Task6: Groupby Sex to Find Survival Rate of Male and Female
#female/male survival rate
df_titanic.groupby(['Sex'], as_index=False).agg({'Survived':'mean'})
##### Task7: Plot Perished vs. Survived Bar for Male and Femail
We will use seaborn countplot() again, but set argument `hue` to 'Survived'.
#Perished vs. Survived for male/female
f,ax=plt.subplots(figsize=(5,5))
sns.countplot('Sex',hue='Survived',data=df_titanic,ax=ax)
ax.set_title('Gender: Perished vs. Survived')
plt.show()
The number of men on the ship is lot more than the number of women. Still the number of survived women is almost twice the number of survived males. Majority women survived while vast majority of men perished.
#### Pclass and Survival
##### Task 8: List survival rate of each Pclass 
df_titanic.groupby(['Pclass'], as_index=False).agg({'Survived':'mean'})
##### Task9: Plot Perished vs. Survived for each Pclass
#bar plot and seaborn countplot
f,ax=plt.subplots(figsize=(5,5))
sns.countplot('Pclass',hue='Survived',data=df_titanic,ax=ax)
ax.set_title('Pclass:Perished vs. Survived')
plt.show()
### Continuous Features

#### Univariate Distribution Plot
There are multiple ways to do histogram. I will demonstrate 3 ways.
- ax.hist(): can not handle NnN value
- seaborn.distplot(): can not handle NaN. Has KDE(kernel density estimation) by default.
- pd.Sereis.hist(): simplest and can handle NaN by default
##### Task10: Plot histogram for Age
Use pandas Series hist() function which handles missing value.
#use dataframe hist() which will handle NaN by default
fig, ax = plt.subplots()
df_titanic.Age.hist(ax=ax, bins=20, edgecolor='black', alpha=0.5)
##### Task11: Stack age histogram of survived on top of overall age histogram
Plot histogram for Age, then filter out survived passenger and plot histogram for Age on same axis. Set different color and label.
#use dataframe hist() which will handle NaN by default
fig, ax = plt.subplots()
df_titanic.Age.hist(ax=ax, label='all', bins=20, edgecolor='black', alpha=0.5)
#stack survived
df_titanic[df_titanic.Survived==1].Age.hist(ax=ax, bins=20, color='g', label='survived', edgecolor='black', alpha=0.5)
ax.set_title('Age Distribution')
ax.legend()
Children have higher survival rate.
[Back to Top](#Table-of-Contents)

## Step 3: Data Preparation
Create new features through feature engineering; Deal with missing values; Clean up data, ie. strip extra white spaces in string values. We will focus on dealing with missing data in this phrase.
#check all missing data
df_titanic.isnull().sum()
### Deal with Missing Data
We will demonstrate filling with mean/mode and estimate from other columns.

#### Fill with Mean/Mode
Embarked only has 2 missing values and there is no obvious way to estimate the missing walue, we will simply fill it with mode of the column, or 'S'
##### Task12: Fill missing Embarked with mode
#fill NaN in Embarked with mode
df_titanic['Embarked'].fillna(df_titanic.Embarked.mode()[0],inplace=True)
df_titanic.info()
#### Fill with Estimated Value

A title is a word used in a person's name, in certain contexts. It may signify either veneration, an official position, or a professional or academic qualification. It's a good indication of age, for example, Mr is for adult man, Master is for young boys.

If we look at all names of Titanic passengers, we can see that the name is in format Last, Title. First. We can use this information to estimate missing ages.

- First, we will use regular expression to extract title from name.
- Then we will convert title to upper case.
- Then we fill missing age with mean age of specific title.
#extract prefix from name
df_titanic['Title']=df_titanic.Name.str.extract('([A-Za-z]+\.)')
df_titanic.head()
##### Task13: convert initial to upper case.
To ensure we get accurate mean age of each initial, we convert initial to all upper case.
df_titanic.Title = df_titanic.Title.str.upper()
df_titanic.head()
##### Task14: Fill missing age with mean age of the initial
df_titanic.Title.value_counts()
df_titanic.Age.fillna(df_titanic.groupby('Title').Age.transform('mean'), inplace=True)
df_titanic.info()
### Feature Engineering
We'll create a new column FamilySize. There are 2 columns related to family size, parch indicates parent or children number, Sibsp indicates sibling and spouse number.

Take one name 'Asplund' as example, we can see that total family size is 7(Parch + SibSp + 1), and each family member has same Fare, which means the Fare is for the whole group. So family size will be an important feature to predict Fare. There're only 4 Asplunds out of 7 in the dataset becasue the dataset is only a subset of all passengers.
df_titanic[df_titanic.Name.str.contains('Asplund')]
##### Task15: Create column 'FamilySize'
FamilySize = Parch + SibSp + 1
df_titanic['FamilySize'] = df_titanic.Parch + df_titanic.SibSp + 1
df_titanic.sample(5)
[Back to Top](#Table-of-Contents)

## Step 4: Modeling

Now we have a relatively clean dataset (except for the **Cabin** column which has many missing values). We can do a classification on Survived to predict whether a passenger could survive the disaster or a regression on Fare to predict ticket fare. This dataset is not a good dataset for regression. But since we don't talk about classification in this workshop we will construct a linear regression on Fare in this exercise.
import statsmodels.formula.api as smf
result = smf.ols("Fare ~ C(Pclass) + C(Embarked) + FamilySize", data=df_titanic).fit()
result.summary()