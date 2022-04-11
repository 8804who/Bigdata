import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def merge_and_get(ldf, rdf, on, how="inner", index=None):
    if index is True:
        return pd.merge(ldf, rdf, how=how, left_index=True, right_index=True)
    else:
        return pd.merge(ldf, rdf, how=how, on=on)


pd.set_option('display.max_columns', None)

df_train = pd.read_csv('D:/공부/수업 자료/4-1/USG공유대학/빅데이터응용/실습자료/titanic/train.csv')
df_test= pd.read_csv('D:/공부/수업 자료/4-1/USG공유대학/빅데이터응용/실습자료/titanic/test.csv')

df_list = []
df_list.append(df_train)
df_list.append(df_test)

df=pd.concat(df_list, sort=False)
df=df.reset_index(drop=True)

number_of_train_dataset=df.Survived.notnull().sum()
number_of_test_dataset=df.Survived.isnull().sum()

y_true=df.pop("Survived")[:number_of_train_dataset]

pd.options.display.float_format='{:.2f}'.format
df.isnull().sum()/len(df)*100

df["Age"].fillna(df.groupby("Pclass")["Age"].transform("mean"),inplace=True)


df.loc[61, "Embarked"] = " S"
df.loc[829, "Embarked"] = " S"

object_columns = ["PassengerId", "Pclass", "Name", "Sex", "Ticket", "Cabin", "Embarked"]
numeric_columns = ["Age", "SibSp", "Parch", "Fare"]

for col_name in object_columns:
    df[col_name] = df[col_name].astype(object)
for col_name in numeric_columns:
    df[col_name] = df[col_name].astype(float)

df["Parch"] = df["Parch"].astype(int)
df["SibSp"] = df["SibSp"].astype(int)

pd.get_dummies(df["Sex"], prefix="Sex")

one_hot_df = merge_and_get(df, pd.get_dummies(df["Sex"], prefix="Sex"), on=None, index=True)
one_hot_df = merge_and_get(one_hot_df, pd.get_dummies(df["Pclass"], prefix="Pclass"), on=None, index=True)
one_hot_df = merge_and_get(one_hot_df, pd.get_dummies(df["Embarked"], prefix="Embarked"), on=None, index=True)

temp_columns = ["Sex", "Pclass", "Embarked"]
for col_name in temp_columns:
    temp_df = pd.merge(one_hot_df[col_name], y_true, left_index=True, right_index=True)
    sns.countplot(x="Survived", hue=col_name, data=temp_df)
    plt.show()

temp_df = pd.merge(one_hot_df[temp_columns], y_true, left_index=True,right_index=True)
g=sns.catplot(x="Embarked", hue="Pclass",col="Survived",data=temp_df,kind="count", height=4, aspect=.7)
plt.show()

temp_df = pd.merge(one_hot_df[temp_columns], y_true, left_index=True, right_index=True)
g=sns.FacetGrid(temp_df, col="Survived")
g.map_dataframe(sns.countplot, "Embarked", hue="Pclass", order=temp_df.Embarked.unique())
g.add_legend()
plt.show()

temp_df = pd.merge(one_hot_df[temp_columns], y_true, left_index=True, right_index=True)
g=sns.catplot(x="Pclass", hue="Sex", col="Survived", data=temp_df, kind="count", height=4, aspect=.7)
plt.show()

temp_df = pd.merge(one_hot_df[temp_columns], y_true, left_index=True, right_index=True)
g=sns.catplot(x="Embarked", hue="Sex", col="Survived", data=temp_df, kind="count", height=4, aspect=.7)
plt.show()

crosscheck_columns = [col_name for col_name in one_hot_df.columns.tolist() if col_name.split("_")[0] in temp_columns and "_" in col_name] + ["Sex"]

temp_df = pd.merge(one_hot_df[crosscheck_columns], y_true, left_index=True, right_index=True)
corr = temp_df.corr()
sns.set()
ax=sns.heatmap(corr, annot=True, linewidth=.5, cmap="YlGnBu")
crosscheck_columns = [col_name for col_name in one_hot_df.columns.tolist() if col_name.split("_")[0] in temp_columns and "_" in col_name] + ["Sex"]
temp_df = pd.merge(one_hot_df[crosscheck_columns], y_true, left_index=True, right_index=True)
corr = temp_df.corr()
sns.set()
ax=sns.heatmap(corr, annot=True, linewidths=.5, cmap="YlGnBu")
plt.show()