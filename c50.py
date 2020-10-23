import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, Normalizer, Binarizer
from sklearn.model_selection import cross_val_score 
from rpy2 import robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
import matplotlib.pyplot as plt
import math

df = pd.read_csv("audit_risk_norm.csv")

risk = df["Risk"]
del df["Risk"]

# normalizacja minmax
minmax = MinMaxScaler(feature_range=(0,1))
df_minmax = minmax.fit_transform(df)
df_minmax = pd.DataFrame(df_minmax, columns=df.columns)
df_minmax = df_minmax.join(risk)

X = df_minmax["Sector_score"].to_numpy().reshape(-1, 1)
y = df_minmax["Risk"].to_numpy()

X_df = df_minmax["Sector_score"]
y_df = df_minmax["Risk"]

clf = DecisionTreeClassifier(random_state=0)
clf.fit(X, y)
# 10-krotna skrośna walidacja
cv_score = cross_val_score(clf, X, y, cv=10)

# odpalenie C5.0
C50 = importr('C50')   
gmodels = importr('gmodels') 
C5_0 = ro.r('C5.0')

X_df = pd.DataFrame(X_df, columns = ['Sector_score'])
y_df = pd.DataFrame(y_df, columns = ['Risk'])

with localconverter(ro.default_converter + pandas2ri.converter):
  r_X_df = ro.conversion.py2rpy(X_df)

with localconverter(ro.default_converter + pandas2ri.converter):
  r_y_df = ro.conversion.py2rpy(y_df)

C50.C5_0(r_X_df, ro.vectors.FactorVector(r_y_df))