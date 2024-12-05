#import all necessary libraris
import numpy as np
import pandas as pd
#for creating and plotting the correlation matrix
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
import seaborn as sns

#libraries for implementing SMOTE{to handle imbalance class}
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix,roc_auc_score,roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from os import path
#download dataset and import the csv file uning pandas
path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
data = pd.read_csv(f"{path}/creditcard.csv")
#print some data to analeyze
print(data.head(6))
