

-----Important----
#In all feature selection procedures, it is good practice to select the features by examining only the training set. And this is to avoid overfit


#Constant Features
#Check for null values
[col for col in data.columns if data[col].isnull.sum() > 0]

---### Using variance threshold from sklearn
#threshold=0.01 indicates variance of 1%
from sklearn.feature_selection import VarianceThreshold
sel = VarianceThreshold(threshold=0)
sel.fit(X_train)  # fit finds the features with zero variance

# Which are the constant features
sum(sel.get_support())
[x for x in X_train.columns if x not in X_train.columns[sel.get_support()]]

#Transforming the datasets by the constant features
X_train = sel.transform(X_train)
X_test = sel.transform(X_test)

X_train.shape, X_test.shape

#alternatively
# short and easy: find constant features
# in this case, all features are numeric, so this will suffice
constant_features = [
    feat for feat in X_train.columns if X_train[feat].std() == 0]
X_train.drop(labels=constant_features, axis=1, inplace=True)
X_test.drop(labels=constant_features, axis=1, inplace=True)

---#Removing constant features for categorical variables
# and now find those columns that contain only 1 label:
constant_features = [
    feat for feat in X_train.columns if len(X_train[feat].unique()) == 1]

---Removing Duplicate Features
#Using scikitlearn - we need to transpose the dataframe and identify indentical rows
#Using pandas transpose is computationally expensive, so the computer may run out of memory. 
#That is why we can only use this code block on small datasets.
data_t = X_train.T

# check if there are duplicated rows (the columns of the original dataframe)
data_t.duplicated().sum()
data_t[data_t.duplicated()]

-extracting the features that are duplicated
duplicated_features = data_t[data_t.duplicated()].index.values










