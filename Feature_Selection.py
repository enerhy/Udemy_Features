

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

#extracting the features that are duplicated
duplicated_features = data_t[data_t.duplicated()].index.values
duplicated_features = [col for col in data.columns if col not in data_unique.columns]

-creating a dataframe without the duplicate features (keeping the first feature)
data_unique = data_t.drop_duplicates(keep='first').T
data_unique.shape


#FOR BIG DATASETS:
# check for duplicated features in the training set
duplicated_feat = []
for i in range(0, len(X_train.columns)):
    if i % 10 == 0:  # this helps me understand how the loop is going
        print(i)

    col_1 = X_train.columns[i]

    for col_2 in X_train.columns[i + 1:]:
        if X_train[col_1].equals(X_train[col_2]):
            duplicated_feat.append(col_2)
         
# let's print the list of duplicated features
set(duplicated_feat) 
len(set(duplicated_feat))

#################
# we can go ahead and try to identify which set of features
# are identical

duplicated_feat = []
for i in range(0, len(X_train.columns)):

    col_1 = X_train.columns[i]

    for col_2 in X_train.columns[i + 1:]:

        # if the features are duplicated
        if X_train[col_1].equals(X_train[col_2]):

            #print them
            print(col_1)
            print(col_2)
            print()

            # and then append the duplicated one to a
            # list
            duplicated_feat.append(col_2)








