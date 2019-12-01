

-----Important----
#In all feature selection procedures, it is good practice to select the features by examining only the training set. And this is to avoid overfit

Compare performance in machine learning models
#With Random Forest
def run_randomForests(X_train, X_test, y_train, y_test):
    rf = RandomForestClassifier(n_estimators=200, random_state=39, max_depth=4)
    rf.fit(X_train, y_train)
    print('Train set')
    pred = rf.predict_proba(X_train)
    print('Random Forests roc-auc: {}'.format(roc_auc_score(y_train, pred[:,1])))
    print('Test set')
    pred = rf.predict_proba(X_test)
    print('Random Forests roc-auc: {}'.format(roc_auc_score(y_test, pred[:,1])))

# original
run_randomForests(X_train_original.drop(labels=['ID'], axis=1),
                  X_test_original.drop(labels=['ID'], axis=1),
                  y_train, y_test)
    
# filter methods - correlation
run_randomForests(X_train.drop(labels=['ID'], axis=1),
                  X_test.drop(labels=['ID'], axis=1),
                  y_train, y_test)


------------

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

# Dropping the dupplicated features
X_train.drop(labels=duplicated_feat, axis=1, inplace=True)
X_test.drop(labels=duplicated_feat, axis=1, inplace=True)


--------Filtering based on Correlation
#Visualising correlated featres
corrmat = X_train.corr()
fig, ax = plt.subplots()
fig.set_size_inches(11,11)
sns.heatmap(corrmat)

----Brute Force
# with the following function we can select highly correlated features
# it will remove the first feature that is correlated with anything else
# without any other insight.

def correlation(dataset, threshold):
    col_corr = set()  # Set of all the names of correlated columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold: # we are interested in absolute coeff value
                colname = corr_matrix.columns[i]  # getting the name of column
                col_corr.add(colname)
    return col_corr

X_train.drop(labels=corr_features, axis=1, inplace=True)
X_test.drop(labels=corr_features, axis=1, inplace=True)
X_train.shape, X_test.shape


------Second Approach
# build a dataframe with the correlation between features
# remember that the absolute value of the correlation
# coefficient is important and not the sign

corrmat = X_train.corr()
corrmat = corrmat.abs().unstack() # absolute value of corr coef
corrmat = corrmat.sort_values(ascending=False)
corrmat = corrmat[corrmat >= 0.8]
corrmat = corrmat[corrmat < 1]
corrmat = pd.DataFrame(corrmat).reset_index()
corrmat.columns = ['feature1', 'feature2', 'corr']
corrmat.head()


# find groups of correlated features

grouped_feature_ls = []
correlated_groups = []

for feature in corrmat.feature1.unique():
    if feature not in grouped_feature_ls:

        # find all features correlated to a single feature
        correlated_block = corrmat[corrmat.feature1 == feature]
        grouped_feature_ls = grouped_feature_ls + list(
            correlated_block.feature2.unique()) + [feature]

        # append the block of features to the list
        correlated_groups.append(correlated_block)

print('found {} correlated groups'.format(len(correlated_groups)))
print('out of {} total features'.format(X_train.shape[1]))


# now we can visualise each group. We see that some groups contain
# only 2 correlated features, some other groups present several features 
# that are correlated among themselves.

for group in correlated_groups:
    print(group)
    print()

# we can now investigate further features within one group.
# let's for example select group 3

group = correlated_groups[2]
group

# we could select the features with less missing data
for feature in list(group.feature2.unique())+['v17']:
    print(X_train[feature].isnull().sum())
    
#Alternatively, we could build a machine learning algorithm using all the
#features from the above list, and select the more predictive one.    
from sklearn.ensemble import RandomForestClassifier
features = list(group.feature2.unique())+['v17']
rf = RandomForestClassifier(n_estimators=200, random_state=39, max_depth=4)
rf.fit(X_train[features].fillna(0), y_train)

# we get the feature importance attributed by the random forest model
importance = pd.concat(
    [pd.Series(features),
     pd.Series(rf.feature_importances_)], axis=1)

importance.columns = ['feature', 'importance']
importance.sort_values(by='importance', ascending=False)











