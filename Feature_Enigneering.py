
# Introducing nan values, where another sign indicates that
data = data.replace('?', np.nan)

# 2 values in one cell - splitting
def get_first_cabin(row):
    try:
        return row.split()[0]
    except:
        return np.nan 
      
data['cabin'] = data['cabin'].apply(get_first_cabin)

# Use only some columns of a dataset
data = pd.read_csv('../loan.csv', usecols=use_cols).sample(
    10000, random_state=44)


-----# Date Time Variables
data.dtypes
# now let's parse the dates, currently coded as strings, into datetime format

data['issue_dt'] = pd.to_datetime(data.issue_d)
data['last_pymnt_dt'] = pd.to_datetime(data.last_pymnt_d)

data[['issue_d', 'issue_dt', 'last_pymnt_d', 'last_pymnt_dt']].head()

#Pivot into a DataFrame
data.groupby(['issue_dt', 'grade'])['loan_amnt'].sum().unstack()


----Missing Data
data.isnull().sum()
data.isnull().mean()
data.emp_length.value_counts() / len(data)  #to find other sorts of indication for missing data (e.g ?) 

# Check for Missing Data Not At Random (MNAR) - systematic missing data
# creating a binary variable indicating whether a value is missing
data['cabin_null'] = np.where(data['cabin'].isnull(), 1, 0)

# evaluate % of missing data in the target classes
data.groupby(['survived'])['cabin_null'].mean()


# Dict with categories to change column valaues
length_dict = {k: '0-10 years' for k in data.emp_length.unique()}
length_dict['10+ years'] = '10+ years'
length_dict['n/a'] = 'n/a'
# Mapping
data['emp_length_redefined'] = data.emp_length.map(length_dict

# Subset of data whith no nan in a certain coulumn
data.(subset=['emp_title'])
                                                   
employed = len(data.dropna(subset=['emp_title']))
# % of borrowers within each category
data.dropna(subset=['emp_title']).groupby(
    ['emp_length_redefined'])['emp_length'].count().sort_values() / employe
/--------------------------						   
						  )

------- Imputation for Missing values

# these are the objects we need to impute missing data
# with sklearn
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# first we need to make lists, indicating which features
# will be imputed with each method
numeric_features_mean = ['LotFrontage']
numeric_features_median = ['MasVnrArea', 'GarageYrBlt']

# Instantiating the imputers within a pipeline
numeric_mean_imputer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
])

numeric_median_imputer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
])

# Use ColumnTransformer
# remainder = True to indicate that we want all columns at the end of the
# transformation 
preprocessor = ColumnTransformer(transformers=[
    ('mean_imputer', numeric_mean_imputer, numeric_features_mean),
    ('median_imputer', numeric_median_imputer, numeric_features_median)
], remainder='passthrough')
# now we fit the preprocessor
preprocessor.fit(X_train)
# and now we can impute the data
X_train = preprocessor.transform(X_train)


# Transform into a DataFrame
# IF we used all features in the transformer:
X_train = pd.DataFrame(X_train,
             columns=features_numeric+features_categoric)
X_test = pd.DataFrame(X_test,
             columns=features_numeric+features_categoric)

# If only some features were in the transformer:

#get the reminding coulmbs:
preprocessor.transformers_
#put them into a list and concatinate with the other columns
remainder_cols = [cols_to_use[c] for c in [0, 1, 2, 3, 4, 5]]
pd.DataFrame(X_train,
             columns = numeric_features_mean+numeric_features_median+remainder_cols).head()


# and we can look at the parameters learnt like this:
preprocessor.named_transformers_['mean_imputer'].named_steps['imputer'].statistics_
preprocessor.transformers



---------------CATEGORICAL VARIABLES------------

-----ONE-HOT-Encoding
# Trees do not perform well in datasets with big feature spaces.
# Thus One-Hot-Encoding is not the best option for them

#Scikit Learn - it transforms it into array without labels
# we create and train the encoder
encoder = OneHotEncoder(categories='auto',
                       drop='first', # to return k-1, use drop=false to return k dummies
                       sparse=False,
                       handle_unknown='ignore') # helps deal with rare labels

encoder.fit(X_train.fillna('Missing'))

# With Get-Dumies: If the test set is not with the same categories - this will be a problem
def get_OHE(df):
    df_OHE = pd.concat(
        [df[['pclass', 'age', 'sibsp', 'parch', 'fare']],
         pd.get_dummies(df[['sex', 'cabin', 'embarked']], drop_first=True)],
        axis=1)
    return df_OHE

X_train_OHE = get_OHE(X_train)
X_test_OHE = get_OHE(X_test)

X_train_OHE.head()



#One hot encoding with Feature-Engine
ohe_enc = OneHotCategoricalEncoder(
    top_categories=None, # we can choose to encode only top categories and see them with ohe_enc.encoder_dict
    variables=['sex', 'embarked'], # we can select which variables to encode, or not include the argument to select all
    drop_last=True) # to return k-1, false to return k

ohe_enc.fit(X_train)
tmp = ohe_enc.transform(X_test)

ohe_enc.variables # returns the variable that will be encoded


-----Ordinal encoding is suitable for Tree based models
# for integer encoding using sklearn
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict

d = defaultdict(LabelEncoder)
# Encoding the variable
train_transformed = X_train.apply(lambda x: d[x.name].fit_transform(x))

# # Using the dictionary to encode future data
test_transformed = X_test.apply(lambda x: d[x.name].transform(x))
#OR to apply on a certain variables
test_transformed = X_test[['variable_names']].apply(lambda x: d[x.name].transform(x))

# # Inverse the encoded
tmp = train_transformed.apply(lambda x: d[x.name].inverse_transform(x))
tmp.head()


--WITH Feature-Engine
# for integer encoding using feature-engine
from feature_engine.categorical_encoders import OrdinalCategoricalEncoder
ordinal_enc = OrdinalCategoricalEncoder(
    encoding_method='arbitrary',
    variables=['Neighborhood', 'Exterior1st', 'Exterior2nd'])

ordinal_enc.fit(X_train)
# check the mapping
ordinal_enc.encoder_dict_
ordinal_enc.variables
X_train = ordinal_enc.transform(X_train)
X_test = ordinal_enc.transform(X_test)


--------Count/ Frequency encoding
# suitable for Tree based models
# - cannot handle new categories in the test set
# - two categories will be replaced with the same number if appear equaly



-------Ordered Integer Encoding
# Suitable also for linear models
# - cannot handle new categories
def find_category_mappings(df, variable, target):

    # first  we generate an ordered list with the labels
    ordered_labels = X_train.groupby([variable
                                      ])[target].mean().sort_values().index

    # return the dictionary with mappings
    return {k: i for i, k in enumerate(ordered_labels, 0)}


def integer_encode(train, test, variable, ordinal_mapping):

    X_train[variable] = X_train[variable].map(ordinal_mapping)
    X_test[variable] = X_test[variable].map(ordinal_mapping)
	
	
# and now we run a loop over the remaining categorical variables
for variable in ['Exterior1st', 'Exterior2nd']:

    mappings = find_category_mappings(X_train, variable, 'SalePrice')

    integer_encode(X_train, X_test, variable, mappings)
					      
					      
--# WITH Feature-Engine				      

ordinal_enc = OrdinalCategoricalEncoder(
    # NOTE that we indicate ordered in the encoding_method, otherwise it assings numbers arbitrarily
    encoding_method='ordered',
    variables=['Neighborhood', 'Exterior1st', 'Exterior2nd'])

ordinal_enc.fit(X_train, y_train)
X_train = ordinal_enc.transform(X_train)
X_test = ordinal_enc.transform(X_test)

ordinal_enc.encoder_dict_
ordinal_enc.variables


----------Encoding with Mean-Encoding
#Replacing categorical labels with this code and method will generate missing values
#for categories present in the test set that were not seen in the training set. 
#Therefore it is extremely important to handle rare labels before-hand

def find_category_mappings(df, variable, target):
    return df.groupby([variable])[target].mean().to_dict()

def integer_encode(train, test, variable, ordinal_mapping):
    X_train[variable] = X_train[variable].map(ordinal_mapping)
    X_test[variable] = X_test[variable].map(ordinal_mapping)
	
for variable in ['sex', 'embarked']:
    mappings = find_category_mappings(X_train, variable, 'survived')
    integer_encode(X_train, X_test, variable, mappings)

	
-- #With Feature-Engine
from feature_engine.categorical_encoders import MeanCategoricalEncoder
mean_enc = MeanCategoricalEncoder(
    variables=['cabin', 'sex', 'embarked'])
mean_enc.fit(X_train, y_train)
X_train = mean_enc.transform(X_train)
X_test = mean_enc.transform(X_test)

mean_enc.encoder_dict_
mean_enc.variables


					      -
---------Probability Ration Encoding
# Only for binary classification problems
#Replacing categorical labels with this code and method will generate missing values
#for categories present in the test set that were not seen in the training set. 
#Therefore it is extremely important to handle rare labels before-hand
			   
ratio_enc = WoERatioCategoricalEncoder(
    encoding_method = 'ratio',
    variables=['cabin', 'sex', 'embarked'])						   

ratio_enc.fit(X_train, y_train)						   
X_train = ratio_enc.transform(X_train)
X_test = ratio_enc.transform(X_test)		



--------------RARE LABELS---------

def find_non_rare_labels(df, variable, tolerance):
    temp = df.groupby([variable])[variable].count() / len(df)
    non_rare = [x for x in temp.loc[temp>tolerance].index.values]
    return non_rare
# non-rare Labels
find_non_rare_labels(X_train, 'Neighborhood', 0.05)

# rare labels
[x for x in X_train['Neighborhood'].unique(
) if x not in find_non_rare_labels(X_train, 'Neighborhood', 0.05)]

####
def rare_encoding(X_train, X_test, variable, tolerance):

    X_train = X_train.copy()
    X_test = X_test.copy()

    # find the most frequent category
    frequent_cat = find_non_rare_labels(X_train, variable, tolerance)

    # re-group rare labels
    X_train[variable] = np.where(X_train[variable].isin(
        frequent_cat), X_train[variable], 'Rare')
    
    X_test[variable] = np.where(X_test[variable].isin(
        frequent_cat), X_test[variable], 'Rare')

    return X_train, X_test

# Transforming						   
for variable in ['Neighborhood', 'Exterior1st', 'Exterior2nd']:
    X_train, X_test = rare_encoding(X_train, X_test, variable, 0.05)
	
---With Feature-Engine

from feature_engine.categorical_encoders import RareLabelCategoricalEncoder						   
# Rare value encoder
rare_encoder = RareLabelCategoricalEncoder(
    tol=0.05,  # minimal percentage to be considered non-rare
    n_categories=4, # minimal number of categories the variable should have to re-cgroup rare categories
    variables=['Neighborhood', 'Exterior1st', 'Exterior2nd',
               'MasVnrType', 'ExterQual', 'BsmtCond'] # variables to re-group
)  
																									 
rare_encoder.fit(X_train)
X_train = rare_encoder.transform(X_train)
X_test = rare_encoder.transform(X_test)

rare_encoder.variables
# the encoder_dict_ is a dictionary of variable: frequent labels pair
rare_encoder.encoder_dict_


----------DISCRETISATION---------
----Equal width discretisation
# with Scikit Learn
from sklearn.preprocessing import KBinsDiscretizer
disc = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform')
disc.fit(X_train[['age', 'fare']])

train_t = disc.transform(X_train[['age', 'fare']])
train_t = pd.DataFrame(train_t, columns = ['age', 'fare'])

test_t = disc.transform(X_test[['age', 'fare']])
test_t = pd.DataFrame(test_t, columns = ['age', 'fare'])

#Visualisation

t1 = train_t.groupby(['age'])['age'].count() / len(train_t)
t2 = test_t.groupby(['age'])['age'].count() / len(test_t)

tmp = pd.concat([t1, t2], axis=1)
tmp.columns = ['train', 'test']
tmp.plot.bar()
plt.xticks(rotation=0)
plt.ylabel('Number of observations per bin')

#####


---#with Feature-Engine
disc = EqualWidthDiscretiser(bins=10, variables = ['age', 'fare'])
disc.fit(X_train)
train_t = disc.transform(X_train)
test_t = disc.transform(X_test)



-----Equal-Frequencz discrtisation
--#with Feature-Engine
disc = EqualFrequencyDiscretiser(q=10, variables = ['age', 'fare'])
disc.fit(X_train)
train_t = disc.transform(X_train)
test_t = disc.transform(X_test)

disc.binner_dict_

--#witg Scikit Learn
disc = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')
disc.fit(X_train[['age', 'fare']])
train_t = disc.transform(X_train[['age', 'fare']])
train_t = pd.DataFrame(train_t, columns = ['age', 'fare'])

disc.bin_edges_


----K-means discretisation
# - outliers may influence the centroid
# good to combine with categorical encoding

from sklearn.preprocessing import KBinsDiscretizer
disc = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='kmeans')
disc.fit(X_train[['age', 'fare']])
train_t = disc.transform(X_train[['age', 'fare']])
train_t = pd.DataFrame(train_t, columns = ['age', 'fare'])
train_t.head()

disc.bin_edges_




