------FEATURE ENGINEERING STEPS
---1. Analyse variable types and values and characteristics
# Data types - # make list of variables  types
data.dtypes

# numerical: discrete vs continuous - SELECTION
discrete = [var for var in data.columns if data[var].dtype!='O' and var!='survived' and data[var].nunique()<10]
#OR 
''''for var in numerical:
    if len(data[var].unique()) < 20 and var not in year_vars:
        print(var, ' values: ', data[var].unique())
        discrete.append(var)
print()
print('There are {} discrete variables'.format(len(discrete)))''''

continuous = [var for var in data.columns if data[var].dtype!='O' and var!='survived' and var not in discrete]
#OR
'''numerical = [var for var in numerical if var not in discrete and var not in [
    'Id', 'SalePrice'] and var not in year_vars]

print('There are {} numerical and continuous variables'.format(len(numerical)))'''


# mixed
mixed = ['cabin']

# categorical
categorical = [var for var in data.columns if data[var].dtype=='O' and var not in mixed]


# Missing Data
data.isnull().mean()
# Cardinality
data[categorical+mixed].nunique()
# Outliers
data[continuous].boxplot(figsize=(10,4))
# outliers in discrete
data[discrete].boxplot(figsize=(10,4))
# feature magnitude
data.describe()



# Separating mixed values
data['cabin_num'] = data['cabin'].str.extract('(\d+)') # captures numerical part
data['cabin_num'] = data['cabin_num'].astype('float')
data['cabin_cat'] = data['cabin'].str[0] # captures the first letter

# show dataframe
data.head()

# drop original mixed
data.drop(['cabin'], axis=1, inplace=True)

-----HERE SPLITTING IN TRAIN AND TEST

# MIssing Data
# numerical
X_train.select_dtypes(exclude='O').isnull().mean()
# categorical
X_train.select_dtypes(include='O').isnull().mean()

# Cardinality and rare labels
# check cardinality again
X_train[['cabin_cat', 'sex', 'embarked']].nunique()
# check variable frequency
var = 'cabin_cat'
(X_train[var].value_counts() / len(X_train)).sort_values()


# Variables Distribution
X_train.select_dtypes(exclude='O').hist(bins=30, figsize=(8,8))
plt.show()


---2. BUILDING THE PIPLINE

titanic_pipe = Pipeline([

    # missing data imputation - section 4
    ('imputer_num',
     mdi.ArbitraryNumberImputer(arbitrary_number=-1,
                                variables=['age', 'fare', 'cabin_num'])),
    ('imputer_cat',
     mdi.CategoricalVariableImputer(variables=['embarked', 'cabin_cat'])),

    # categorical encoding - section 6
    ('encoder_rare_label',
     ce.RareLabelCategoricalEncoder(tol=0.01,
                                    n_categories=6,
                                    variables=['cabin_cat'])),
    ('categorical_encoder',
     ce.OrdinalCategoricalEncoder(encoding_method='ordered',
                                  variables=['cabin_cat', 'sex', 'embarked'])),

    # Gradient Boosted machine
    ('gbm', GradientBoostingClassifier(random_state=0))
])


# let's fit the pipeline and make predictions
titanic_pipe.fit(X_train, y_train)

X_train_preds = titanic_pipe.predict_proba(X_train)[:,1]
X_test_preds = titanic_pipe.predict_proba(X_test)[:,1]


# let's explore the importance of the features
importance = pd.Series(titanic_pipe.named_steps['gbm'].feature_importances_)
importance.index = data.drop('survived', axis=1).columns
importance.sort_values(inplace=True, ascending=False)
importance.plot.bar(figsize=(12,6))


--USING GRIDSEARCH IN THE PIPELINE
param_grid = {
    # try different feature engineering parameters
    'imputer_num__arbitrary_number': [-1, 99],
    'encoder_rare_label__tol': [0.1, 0.2],
    'categorical_encoder__encoding_method': ['ordered', 'arbitrary'],
    
    # try different gradient boosted tree model paramenters
    'gbm__max_depth': [None, 1, 3],
}

# now we set up the grid search with cross-validation
grid_search = GridSearchCV(titanic_pipe, param_grid,
                           cv=5, iid=False, n_jobs=-1, scoring='roc_auc')

# and now we train over all the possible combinations of the parameters above
grid_search.fit(X_train, y_train)

# and we print the best score over the train set
print(("best roc-auc from grid search: %.3f"
       % grid_search.score(X_train, y_train)))

# we can print the best estimator parameters like this
grid_search.best_estimator_
# and find the best fit parameters like this
grid_search.best_params_
# here we can see all the combinations evaluated during the gridsearch
grid_search.cv_results_['params']
# and here the scores for each of one of the above combinations
grid_search.cv_results_['mean_test_score']
print(("best linear regression from grid search: %.3f"
       % grid_search.score(X_test, y_test)))





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

# Extracting week of year from date, varies from 1 to 52
data['issue_dt_week'] = data['issue_dt'].dt.week

# Extracting month from date - 1 to 12
data['issue_dt_month'] = data['issue_dt'].dt.month

# Extract quarter from date variable - 1 to 4
data['issue_dt_quarter'] = data['issue_dt'].dt.quarter

# extract year 
data['issue_dt_year'] = data['issue_dt'].dt.year

# we can extract the day in different formats

# day of the week - name
data['issue_dt_dayofweek'] = data['issue_dt'].dt.weekday_name

# was the application done on the weekend?
data['issue_dt_is_weekend'] = np.where(data['issue_dt_dayofweek'].isin(['Sunday', 'Saturday']), 1,0)

# Extract the time elapsed
data['issue_dt'] - data['last_pymnt_dt']
(data['last_pymnt_dt'] - data['issue_dt']).dt.days.head()

# or the time difference to today
(datetime.datetime.today() - data['issue_dt']).head()

# calculate number of months passed between 2 dates
data['months_passed'] = (data['last_pymnt_dt'] - data['issue_dt']) / np.timedelta64(1, 'M')
data['months_passed'] = np.round(data['months_passed'],0)



#Pivot into a DataFrame
data.groupby(['issue_dt', 'grade'])['loan_amnt'].sum().unstack()

----MISSING DATA
data.isnull().sum()
data.isnull().mean()
data.emp_length.value_counts() / len(data)  #to find other sorts of indication for missing data (e.g ?) 

#Print features with missing values
# examine percentage of missing values
for col in numerical+year_vars:
    if X_train[col].isnull().mean() > 0:
        print(col, X_train[col].isnull().mean())
	

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
X_train_droped_1 = pd.DataFrame(X_train_droped, columns=categorical_dropped+continuous+discrete)
X_test_droped_1 = pd.DataFrame(X_test_droped, columns=categorical_dropped+continuous+discrete)
# To set in the initial Order
X_train_droped_2 = X_train_droped_1[columns_after_dropping]



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

# Identification
# plot number of categories per categorical variable
data[categorical].nunique().plot.bar(figsize=(10,6))
plt.title('CARDINALITY: Number of categories in categorical variables')
plt.xlabel('Categorical variables')
plt.ylabel('Number of different categories')

#Explore original relationship between categorical variables and target
for var in ['Neighborhood', 'Exterior1st', 'Exterior2nd']:
    
    fig = plt.figure()
    fig = X_train.groupby([var])['SalePrice'].mean().plot()
    fig.set_title('Relationship between {} and SalePrice'.format(var))
    fig.set_ylabel('Mean SalePrice')
    plt.show()


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
 #---IF Y is already separated of X
 #    ordered_labels=pd.concat((X_train_droped, y_train), axis=1).groupby([var
 #	])[target].mean().sort_values().index

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
# Check for rare labels
for col in cols:
    
    print(X_train
	  .groupby(col)[col].count() / len(X_train)) # frequency
    print()

#OR
multi_cat_cols = []
for col in X_train.columns:
    if X_train[col].dtypes =='O': # if variable  is categorical
        if X_train[col].nunique() > 10: # and has more than 10 categories
            multi_cat_cols.append(col)  # add to the list        
            print(X_train.groupby(col)[col].count()/ len(X_train)) # and print the percentage of observations within each category
            print()
		


# Visualise Example
total_houses = len(df)

# for each categorical variable
for col in categorical:

    # count the number of houses per category
    # and divide by total houses

    # aka percentage of houses per category

    temp_df = pd.Series(df[col].value_counts() / total_houses)

    # make plot with the above percentages
    fig = temp_df.sort_values(ascending=False).plot.bar()
    fig.set_xlabel(col)

    # add a line at 5 % to flag the threshold for rare categories
    fig.axhline(y=0.05, color='red')
    fig.set_ylabel('Percentage of houses')
    plt.show()

# Checking non-visual
for col in cols:
    
    print(X_train.groupby(col)[col].count() / len(X_train)) # frequency
    print()


--Regrouping
-# With Pandas

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


-# With Feature-Engine
# Rare value encoder
rare_encoder = RareLabelCategoricalEncoder(
    tol=0.05,  # minimal percentage to be considered non-rare
    n_categories=4, # minimal number of categories the variable should have to re-cgroup rare categories
    variables=['Neighborhood', 'Exterior1st', 'Exterior2nd',
               'MasVnrType', 'ExterQual', 'BsmtCond'] # variables to re-group
)  




----OUTLIERS
# let's make boxplots to visualise outliers in the continuous variables 
# and histograms to get an idea of the distribution

for var in numerical:
    plt.figure(figsize=(6,4))
    plt.subplot(1, 2, 1)
    fig = data.boxplot(column=var)
    fig.set_title('')
    fig.set_ylabel(var)
    
    plt.subplot(1, 2, 2)
    fig = data[var].hist(bins=20)
    fig.set_ylabel('Number of houses')
    fig.set_xlabel(var)

    plt.show()

# OR
# function to create histogram, Q-Q plot and
# boxplot
def diagnostic_plots(df, variable):
    # function takes a dataframe (df) and
    # the variable of interest as arguments

    # define figure size
    plt.figure(figsize=(16, 4))

    # histogram
    plt.subplot(1, 3, 1)
    sns.distplot(df[variable], bins=30)
    plt.title('Histogram')

    # Q-Q plot
    plt.subplot(1, 3, 2)
    stats.probplot(df[variable], dist="norm", plot=pylab)
    plt.ylabel('RM quantiles')

    # boxplot
    plt.subplot(1, 3, 3)
    sns.boxplot(y=df[variable])
    plt.title('Boxplot')

    plt.show()


# outlies in discrete variables
for var in discrete:
    (data.groupby(var)[var].count() / np.float(len(data))).plot.bar()
    plt.ylabel('Percentage of observations per label')
    plt.title(var)
    plt.show()


	
#OUTLIERS DETECTION
# function to find upper and lower boundaries for normally distributed variables

def find_normal_boundaries(df, variable):
    # calculate the boundaries outside which sit the outliers
    # for a Gaussian distribution

    upper_boundary = df[variable].mean() + 3 * df[variable].std()
    lower_boundary = df[variable].mean() - 3 * df[variable].std()

    return upper_boundary, lower_boundary

# For skewed variables
def find_skewed_boundaries(df, variable, distance):

    # Let's calculate the boundaries outside which sit the outliers
    # for skewed distributions

    # distance passed as an argument, gives us the option to
    # estimate 1.5 times or 3 times the IQR to calculate
    # the boundaries.

    IQR = df[variable].quantile(0.75) - df[variable].quantile(0.25)

    lower_boundary = df[variable].quantile(0.25) - (IQR * distance)
    upper_boundary = df[variable].quantile(0.75) + (IQR * distance)

    return upper_boundary, lower_boundary

upper_boundary, lower_boundary = find_skewed_boundaries(boston, 'LSTAT', 1.5)
upper_boundary, lower_boundary


# Capping Quantiles
def find_boundaries(df, variable):

    # the boundaries are the quantiles

    lower_boundary = df[variable].quantile(0.05)
    upper_boundary = df[variable].quantile(0.95)

    return upper_boundary, lower_boundary


# Printing the outliers
for var in df[continuous]:
    # define figure size
    print('For feature: ' + str(var))
    print('total: {}'.format(len(df)))  
    
    upper_boundary, lower_boundary = find_skewed_boundaries(df, var, 1.5)
    print('over the upper bound: {}'.format(
        round(len(df[df[var] > upper_boundary])/len(df), 2)))
    print()
    print('under the lower bound: {}'.format(
        round(len(df[df[var] < lower_boundary])/len(df), 2)))

	
# CAPPING
oston['RM']= np.where(boston['RM'] > RM_upper_limit, RM_upper_limit,
                       np.where(boston['RM'] < RM_lower_limit, RM_lower_limit, boston['RM']))
	
#My function
for var in df[continuous]:
    # define figure size
    print('For feature: ' + str(var))
    print('total: {}'.format(len(df)))  
    
    upper_boundary, lower_boundary = find_skewed_boundaries(df, var, 1.5)
    print('over the upper bound: {}'.format(
        round(len(df[df[var] > upper_boundary])/len(df), 2)))
    print()
    print('under the lower bound: {}'.format(
        round(len(df[df[var] < lower_boundary])/len(df), 2)))
    
    df[var]= np.where(df[var] > upper_boundary, upper_boundary,
                       np.where(df[var] < lower_boundary, lower_boundary, df[var]))
	
	
	
	
	
	
	
	


---# Transforming						   
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


----Monotonic Encoding with discretisation
# set up the equal frequency discretiser
# to encode variables we need them returned as objects for feature-engine
disc = EqualFrequencyDiscretiser(
    q=10, variables=['age', 'fare'], return_object=True)
# find the intervals
disc.fit(X_train)
# transform train and text
train_t = disc.transform(X_train)
test_t = disc.transform(X_test)

#Ordinal encoding
enc = OrdinalCategoricalEncoder(encoding_method = 'ordered')
enc.fit(train_t, y_train)
train_t = enc.transform(train_t)
test_t = enc.transform(test_t)




-----Descision Tree for discretisation
# - potentially leads to overfittng the feature
--#With ScikitLearn
tree_model = DecisionTreeClassifier(max_depth=3)
tree_model.fit(X_train['age'].to_frame(), y_train)
X_train['Age_tree'] = tree_model.predict_proba(X_train['age'].to_frame())[:,1]
X_train['Age_tree'].unique()

# let's see the Age limits buckets generated by the tree
# by capturing the minimum and maximum age per each probability bucket, 
# we get an idea of the bucket cut-offs
pd.concat( [X_train.groupby(['Age_tree'])['age'].min(),
            X_train.groupby(['Age_tree'])['age'].max()], axis=1)

X_test['Age_tree'] = tree_model.predict_proba(X_test['age'].to_frame())[:,1]

# monotonic relationship with target
pd.concat([X_test, y_test], axis=1).groupby(['Age_tree'])['survived'].mean().plot()
plt.title('Monotonic relationship between discretised Age and target')
plt.ylabel('Survived')


#Testing out different parameters
# Build trees of different depths, and calculate the roc-auc of each tree
# choose the depth that generates the best roc-auc

score_ls = []  # here we store the roc auc
score_std_ls = []  # here we store the standard deviation of the roc_auc

for tree_depth in [1, 2, 3, 4]:
    # call the model
    tree_model = DecisionTreeClassifier(max_depth=tree_depth)
    # train the model using 3 fold cross validation
    scores = cross_val_score(
        tree_model, X_train['age'].to_frame(), y_train, cv=3, scoring='roc_auc')
    # save the parameters
    score_ls.append(np.mean(scores))
    score_std_ls.append(np.std(scores))
    
# capture the parameters in a dataframe
temp = pd.concat([pd.Series([1, 2, 3, 4]), pd.Series(
    score_ls), pd.Series(score_std_ls)], axis=1)

temp.columns = ['depth', 'roc_auc_mean', 'roc_auc_std']
temp


---#With Feature-Engine
treeDisc = DecisionTreeDiscretiser(cv=10, scoring='accuracy',
                                   variables=['age', 'fare'],
                                   regression=False,
                                   param_grid={'max_depth': [1, 2, 3],
                                              'min_samples_leaf':[10,4]})

treeDisc.fit(X_train, y_train)

# we can inspect the best params
treeDisc.binner_dict_['age'].best_params_
treeDisc.scores_dict_['age']
# and the score
treeDisc.scores_dict_['fare']

train_t = treeDisc.transform(X_train)
test_t = treeDisc.transform(X_test)


------Scalling
1. Standardisation
# - sensitive to outliers
2. Mean Normalisation
# - sensitive to outliers
3. MinMaxScaller
# Mean and Variance varies - not set


------MIXES VARIABLES
# Numeric and categorical
1. Values are eigther numbers or strings
# extract numerical part
data['open_il_24m_numerical'] = pd.to_numeric(data["open_il_24m"],
                                              errors='coerce',
                                              downcast='integer')
# extract categorical part
data['open_il_24m_categorical'] = np.where(data['open_il_24m_numerical'].isnull(),
                                           data['open_il_24m'],
                                           np.nan)

2: the observations of the variable contain numbers and strings
# let's extract the numerical and categorical part for cabin
data['cabin_num'] = data['cabin'].str.extract('(\d+)') # captures numerical part
data['cabin_cat'] = data['cabin'].str[0] # captures the first letter

OR
# extract the last bit of ticket as number
data['ticket_num'] = data['ticket'].apply(lambda s: s.split()[-1])
data['ticket_num'] = pd.to_numeric(data['ticket_num'],
                                   errors='coerce',
                                   downcast='integer')

# extract the first part of ticket as category
data['ticket_cat'] = data['ticket'].apply(lambda s: s.split()[0])
data['ticket_cat'] = np.where(data['ticket_cat'].str.isdigit(), np.nan,
                              data['ticket_cat'])


# Capture variables that contains certain Characters / Letters
# list of variables that contain year information
year_vars = [var for var in numerical if 'Yr' in var or 'Year' in var]

# function to calculate elapsed time
def elapsed_years(df, var):
    # capture difference between year variable and
    # year the house was sold
    
    df[var] = df['YrSold'] - df[var]
    return df

for var in ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt']:
    X_train = elapsed_years(X_train, var)
    X_test = elapsed_years(X_test, var)



