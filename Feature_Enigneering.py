
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
                                                   
                                                   
                                                   
