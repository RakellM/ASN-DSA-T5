# %%
## LIBRARY
import os
import kaggle
import pandas as pd


# %%
## PATH & OTHERS
# Project Directory
project_dir = os.path.join(os.path.expanduser("~"), "OneDrive", "Project_Code", 
                           "ASN-DSA-T5", "19-RML", "Homework")

# Data path
data_raw_path = os.path.join(project_dir, "data", "raw")
os.makedirs(data_raw_path, exist_ok=True)

data_processed_path = os.path.join(project_dir, "data", "processed")
os.makedirs(data_processed_path, exist_ok=True)

data_srcA_path = os.path.join(project_dir, "src", "analytics")
os.makedirs(data_srcA_path, exist_ok=True)

data_srcE_path = os.path.join(project_dir, "src", "EDA")
os.makedirs(data_srcE_path, exist_ok=True)


# %%
## IMPORT DATA
# Download data from Kaggle
kaggle.api.authenticate()
kaggle.api.dataset_download_files('thedevastator/higher-education-predictors-of-student-retention', 
                                  path=data_raw_path, 
                                  unzip=True)


# %%
# Import data toa  dataframe
df_dataset = pd.read_csv(os.path.join(project_dir, "data/raw/", "dataset.csv"))


# %%
## EDA 1
# Check variables
df_dataset.info()

df_dataset.shape

df_dataset.columns


# %%
# First 10 rows of the data
df_dataset.head(10)

# %%
# Check nulls
df_dataset.isna().sum()


# %%
# Rename columns
df = df_dataset.copy()

df = df.rename(columns={
    'Marital status' : "marital_status", 
    'Application mode' : "application_mode", 
    'Application order' : "application_order", 
    'Course' : "course",
    'Daytime/evening attendance' : "attendence_daytime", 
    'Previous qualification' : "qualification_previous", 
    'Nacionality' : "nationality",
    "Mother's qualification" : "qualification_mother", 
    "Father's qualification" : "qualification_father",
    "Mother's occupation" : "occupation_mother",
    "Father's occupation" : "occupation_father", 
    'Displaced' : "displaced_yes",
    'Educational special needs' : "special_needs_yes", 
    'Debtor' : "debtor_yes", 
    'Tuition fees up to date' : "tuition_fees_up_to_date",
    'Gender' : "gender_male", 
    'Scholarship holder' : "scholarship_holder", 
    'Age at enrollment' : "age_enrollment", 
    'International' : "international_student_yes",
    'Curricular units 1st sem (credited)' : "curricular_units_1_credited" ,
    'Curricular units 1st sem (enrolled)' : "curricular_units_1_enrolled" ,
    'Curricular units 1st sem (evaluations)' : "curricular_units_1_evaluations" ,
    'Curricular units 1st sem (approved)' : "curricular_units_1_approved" ,
    'Curricular units 1st sem (grade)' : "curricular_units_1_grade" ,
    'Curricular units 1st sem (without evaluations)' : "curricular_units_1_without_evaluations" ,
    'Curricular units 2nd sem (credited)' : "curricular_units_2_credited" ,
    'Curricular units 2nd sem (enrolled)' : "curricular_units_2_enrolled" ,
    'Curricular units 2nd sem (evaluations)' : "curricular_units_2_evaluations" ,
    'Curricular units 2nd sem (approved)' : "curricular_units_2_approved" ,
    'Curricular units 2nd sem (grade)' : "curricular_units_2_grade" ,
    'Curricular units 2nd sem (without evaluations)' : "curricular_units_2_without_evaluations" , 
    'Unemployment rate' : "umemployment_rate",
    'Inflation rate' : "inflation_rate", 
    'GDP' : "GDP", 
    'Target' : "target"
})

# %%
### Univariate Analysis
#### Variable : target

# Unique categories
df['target'].unique()

# Category frequency count
freq_target = df.groupby('target').size().reset_index(name='Count')
total_entries = len(df)
freq_target['Percentage'] = round((freq_target['Count'] / total_entries) * 100 , 2)

print(freq_target)


# %%
# Create Dropout column
df['Dropout'] = df["target"].apply(lambda x : 1 if x == "Dropout" else 0)

# Category frequency count
freq_target_status = df.groupby('Dropout').size().reset_index(name='Count')
total_entries = len(df)
freq_target_status['Percentage'] = round((freq_target_status['Count'] / total_entries) * 100 , 2)

print(freq_target_status)


# %%
#### Variable: application_mode
# Unique categories
print(df['application_mode'].unique())

# Category frequency count
freq_application_mode = df.groupby('application_mode').size().reset_index(name='Count')
total_entries = len(df)
freq_application_mode['Percentage'] = round((freq_application_mode['Count'] / total_entries) * 100 , 2)

freq_application_mode


# %%
#### Variable: application_order
# Unique categories
print(df['application_order'].unique())

# Category frequency count
freq_application_order = df.groupby('application_order').size().reset_index(name='Count')
total_entries = len(df)
freq_application_order['Percentage'] = round((freq_application_order['Count'] / total_entries) * 100 , 2)

freq_application_order

# %%
#### Variable: course
# Unique categories
print(df['course'].unique())

# Category frequency count
freq_course = df.groupby('course').size().reset_index(name='Count')
total_entries = len(df)
freq_course['Percentage'] = round((freq_course['Count'] / total_entries) * 100 , 2)

freq_course

# %%
#### Variable: attendence_daytime
# Unique categories
print(df['attendence_daytime'].unique())

# Category frequency count
freq_attendence_daytime = df.groupby('attendence_daytime').size().reset_index(name='Count')
total_entries = len(df)
freq_attendence_daytime['Percentage'] = round((freq_attendence_daytime['Count'] / total_entries) * 100 , 2)

freq_attendence_daytime

# %%
#### Variable: qualification_previous
# Unique categories
print(df['nationality'].unique())

# Category frequency count
freq_nationality = df.groupby('nationality').size().reset_index(name='Count')
total_entries = len(df)
freq_nationality['Percentage'] = round((freq_nationality['Count'] / total_entries) * 100 , 2)

freq_nationality


# %%
#### qualification_mother
# Unique categories
print(df['qualification_mother'].unique())

# Category frequency count
freq_qualification_mother = df.groupby('qualification_mother').size().reset_index(name='Count')
total_entries = len(df)
freq_qualification_mother['Percentage'] = round((freq_qualification_mother['Count'] / total_entries) * 100 , 2)

freq_qualification_mother


# %%
#### qualification_father
# Unique categories
print(df['qualification_father'].unique())

# Category frequency count
freq_qualification_father = df.groupby('qualification_father').size().reset_index(name='Count')
total_entries = len(df)
freq_qualification_father['Percentage'] = round((freq_qualification_father['Count'] / total_entries) * 100 , 2)

freq_qualification_father


# %%
#### occupation_mother
# Unique categories
print(df['occupation_mother'].unique())

# Category frequency count
freq_occupation_mother = df.groupby('occupation_mother').size().reset_index(name='Count')
total_entries = len(df)
freq_occupation_mother['Percentage'] = round((freq_occupation_mother['Count'] / total_entries) * 100 , 2)

freq_occupation_mother


# %%
#### occupation_father
# Unique categories
print(df['occupation_father'].unique())

# Category frequency count
freq_occupation_father = df.groupby('occupation_father').size().reset_index(name='Count')
total_entries = len(df)
freq_occupation_father['Percentage'] = round((freq_occupation_father['Count'] / total_entries) * 100 , 2)

freq_occupation_father


# %%
#### displaced_yes
# Unique categories
print(df['displaced_yes'].unique())

# Category frequency count
freq_displaced_yes = df.groupby('displaced_yes').size().reset_index(name='Count')
total_entries = len(df)
freq_displaced_yes['Percentage'] = round((freq_displaced_yes['Count'] / total_entries) * 100 , 2)

freq_displaced_yes


# %%
#### special_needs_yes
# Unique categories
print(df['special_needs_yes'].unique())

# Category frequency count
freq_special_needs_yes = df.groupby('special_needs_yes').size().reset_index(name='Count')
total_entries = len(df)
freq_special_needs_yes['Percentage'] = round((freq_special_needs_yes['Count'] / total_entries) * 100 , 2)

freq_special_needs_yes


# %%
#### debtor_yes
# Unique categories
print(df['debtor_yes'].unique())

# Category frequency count
freq_debtor_yes = df.groupby('debtor_yes').size().reset_index(name='Count')
total_entries = len(df)
freq_debtor_yes['Percentage'] = round((freq_debtor_yes['Count'] / total_entries) * 100 , 2)

freq_debtor_yes


# %%
#### tuition_fees_up_to_date
# Unique categories
print(df['tuition_fees_up_to_date'].unique())

# Category frequency count
freq_tuition_fees_up_to_date = df.groupby('tuition_fees_up_to_date').size().reset_index(name='Count')
total_entries = len(df)
freq_tuition_fees_up_to_date['Percentage'] = round((freq_tuition_fees_up_to_date['Count'] / total_entries) * 100 , 2)

freq_tuition_fees_up_to_date


# %%
#### gender_male
# Unique categories
print(df['gender_male'].unique())

# Category frequency count
freq_gender_male = df.groupby('gender_male').size().reset_index(name='Count')
total_entries = len(df)
freq_gender_male['Percentage'] = round((freq_gender_male['Count'] / total_entries) * 100 , 2)

freq_gender_male


# %%
#### scholarship_holder
# Unique categories
print(df['scholarship_holder'].unique())

# Category frequency count
freq_scholarship_holder = df.groupby('scholarship_holder').size().reset_index(name='Count')
total_entries = len(df)
freq_scholarship_holder['Percentage'] = round((freq_scholarship_holder['Count'] / total_entries) * 100 , 2)

freq_scholarship_holder


# %%
#### age_enrollment
# Descriptive Statistics
print(df['age_enrollment'].describe())


# %%
#### international_student_yes
# Unique categories
print(df['international_student_yes'].unique())

# Category frequency count
freq_international_student_yes = df.groupby('international_student_yes').size().reset_index(name='Count')
total_entries = len(df)
freq_international_student_yes['Percentage'] = round((freq_international_student_yes['Count'] / total_entries) * 100 , 2)

freq_international_student_yes


# %%
#### Variable: curricular_units
# Semester 1
sem1_columns = [
    'curricular_units_1_credited', 
    'curricular_units_1_enrolled', 
    'curricular_units_1_evaluations', 
    'curricular_units_1_approved', 
    'curricular_units_1_grade', 
    'curricular_units_1_without_evaluations'
]

df[sem1_columns].describe().T


# %%
# Semester 2
sem2_columns = [
    'curricular_units_2_credited', 
    'curricular_units_2_enrolled', 
    'curricular_units_2_evaluations', 
    'curricular_units_2_approved', 
    'curricular_units_2_grade', 
    'curricular_units_2_without_evaluations'
]

df[sem2_columns].describe().T


# %%
#### Variable: Macro economics
macro_econ_columns = [
    'umemployment_rate',
    'inflation_rate',
    'GDP'
]

df[macro_econ_columns].describe().T


# %%
### Transform Categorical variables into category
categorical_variables = [
    "marital_status", 
    "application_mode", 
    "application_order", 
    "course",
    "attendence_daytime", 
    "qualification_previous", 
    "nationality",
    "qualification_mother", 
    "qualification_father",
    "occupation_mother",
    "occupation_father", 
    "displaced_yes",
    "special_needs_yes", 
    "debtor_yes", 
    "tuition_fees_up_to_date",
    "gender_male", 
    "scholarship_holder", 
    "international_student_yes",
]

for var in categorical_variables:
    df[var] = df[var].astype('object')

# df.info()


# %%
# Drop target variable
df = df.drop('target', axis=1)

# %%
## Split between Train and Test
X = df.drop('Dropout', axis=1)
y = df['Dropout']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

print(f"train target distribution: {y_train.mean():0.4f}")
print(f" test target distribution: {y_test.mean():0.4f}")

