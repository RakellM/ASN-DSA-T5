# %%
# LIBRARY
##################################
import pandas as pd

# %%
# DATA
##################################
df = pd.read_sas("../data/titanic.sas7bdat")

# %%
#- Group by Class and sum Survival
survival_by_class = df.groupby('Class')['Survival'].sum()
print(survival_by_class)

# %%
## Contingency Table
##################################

### Class = 1
#- Create a binary variable: 1 if Class == 1, 0 otherwise
df['Class1'] = (df['Class'] == 1).astype(int)

#- Create a 2x2 contingency table: Class1 vs Survival
table_c1 = pd.crosstab(df['Class1'], df['Survival'])
print(table_c1)

# Variables
Y = 'Survival'
X = 'Class'

# %%
#### Probability of Survivel given Class = 1
#- P(Y = Survive | X = Class1) = ? = PSC1
#- P(Y = Survive & X = Class1) = ? = PSnC1
#- P(X = Class1) = ? = PC1
PSnC1 = table_c1.iloc[1,1]
PC1 = table_c1.iloc[1,0] + table_c1.iloc[1,1]

PSC1 = round(PSnC1 / PC1 , 2)
PSC1_pct = PSC1 * 100

x_nbr = 1
print(f"The probability of {Y} from {X} {x_nbr} is {PSC1} or {PSC1_pct:.0f}%.")

# %%
#### Probability of non Survivel given Class = 1
#- P(Y = Not Survive | X = Class1) = ? = PS0C1
#- P(Y = Not Survive & X = Class1) = ? = PS0nC1
#- P(X = Class1) = ? = PC1
PS0nC1 = table_c1.iloc[1,0]
PC1 = table_c1.iloc[1,0] + table_c1.iloc[1,1]

PS0C1 = round(PS0nC1 / PC1 , 2)
PS0C1_pct = PS0C1 * 100

x_nbr = 1
print(f"The probability of not {Y} from {X} {x_nbr} is {PS0C1} or {PS0C1_pct:.0f}%.")

# %%
#### Chance
#- Chance(Y = S1 | X = C1) = P(Y = S1 | X = C1) / P(Y = S0 | X = C1)
chance_S1C1 = PSC1 / PS0C1

chance_S1C1_f = round(chance_S1C1 - 1, 2)
chance_S1C1_pct = chance_S1C1_f * 100

if chance_S1C1_f >= 0:
    text1 = 'greater'
else:
    text1 = 'lower'

x_nbr = 1
print(f"The probability of {Y} from {X} {x_nbr} is {abs(chance_S1C1_f)} times {text1} than not {Y}.")
print(f"The probability of {Y} from {X} {x_nbr} is {abs(chance_S1C1_pct):.0f}% {text1} than not {Y}.")


# %%
df['Class2'] = (df['Class'] == 2).astype(int)

table_c2 = pd.crosstab(df['Class2'], df['Survival'])
print(table_c2)

# %%
#### Probability of Survivel given Class = 2
#- P(Y = Survive | X = Class2) = ? = PSC2
#- P(Y = Survive & X = Class2) = ? = PSnC2
#- P(X = Class2) = ? = PC2
PSnC2 = table_c2.iloc[1,1]
PC2 = table_c2.iloc[1,0] + table_c2.iloc[1,1]

PSC2 = round(PSnC2 / PC2 , 2)
PSC2_pct = PSC2 * 100

x_nbr = 2
print(f"The probability of {Y} from {X} {x_nbr} is {PSC2} or {PSC2_pct:.0f}%.")

# %%
#### Probability of non Survivel given Class = 2
#- P(Y = Not Survive | X = Class2) = ? = PS0C2
#- P(Y = Not Survive & X = Class2) = ? = PS0nC2
#- P(X = Class2) = ? = PC2
PS0nC2 = table_c2.iloc[1,0]
PC2 = table_c2.iloc[1,0] + table_c2.iloc[1,1]

PS0C2 = round(PS0nC2 / PC2 , 2)
PS0C2_pct = PS0C2 * 100

x_nbr = 2
print(f"The probability of not {Y} from {X} {x_nbr} is {PS0C2} or {PS0C2_pct:.0f}%.")

# %%
#### Chance
#- Chance(Y = S1 | X = C2) = P(Y = S1 | X = C2) / P(Y = S0 | X = C2)
chance_S1C2 = PSC2 / PS0C2

chance_S1C2_f = round(chance_S1C2 - 1, 2)
chance_S1C2_pct = chance_S1C2_f * 100

if chance_S1C2_f >= 0:
    text1 = 'greater'
else:
    text1 = 'lower'

x_nbr = 2
print(f"The probability of {Y} from {X} {x_nbr} is {abs(chance_S1C2_f)} times {text1} than not {Y}.")
print(f"The probability of {Y} from {X} {x_nbr} is {abs(chance_S1C2_pct):.0f}% {text1} than not {Y}.")


# %%
df['Class3'] = (df['Class'] == 3).astype(int)

table_c3 = pd.crosstab(df['Class3'], df['Survival'])
print(table_c3)

# %%
#### Probability of Survivel given Class = 3
#- P(Y = Survive | X = Class3) = ? = PSC3
#- P(Y = Survive & X = Class3) = ? = PSnC3
#- P(X = Class3) = ? = PC3
PSnC3 = table_c3.iloc[1,1]
PC3 = table_c3.iloc[1,0] + table_c3.iloc[1,1]

PSC3 = round(PSnC3 / PC3 , 2)
PSC3_pct = PSC3* 100

x_nbr = 3
print(f"The probability of {Y} from {X} {x_nbr} is {PSC3} or {PSC3_pct:.0f}%.")

# %%
#### Probability of non Survivel given Class = 3
#- P(Y = Not Survive | X = Class3) = ? = PS0C3
#- P(Y = Not Survive & X = Class3) = ? = PS0nC3
#- P(X = Class2) = ? = PC2
PS0nC3 = table_c3.iloc[1,0]
PC3 = table_c3.iloc[1,0] + table_c3.iloc[1,1]

PS0C3 = round(PS0nC3 / PC3 , 2)
PS0C3_pct = PS0C3 * 100

x_nbr = 3
print(f"The probability of not {Y} from {X} {x_nbr} is {PS0C3} or {PS0C3_pct:.0f}%.")

# %%
#### Chance
#- Chance(Y = S1 | X = C3) = P(Y = S1 | X = C3) / P(Y = S0 | X = C3)
chance_S1C3 = PSC3 / PS0C3

chance_S1C3_f = round(chance_S1C3 - 1, 2)
chance_S1C3_pct = chance_S1C3_f * 100

if chance_S1C3_f >= 0:
    text1 = 'greater'
else:
    text1 = 'lower'

x_nbr = 3
print(f"The probability of {Y} from {X} {x_nbr} is {abs(chance_S1C3_f)} times {text1} than not {Y}.")
print(f"The probability of {Y} from {X} {x_nbr} is {abs(chance_S1C3_pct):.0f}% {text1} than not {Y}.")


# %%
### Odds Ratio
#- Odds Ratio between Class 1 & 3 survive = OR13
OR13 = chance_S1C1 / chance_S1C3

OR13_f = round(OR13 - 1, 2)
OR13_pct = OR13_f * 100

if OR13_f >= 0:
    text1 = 'greater'
else:
    text1 = 'lower'

x_nbr1 = 1
x_nbr2 = 3

print(f"The chance of {Y} from {X} {x_nbr1} is {abs(OR13_f)} times {text1} than be {X} {x_nbr2}.")
print(f"The chance of {Y} from {X} {x_nbr1} is {abs(OR13_pct):.0f}% {text1} than be {X} {x_nbr2}.")

# %%
#- Odds Ratio between Class 3 & 1 survive = OR31
OR31 = chance_S1C3 / chance_S1C1

OR31_f = round(OR31 - 1, 2)
OR31_pct = OR31_f * 100

if OR31_f >= 0:
    text1 = 'greater'
else:
    text1 = 'lower'

x_nbr1 = 3
x_nbr2 = 1

print(f"The chance of {Y} from {X} {x_nbr1} is {abs(OR31_f)} times {text1} than be {X} {x_nbr2}.")
print(f"The chance of {Y} from {X} {x_nbr1} is {abs(OR31_pct):.0f}% {text1} than be {X} {x_nbr2}.")

# %%
