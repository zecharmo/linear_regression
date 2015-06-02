
import pandas as pd
import statsmodels.api as sm
import numpy as np

# retrieve data 
loansData = pd.read_csv('https://spark-public.s3.amazonaws.com/dataanalysis/loansData.csv')

# convert Interest Rate from a percent to a floating point number
loansData['Interest.Rate'] = loansData['Interest.Rate'].map(lambda x: round(float(x.rstrip('%')) / 100, 4))

# convert Loan Length to an integer
loansData['Loan.Length'] = loansData['Loan.Length'].map(lambda x: int(x.rstrip(' months')))

# convert FICO Range from a range to a single score
cleanFICORange = loansData['FICO.Range'].map(lambda x: x.split('-'))
cleanFICORange = cleanFICORange.map(lambda x: [int(n) for n in x])
cleanFICORange = cleanFICORange.map(lambda x: x.pop(0))

# save to a new column FICO Score
loansData['FICO.Score'] = cleanFICORange

# extract columns to use in the linear model
intrate = loansData['Interest.Rate']
loanamt = loansData['Amount.Requested']
fico = loansData['FICO.Score']

# reshape the data from the extracted columns
# use the Interest Rate as the dependent variable
y = np.matrix(intrate).transpose()

# use the FICO Score and Loan Amount as independent variables shaped as columns
x1 = np.matrix(fico).transpose()
x2 = np.matrix(loanamt).transpose()

# put the columns together to create an input matrix with one column per independent variable
x = np.column_stack([x1,x2])

# create a linear model
X = sm.add_constant(x)
model = sm.OLS(y,X)
f = model.fit()

# output the results summary
f.summary()