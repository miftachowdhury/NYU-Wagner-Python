import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


#stop and frisk data downloaded from 'https://www1.nyc.gov/site/nypd/stats/reports-analysis/stopfrisk.page'
#data contained in 14 csv files and 1 xlsx file

#name folder containing stop and frisk data files
import glob
from pathlib import Path
sf_data = Path("C:/Users/Mifta/Documents/NYU Wagner/Python/StatisticsHW/stopfrisk/NYPD Stop Frisk Data 2003_2017")

#check encoding for all files in sf_data, to address encoding concerns in assignment instructions
    #COMMENTED CODE# for currentfile in sf_data.iterdir(): 
    #COMMENTED CODE#    with open(currentfile) as data: print(data)
#encoding for all files in sf_data is 'cp1252'

#csv files 2003-2016: create a dataframe by using a for loop to read in columns of interest from csv files
sf_df=pd.DataFrame()
for f in sf_data.glob('*.csv'):
    df1 = pd.concat([pd.read_csv(f, usecols=['year', 'arstmade', 'race'], dtype='unicode', encoding='cp1252')])
    sf_df = sf_df.append(df1, ignore_index=True)

#xlsx file 2017: create a dataframe by reading in columns of interest from xlsx file
df17 = pd.read_excel(sf_data/'sqf-2017.xlsx', names=('year', 'arstmade', 'race'), usecols="D, W, BN", dtype='unicode', encoding='cp1252')

#recode 2017 race values to match 2003-2016 race values
def race_recode(race):
    if race=='ASIAN/PAC.ISL': new_race='A'
    elif race=='BLACK': new_race='B'
    elif race=='AMER IND': new_race ='I'
    elif race=='BLACK HISPANIC': new_race='P'
    elif race=='WHITE HISPANIC': new_race='Q'
    elif race=='WHITE': new_race='W'
    elif race=='(null)': new_race='X'
    elif race=='MALE': new_race='X' #data entry error led to instances of 'MALE' for race
    else: new_race ='Z'
    return new_race
df17['race']=df17['race'].apply(race_recode)

#concatenate csv df (03-16) and xlsx df(17) to create one dataframe
sf_df=pd.concat([sf_df, df17], ignore_index=True)

#recode black hispanic to black for purposes of this analysis
sf_df['race']=sf_df['race'].where(sf_df['race']!='P', 'B')

#replace empty cells with NA and check how many NAs there are in the dataframe
sf_df.replace(' ', np.nan, inplace=True)
#COMMENTED CODE# print(sf_df.isna().sum())
#NAs - year: 1; arstmade: 2; race: 165

#fill race NAs with X to represent "unknown" per codebook and drop remaining NAs
sf_df['race']=sf_df['race'].fillna('X')
sf_df.dropna(inplace=True)

#descriptives
#check number of rows in dataframe
print('There were', '{:,}'.format(len(sf_df)), 'stops between 2003 and 2017, inclusive.')

df_black = sf_df[sf_df['race']=='B']

#some observations out of curiosity below (can ignore for purposes of this task):
#print(np.count_nonzero(np.where(df_black['arstmade']=='N'))) #2,762,927 stops of black people that did not lead to arrest
#print(np.count_nonzero(df_black['arstmade'])) #2,942,275 stops of black people total
#print(np.count_nonzero(np.where(sf_df['arstmade']=='N'))) #4,766,473 stops did not lead to arrest
#print(np.count_nonzero(sf_df['arstmade'])) #5,076,775 stops total
#check percent of stops that did not lead to arrest, for black individuals only
#print('Out of', '{:,}'.format(np.count_nonzero(df_black['arstmade'])), 'stops of black people,', '{:,}'.format(np.count_nonzero(np.where(df_black['arstmade']=='N'))), 'or around','{0:.2f}%'.format((np.count_nonzero(np.where(df_black['arstmade']=='N'))/np.count_nonzero(df_black['arstmade']))*100), 'did not result in an arrest.')

#check percent of all stops that did not lead to arrest
print('Out of', '{:,}'.format(np.count_nonzero(sf_df['arstmade'])), 'stops,', '{:,}'.format(np.count_nonzero(np.where(sf_df['arstmade']=='N'))), 'or around','{0:.2f}%'.format((np.count_nonzero(np.where(sf_df['arstmade']=='N'))/np.count_nonzero(sf_df['arstmade']))*100), 'did not result in an arrest.')

df_innocent = sf_df[sf_df['arstmade']=='N']

#check percent of stops where suspect was black
print('Around','{0:.2f}%'.format((np.count_nonzero(np.where(sf_df['race']=='B'))/np.count_nonzero(sf_df['race']))*100), 'of all stops were of a black person.')

#check percent of innocent stops where suspect was black
print('Around','{0:.2f}%'.format((np.count_nonzero(np.where(df_innocent['race']=='B'))/np.count_nonzero(df_innocent['race']))*100), 'of innocent stops were of a black person.')

#from https://www1.nyc.gov/assets/planning/download/office/data-maps/nyc-population/census2010/t_sf1_dp_nyc.xlsx row 86, black population in NYC is ~25.5%
print('According to the 2010 census, as reported by the NYC Dept. of City Planning, the population of NYC is ~25.5% black.')


#QUESTION: Of stops that did not result in an arrest, is the person significantly more likely to be black than would be expected based on the demographics of NYC?

#The statistic: Proportion of the population of innocent stops who are black: a sample proportion of 0.5797 (57.97%)

#Method 1: Simulate a random distribution around the 25.5% black population in NYC using total number of stops as the sample size
#and then plot the 57.96% statistic on a histogram of the distribution

black_pop_nyc = [0.255, 0.745] #See above; according to the 2010 census, as reported by NYC DCP, NYC is 25.5% black

def sample_proportions(sample_size, probabilities):
    return np.random.multinomial(sample_size, probabilities) / sample_size

counts = []

for i in range(1000):
    counts.append(100*sample_proportions(5076775, black_pop_nyc)[0])

def setUpPlot():    
    plt.style.use('seaborn-white')
    plt.figure(figsize=(20,9))
    ax = plt.subplot(1,1,1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
setUpPlot()
plt.hist(counts, alpha = 0.25);
plt.scatter(57.97, 0, color='red', s=30);


print('Based on the plot, our test statistic clearly does not fall in the distribution, suggesting that the stops are not random.')


#QUESTION: Of stops that did not result in an arrest, is the person significantly more likely to be black than would be expected based on the demographics of NYC?

#The statistic: Proportion of the population of innocent stops who are black: a sample proportion of 0.5797 (57.97%)

#Method 2: Run a hypothesis test, using data from the innocent stop dataframe, and 2010 Census data from NYC DCP website

from statsmodels.stats.proportion import proportions_ztest
count = np.count_nonzero(np.where(df_innocent['race']=='B'))
nobs = np.count_nonzero(df_innocent['race'])
value = 0.255
stat, pval = proportions_ztest(count, nobs, value)
print('{0:0.3f}'.format(pval))

#Confirmed that the above works because if you change 'value' to 0.58 i.e. closer to the statistic, the p-value starts to increase

print('Based on the p-value of 0.000 from the test, we can reject the null hypothesis, suggesting that the stops are not random.\nThis finding is true at an alpha level of 0.05 or 0.01, and alphas much lower.')
