from Panel import *

df = pd.read_csv('Data/grunfeld.csv')

p = Panel(df, 'FIRM', 'YEAR', False)

print(p.xtreg('I ~ F + C', type = 'pooled', robust_se='HC0').summary())