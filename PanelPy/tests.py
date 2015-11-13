from Panel import *

df = pd.read_csv('../Data/grunfeld.csv')

p = Panel(df, 'FIRM', 'YEAR', False)

print(xtreg('I', ['F', 'C'], p, type='fe', robust_se='cluster').summary())