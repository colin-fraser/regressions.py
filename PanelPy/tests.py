from panel import *

df = pd.read_csv('../Data/grunfeld.csv')

p = Panel(df, 'FIRM', 'YEAR', False)

print(p.data)