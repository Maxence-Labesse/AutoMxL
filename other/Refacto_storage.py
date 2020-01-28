"""

print('\nTOP 10 variables avec variance les plus basses :')
print(df_bis[var_list].var().sort_values().reset_index().rename(columns={'index': 'variable', 0: 'variance'}).head(10))












"""