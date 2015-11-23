# Regressions in Python
This is a package for easily performing regression analysis in Python. All the heavy lifting is being done by [Pandas](http://pandas.pydata.org/) and [Statsmodels](http://statsmodels.sourceforge.net/); this is just an interface that should be familiar to anyone who has used Stata, with some funny implementation details that make the output a bit more like Stata output (i.e. the fixed-effects implementation has an "intercept" term). 

##Examples
```python
    from regressions import RDataFrame
    # RDataFrame subclasses the pandas DataFrame and attaches regression methods such 
    # as regress and xtreg.
    
    grunfeld = RDataFrame.from_csv('regressions/Data/grunfeld.csv')
    grunfeld.head()
    
    #    FIRM  YEAR      I       F      C
    # 0     1  1935  317.6  3078.5    2.8
    # 1     1  1936  391.8  4661.7   52.6
    # 2     1  1937  410.6  5387.1  156.9
    # 3     1  1938  257.7  2792.2  209.2
    # 4     1  1939  330.8  4313.2  203.4
    
    # Linear regression
    grunfeld.regress('I ~ F + C')
    
    #                                 OLS Regression Results                            
    # ==============================================================================
    # Dep. Variable:                      I   R-squared:                       0.812
    # Model:                            OLS   Adj. R-squared:                  0.811
    # Method:                 Least Squares   F-statistic:                     426.6
    # Date:                Mon, 23 Nov 2015   Prob (F-statistic):           2.58e-72
    # Time:                        10:22:00   Log-Likelihood:                -1191.8
    # No. Observations:                 200   AIC:                             2390.
    # Df Residuals:                     197   BIC:                             2399.
    # Df Model:                           2                                         
    # Covariance Type:            nonrobust                                         
    # ==============================================================================
    #                  coef    std err          t      P>|t|      [95.0% Conf. Int.]
    # ------------------------------------------------------------------------------
    # Intercept    -42.7144      9.512     -4.491      0.000       -61.472   -23.957
    # F              0.1156      0.006     19.803      0.000         0.104     0.127
    # C              0.2307      0.025      9.055      0.000         0.180     0.281
    # ==============================================================================
    # Omnibus:                       27.240   Durbin-Watson:                   0.358
    # Prob(Omnibus):                  0.000   Jarque-Bera (JB):               89.572
    # Skew:                           0.469   Prob(JB):                     3.55e-20
    # Kurtosis:                       6.141   Cond. No.                     2.46e+03
    # ==============================================================================
    # 
    # Warnings:
    # [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    # [2] The condition number is large, 2.46e+03. This might indicate that there are
    # strong multicollinearity or other numerical problems.
    
    #Cluster standard errors 
    grunfeld.regress('I ~ F + C', cluster='FIRM')
    
    #                                 OLS Regression Results                            
    # ==============================================================================
    # Dep. Variable:                      I   R-squared:                       0.812
    # Model:                            OLS   Adj. R-squared:                  0.811
    # Method:                 Least Squares   F-statistic:                     51.59
    # Date:                Mon, 23 Nov 2015   Prob (F-statistic):           1.17e-05
    # Time:                        10:23:50   Log-Likelihood:                -1191.8
    # No. Observations:                 200   AIC:                             2390.
    # Df Residuals:                     197   BIC:                             2399.
    # Df Model:                           2                                         
    # Covariance Type:              cluster                                         
    # ==============================================================================
    #                  coef    std err          z      P>|z|      [95.0% Conf. Int.]
    # ------------------------------------------------------------------------------
    # Intercept    -42.7144     20.425     -2.091      0.037       -82.747    -2.682
    # F              0.1156      0.016      7.271      0.000         0.084     0.147
    # C              0.2307      0.085      2.715      0.007         0.064     0.397
    # ==============================================================================
    # Omnibus:                       27.240   Durbin-Watson:                   0.358
    # Prob(Omnibus):                  0.000   Jarque-Bera (JB):               89.572
    # Skew:                           0.469   Prob(JB):                     3.55e-20
    # Kurtosis:                       6.141   Cond. No.                     2.46e+03
    # ==============================================================================
    
    # Warnings:
    # [1] Standard Errors are robust tocluster correlation (cluster)
    # [2] The condition number is large, 2.46e+03. This might indicate that there are
    # strong multicollinearity or other numerical problems.
    
    # Tell regressions.py that this is a panel
    grunfeld.xtset(i='FIRM', t='YEAR')
    
    # Balanced Panel.
    # attribute value
    #         i  FIRM
    #         t  YEAR
    #         n    10
    #         T    20
    #         N   200
    #     Width     5
    
    # Fixed-effects regression
    grunfeld.xtreg('I ~ F + C', 'fe')
    
    #                                 OLS Regression Results                            
    # ==============================================================================
    # Dep. Variable:                      I   R-squared:                       0.767
    # Model:                            OLS   Adj. R-squared:                  0.753
    # Method:                 Least Squares   F-statistic:                     309.0
    # Date:                Mon, 23 Nov 2015   Prob (F-statistic):           3.75e-60
    # Time:                        10:36:17   Log-Likelihood:                -1070.8
    # No. Observations:                 200   AIC:                             2148.
    # Df Residuals:                     188   BIC:                             2157.
    # Df Model:                           2                                         
    # Covariance Type:            nonrobust                                         
    # ==============================================================================
    #                  coef    std err          t      P>|t|      [95.0% Conf. Int.]
    # ------------------------------------------------------------------------------
    # Intercept    -58.7439     12.454     -4.717      0.000       -83.311   -34.177
    # F              0.1101      0.012      9.288      0.000         0.087     0.134
    # C              0.3101      0.017     17.867      0.000         0.276     0.344
    # ==============================================================================
    # Omnibus:                       29.604   Durbin-Watson:                   1.079
    # Prob(Omnibus):                  0.000   Jarque-Bera (JB):              162.947
    # Skew:                           0.283   Prob(JB):                     4.13e-36
    # Kurtosis:                       7.386   Cond. No.                     3.91e+03
    # ==============================================================================
    # 
    # Warnings:
    # [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    # [2] The condition number is large, 3.91e+03. This might indicate that there are
    # strong multicollinearity or other numerical problems.
    
    #Fixed effects with "robust" (HC1) standard errors. Defaults to clustering on the 'i' variable.
    grunfeld.xtreg('I ~ F + C', 'fe', 'robust')
    
    #                             OLS Regression Results                            
    # ==============================================================================
    # Dep. Variable:                      I   R-squared:                       0.767
    # Model:                            OLS   Adj. R-squared:                  0.753
    # Method:                 Least Squares   F-statistic:                     28.31
    # Date:                Mon, 23 Nov 2015   Prob (F-statistic):           0.000131
    # Time:                        10:41:20   Log-Likelihood:                -1070.8
    # No. Observations:                 200   AIC:                             2148.
    # Df Residuals:                     188   BIC:                             2157.
    # Df Model:                           2                                         
    # Covariance Type:              cluster                                         
    # ==============================================================================
    #                  coef    std err          z      P>|z|      [95.0% Conf. Int.]
    # ------------------------------------------------------------------------------
    # Intercept    -58.7439     27.603     -2.128      0.033      -112.845    -4.643
    # F              0.1101      0.015      7.248      0.000         0.080     0.140
    # C              0.3101      0.053      5.878      0.000         0.207     0.413
    # ==============================================================================
    # Omnibus:                       29.604   Durbin-Watson:                   1.079
    # Prob(Omnibus):                  0.000   Jarque-Bera (JB):              162.947
    # Skew:                           0.283   Prob(JB):                     4.13e-36
    # Kurtosis:                       7.386   Cond. No.                     3.91e+03
    # ==============================================================================
  
  
```
