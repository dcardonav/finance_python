

import numpy as np
import pandas as pd
import scipy.stats

import edhec_risk_kit as erk


df_quiz_1 = pd.read_csv('course_1/Portfolios_Formed_on_ME_monthly_EW.csv',
                        header=0, index_col=0, parse_dates=True, na_values=99.99)

columns = ['Lo 20', 'Hi 20']
df_quiz_1 = df_quiz_1[columns]/100
df_quiz_1.index = pd.to_datetime(df_quiz_1.index, format="%Y%m").to_period('M')


## se calcula el retorno por mes con la productoria y luego elevando a 1/# de meses
## luego se anualiza elevando a 1/12
ret_per_month = (df_quiz_1 + 1).prod() ** (1/df_quiz_1.shape[0]) - 1
annualized_return = (ret_per_month + 1) ** (12) -1

## para calcular la volatilidad se calcula la desviación estándar de los retornos
## y se multiplica por la raíz cuadrada de un año (12)
annualized_vol = df_quiz_1.std()*np.sqrt(12)

## para calcular el retorno anualizado sobre un intervalo de tiempo aprovechamos
## la estructura del índice como una fecha en el DataFrame
df_1999_2015 = df_quiz_1['1999':'2015']
ret_per_month = (df_1999_2015 + 1).prod() ** (1/df_1995_2015.shape[0]) - 1
annualized_return = (ret_per_month + 1) ** (12) -1

annualized_vol = df_1999_2015.std()*np.sqrt(12)


## calculating monthly drawdown
drawdowns_small = erk.drawdown(df_1999_2015['Lo 20'])
drawdowns_large = erk.drawdown(df_1999_2015['Hi 20'])

## obtaining the maximum drawdown and when it happened
print(np.round(drawdowns_small.min()*-1, 4))
print(np.round(drawdowns_large.min()*-1, 4))
print(drawdowns_small.idxmin())
print(drawdowns_large.idxmin())


## cargamos los datos para los demás puntos
df_edhec = erk.get_hfi_returns()

## cálculo de la semidesviación estándar (la que aplica sólo cuando tenemos retornos negativos)
semidev = erk.semideviation(df_edhec['2009':'2018'])

## quién ha tenido la semidesviación más alta entre 2009 y 2018
print(semidev.idxmax())
## la más baja
print(semidev.idxmin())

## ahora calculamos la asimetría
skew = erk.skewness(df_edhec['2009':'2018'])
print(skew.idxmin())

# ... y la kurtosis
kurt = erk.kurtosis(df_edhec['2000':])
print(kurt.idxmax())



