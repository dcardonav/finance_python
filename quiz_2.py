

import numpy as np
import pandas as pd
import scipy.stats

import edhec_risk_kit as erk

er = erk.get_hfi_returns()
er = er['2000':]


print(100*erk.var_gaussian(er['Distressed Securities'], level=1))

print(100*erk.var_gaussian(er['Distressed Securities'], level=1, modified=True))

print(100*erk.var_historic(er['Distressed Securities'], level=1))


ind = erk.get_ind_returns()['2013':'2017']
l = ['Books', 'Steel', 'Oil', 'Mines']
ind = ind[['Books', 'Steel', 'Oil', 'Mines']]
er = erk.annualize_rets(ind, 12)
cov = ind.cov()

w_msr = erk.msr(0.1, er, cov)
w_gmv = erk.gmv(cov)


ind = erk.get_ind_returns()['2018':'2018']

cov = (ind[l].cov())
erk.portfolio_vol(w_gmv, cov)*(12**0.5) # Recuerda anualizar la volatilidad!
erk.portfolio_vol(w_msr, cov)*(12**0.5) # Recuerda anualizar la volatilidad!
