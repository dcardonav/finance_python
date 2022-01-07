import numpy as np
import pandas as pd
import scipy.stats
import scipy.optimize

import edhec_risk_kit


def drawdown(return_series: pd.Series):
    """
    Takes a time series of asset returns
    Computes and returns a DataFrame that contains:
        - Wealth index
        - Previous peaks
        - Percent drawdowns
    :param return_series:
    :return:
    """

    wealth_index = 1000 * (1 + return_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks) / previous_peaks

    return pd.DataFrame({
        'wealth': wealth_index,
        'peaks': previous_peaks,
        'drawdown': drawdowns
    })


def get_ffme_returns():
    """
    Load the Fama-French Dataset for the returns of the Top and Bottom Deciles by MarketCap
    """

    me_m = pd.read_csv("course_1/Portfolios_Formed_on_ME_monthly_EW.csv",
                       header=0, index_col=0, na_values=-99.99)
    rets = me_m[['Lo 10', 'Hi 10']]
    rets.columns = ['SmallCap', 'LargeCap']
    rets = rets / 100
    rets.index = pd.to_datetime(rets.index, format='%Y%m').to_period('M')

    return rets


def get_hfi_returns():
    """
    Load and format the EDHEC Hedge Fund Index Returns
    """

    hfi = pd.read_csv("course_1/edhec-hedgefundindices.csv",
                      header=0, index_col=0, parse_dates=True)
    hfi = hfi / 100
    hfi.index = hfi.index.to_period('M')

    return hfi


def skewness(r):
    """
    Alternative to scipy.stats.skew()
    Computes the skewness of the supplied Series or DataFrame
    Returns a float or a Series
    """

    demeaned_r = r - r.mean()
    # use the population standard deviation, so set dof=0
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**3).mean()
    return exp/sigma_r**3




def kurtosis(r):
    """
    Alternative to scipy.stats.kurtosis()
    Computes the kurtosis of the supplied Series or DataFrame
    Returns a float or a Series
    """
    demeaned_r = r - r.mean()
    # use the population standard deviation, so set dof=0
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**4).mean()
    return exp/sigma_r**4


def is_normal(r, level=0.01):
    """
    Applies the Jarque-Bera test to determine if a Series is normal or not
    Test is applied at the 1% level by default
    Returns True if the hypothesis of normality is not rejected, False otherwise
    """
    statistic, p_value = scipy.stats.jarque_bera(r)

    return p_value > level

def semideviation(r):
    """
    Returns the semi-deviation (a.k.a.: negative semideviation of r
    r must be a Series or DataFrame
    """

    is_negative = r < 0
    return r[is_negative].std(ddof=0)

def var_historic(r, level=5):
    """
    Returns the historic Value at Risk at a specified level
    i.e. returns the number such that "level" percent of the returns
    fall below that number, and the (100-level) percent are above
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(var_historic, level=level) # calls the function again, but this time over each column
    elif isinstance(r, pd.Series):
        return -np.percentile(r, level) # minus for reporting purposes, but remember: those are losses
    else:
        raise TypeError("Expected r to be Series or DataFrame")

def var_gaussian(r, level=5, modified=False):
    """
    Returns the Parametric Gaussian VaR of a Series or DataFrame
    If "modified" is True, then the modified VaR is returned,
    using the Cornish-Fisher modification
    """
    # compute the Z score assuming it was Gaussian
    z = scipy.stats.norm.ppf(level/100)
    if modified:
        # modify the Z score based on observed skewness and kurtosis
        s = skewness(r)
        k = kurtosis(r)
        z = (z +
                (z**2 - 1)*s/6 +
                (z**3 -3*z)*(k-3)/24 -
                (2*z**3 - 5*z)*(s**2)/36
            )
    return -(r.mean() + z*r.std(ddof=0))


def cvar_historic(r, level=5):
    """
    Computes the Conditional VaR of Series or DataFrame
    """

    if isinstance(r, pd.Series):
        is_beyond = r <= -var_historic(r, level=level)
        return -r[is_beyond].mean() # average losses AFTER level percentile
    elif isinstance(r, pd.DataFrame):
        return r.aggregate(cvar_historic, level=level)  # calls the function again, but this time over each column
    else:
        raise TypeError("Expected r to be Series or DataFrame")


def get_ind_returns():
    """
    Load and format the Ken French 30 Industry Portfolios Values Weighted Monthly Returns
    """

    # load the dataset, convert to dates and put months
    ind = pd.read_csv('course_1/ind30_m_vw_rets.csv', header=0, index_col=0) / 100
    ind.index = pd.to_datetime(ind.index, format="%Y%m").to_period('M')
    # remove spaces from the names
    ind.columns = ind.columns.str.strip()

    return ind

def get_ind_size():
    """
    Load and format the Ken French 30 Industry Portfolios with the value of each industry
    """

    # load the dataset, convert to dates and put months
    ind = pd.read_csv('course_1/ind30_m_size.csv', header=0, index_col=0)
    ind.index = pd.to_datetime(ind.index, format="%Y%m").to_period('M')
    # remove spaces from the names
    ind.columns = ind.columns.str.strip()

    return ind

def get_ind_n_firms():
    """
    Load and format the Ken French 30 Industry Portfolios Values with the number of firms in each industry
    """

    # load the dataset, convert to dates and put months
    ind = pd.read_csv('course_1/ind30_m_nfirms.csv', header=0, index_col=0)
    ind.index = pd.to_datetime(ind.index, format="%Y%m").to_period('M')
    # remove spaces from the names
    ind.columns = ind.columns.str.strip()

    return ind

def annualize_rets(r, periods_per_year):
    """
    Annualizes a set of returns
    We should infer the periods per year
    """

    compounded_growth = (1+r).prod()
    n_periods = r.shape[0]

    return  compounded_growth**(periods_per_year/n_periods)-1

def annualize_vol(r, periods_per_year):
    """
    Annualizes the volatility of a set of returns
    """
    return r.std()*(periods_per_year**0.5)

def sharpe_ratio(r, riskfree_rate, periods_per_year):
    """
    Computes the annualized sharpe ratio of a set of returns
    """

    rf_per_period = (1+riskfree_rate)**(1/periods_per_year)-1
    excess_ret = r - rf_per_period
    ann_ex_ret = annualize_rets(excess_ret, periods_per_year)
    ann_vol = annualize_vol(r, periods_per_year)

    return ann_ex_ret/ann_vol

def portfolio_return(weights, returns):
    """
    Going from weights (how much to allocate in each asset) to the return of that bundle
    """
    return weights.T @ returns

def portfolio_vol(weights, covmat):
    """
    Going from weights (how much to allocate in each asset) to the volatility of that bundle
    """
    return (weights.T @ covmat @ weights)**0.5

def plot_ef2(n_points, er, cov, style='.-'):
    """
    Plots the 2-asset efficient frontier
    """

    if er.shape[0] != 2 or er.shape[0] != 2:
        raise ValueError("plot_ef2 can only plot 2-asset frontiers")

    weights = [np.array([w, 1 - w]) for w in np.linspace(0, 1, n_points)]
    # with the sequence of weights we now retrieve the returns and the volatilities
    rets = [portfolio_return(w, er) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]
    ef = pd.DataFrame(data={"Ret": rets, "Vol": vols})
    return ef.plot.line(x="Vol", y="Ret", style=style)


def plot_efn(n_points, er, cov, style='.-', show_cml=False, riskfree_rate=0, show_ew=False, show_gmv=False):
    """
    Plots the N-asset efficient frontier
    """

    # This is the critical part: finding the optimal mix of weights
    weights = optimal_weights(n_points, er, cov)
    # with the sequence of weights we now retrieve the returns and the volatilities
    rets = [portfolio_return(w, er) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]
    ef = pd.DataFrame(data={"Ret": rets, "Vol": vols})
    ax = ef.plot.line(x="Vol", y="Ret", style=style)

    # Code to plot the equally-weighted portfolio
    if show_ew:
        n = er.shape[0]
        w_ew = np.repeat(1/n, n)
        r_ew = portfolio_return(w_ew, er)
        vol_ew = portfolio_vol(w_ew, cov)
        # display on the existing axis
        ax.plot([vol_ew], [r_ew], color='goldenrod', marker='o', markersize=10)

    # Code to plot the Global-Minimum-Variance portfolio
    if show_gmv:
        n = er.shape[0]
        # only depends on the covariance matrix
        w_gmv = gmv(cov)
        r_gmv = portfolio_return(w_gmv, er)
        vol_gmv = portfolio_vol(w_gmv, cov)
        # display on the existing axis
        ax.plot([vol_gmv], [r_gmv], color='midnightblue', marker='o', markersize=10)

    if show_cml:
        ax.set_xlim(left=0)
        w_msr = msr(riskfree_rate, er, cov)
        r_msr = portfolio_return(w_msr, er)
        vol_msr = portfolio_vol(w_msr, cov)

        # Now we plot the Capital Market Line
        cml_x = [0, vol_msr]
        cml_y = [riskfree_rate, r_msr]
        ax.plot(cml_x, cml_y, color='green', marker='o', linestyle='dashed', markersize=12, linewidth=2)

    return ax

def gmv(cov):
    """
    Returns the weights associated to the Global Minimum Variance portfolio
    """

    n = cov.shape[0]
    return msr(0, np.repeat(1, n), cov)



def optimal_weights(n_points, er, cov):
    """
    List of weights to run the optimizer on to minimize the vol
    """

    # show me the minimum volatility for a space of min and max returns
    target_rs = np.linspace(er.min(), er.max(), n_points)
    weights = [minimize_vol(target_return, er, cov) for target_return in target_rs]

    return weights


def minimize_vol(target_return, er, cov):
    """
    Go from target return to optimal n-dim weight vector
    """

    n = er.shape[0]
    # the optimizer needs an initial point
    init_guess = np.repeat(1/n, n)

    # don't want to go lower than zero and not higher than 1 (no shorting, no leverage)
    bounds = ((0.0, 1.0),) * n # multiplying a tuple generates multiple tuples

    # let us specify the constraints
    return_is_target = {
        'type': 'eq',
        # arguments required for the evaluation function
        'args': (er,),
        # This is the function we're going to use to check if we achieved the expected return
        'fun': lambda weights, er:  target_return - edhec_risk_kit.portfolio_return(weights, er)
    }

    weights_sum_to_1 = {
        'type': 'eq',
        'fun': lambda weights: np.sum(weights) - 1
    }

    results = scipy.optimize.minimize(portfolio_vol, # function to minimize
                                      init_guess, # starting point
                                      args=(cov,), # argument required by the function to minimize
                                      method="SLSQP", # quadratic programming algorithm
                                      options={'disp':False}, # don't show solver output
                                      constraints=(return_is_target, weights_sum_to_1), # problem constraints
                                      bounds=bounds) # bounds on the solution

    return results.x


def msr(riskfree_rate, er, cov):
    """
    Gives the optimum weights in pressence of a risk-free assets (Maximum Sharpe Ratio)
    """

    n = er.shape[0]
    # the optimizer needs an initial point
    init_guess = np.repeat(1/n, n)

    # don't want to go lower than zero and not higher than 1 (no shorting, no leverage)
    bounds = ((0.0, 1.0),) * n # multiplying a tuple generates multiple tuples

    weights_sum_to_1 = {
        'type': 'eq',
        'fun': lambda weights: np.sum(weights) - 1
    }

    results = scipy.optimize.minimize(neg_sharpe_ratio, # function to maximize by negation
                                      init_guess, # starting point
                                      args=(riskfree_rate, er, cov,),  # argument required by the function to minimize
                                      method="SLSQP",  # quadratic programming algorithm
                                      options={'disp':False},  # don't show solver output
                                      constraints=(weights_sum_to_1),  # problem constraints
                                      bounds=bounds)  # bounds on the solution

    return results.x


def neg_sharpe_ratio(weights, riskfree_rate, er, cov):
    """
    Returns the negative of the Sharpe Ratio under given weights
    """
    r = portfolio_return(weights, er)
    vol = portfolio_vol(weights, cov)

    return -(r-riskfree_rate)/vol


def get_total_market_index_returns_1():
    ind_return = get_ind_returns()
    ind_nfirms = get_ind_n_firms()
    ind_size = get_ind_size()

    # In this section we are going to build a market index

    # Compute market capitalization
    ind_mktcap = ind_nfirms * ind_size

    # Compute total market capitalization
    total_mktcap = ind_mktcap.sum(axis='columns')
    total_mktcap.plot()

    # Compute the capitalization weight. This calculates the participation of each industry
    # in the total market capitalization
    ind_capweight = ind_mktcap.divide(total_mktcap, axis='rows')
    ind_capweight[['Fin', 'Steel']].plot(figsize=(12, 6))

    # Weighted average of returns, whole market
    total_market_return = (ind_capweight * ind_return).sum(axis='columns')

    return  total_market_return


def get_total_market_index_returns():
    """
    Load the 30 industry portfolio data and derive the returns of a capweighted total market index
    """
    ind_nfirms = get_ind_n_firms()
    ind_size = get_ind_size()
    ind_return = get_ind_returns()
    ind_mktcap = ind_nfirms * ind_size
    total_mktcap = ind_mktcap.sum(axis=1)
    ind_capweight = ind_mktcap.divide(total_mktcap, axis="rows")
    total_market_return = (ind_capweight * ind_return).sum(axis="columns")
    return total_market_return


def run_ccpi(risky_r, safe_r=None, m =3, start=1000, floor=0.8, riskfree_rate=00.03, drawdown=None):
    """
    Run a backtest of the CPPI stategy, given a set of returns for the risky asset
    Returns a dictionary containing: Asset Value History, Risk Budget History, Risky Weight History
    :return:
    """
    dates = risky_r.index
    n_steps = len(dates)
    floor_value = start * floor
    peak = start
    if isinstance(risky_r, pd.Series):
        risky_r = pd.DataFrame(risky_r, columns=['R'])

    if safe_r is None:
        safe_r = pd.DataFrame().reindex_like(risky_r)
        safe_r.values[:] = riskfree_rate/12

    account_history = pd.DataFrame().reindex_like(risky_r)
    cushion_history = pd.DataFrame().reindex_like(risky_r)
    risky_w_history = pd.DataFrame().reindex_like(risky_r)

    account_value = start
    for step in range(0, n_steps):
        if drawdown is not None:
            peak = np.maximum(peak, account_value)
            floor_value = peak * (1-drawdown)

        cushion = (account_value - floor_value) / account_value
        risky_w = m * cushion
        risky_w = np.minimum(risky_w, 1)
        risky_w = np.maximum(risky_w, 0)
        safe_w = 1 - risky_w
        risky_alloc = account_value * risky_w
        safe_alloc = account_value * safe_w

        # update the account value
        account_value = risky_alloc * (1 + risky_r.iloc[step]) + safe_alloc * (1 + safe_r.iloc[step])

        # save the values to be analyzed later
        cushion_history.iloc[step] = cushion
        risky_w_history.iloc[step] = risky_w
        account_history.iloc[step] = account_value

    risky_wealth = start*(1+risky_r).cumprod()
    backtest_result = {
        'Wealth': account_history,
        'Risky Wealth': risky_wealth,
        'Risky Budget': cushion_history,
        'Risky Allocation': risky_w_history,
        'm': m,
        'start': start,
        'floor': floor,
        'risky_r': risky_r,
        'safe_r': safe_r
    }

    return backtest_result

def summary_stats(r, riskfree_rate=0.03):
    """
    Return a DataFrame that contains aggregated summary stats for the returns in the columns of r
    """
    ann_r = r.aggregate(annualize_rets, periods_per_year=12)
    ann_vol = r.aggregate(annualize_vol, periods_per_year=12)
    ann_sr = r.aggregate(sharpe_ratio, riskfree_rate=riskfree_rate, periods_per_year=12)
    dd = r.aggregate(lambda r: drawdown(r).drawdown.min())
    skew = r.aggregate(skewness)
    kurt = r.aggregate(kurtosis)
    cf_var5 = r.aggregate(var_gaussian, modified=True)
    hist_cvar5 = r.aggregate(cvar_historic)

    return pd.DataFrame({
        "Annualized Return": ann_r,
        "Annualized Vol": ann_vol,
        "Skewness": skew,
        "Kurtosis": kurt,
        "Cornish-Fisher VaR (5%)": cf_var5,
        "Historic CVaR (5%)": hist_cvar5,
        "Sharpe Ratio": ann_sr,
        "Max Drawdown": dd
    })