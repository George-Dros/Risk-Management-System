import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

def portfolio_return(weights, returns):
    """
    Weights -> Returns
    """
    return weights.T @ returns


def portfolio_vol(weights, covmat):
    """
    Weights -> Volatility
    """
    return (weights.T @ covmat @ weights)**0.5

def annualize_vol(r, periods_per_year):
    """
    Annualizes the volatility of a set of returns
    """
    return r.std() * (periods_per_year**0.5)


def annualize_rets(r, periods_per_year):
    """
    Annualizes a set of returns
    """
    compounded_growth = (1+r).prod()
    n_periods = r.shape[0]
    return compounded_growth**(periods_per_year/n_periods)-1

def compound_returns(s, start=100):
    '''
    Compound a pd.Dataframe or pd.Series of returns from an initial default value equal to 100.
    In the former case, the method compounds the returns for every column (Series) by using pd.aggregate. 
    The method returns a pd.Dataframe or pd.Series - using cumprod(). 
    See also the COMPOUND method.
    '''
    if isinstance(s, pd.DataFrame):
        return s.aggregate( compound_returns, start=start )
    elif isinstance(s, pd.Series):
        return start * (1 + s).cumprod()
    else:
        raise TypeError("Expected pd.DataFrame or pd.Series")


def var_historic(r, level=5):
    """
    VaR Historic
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(var_historic, level=level)
    elif isinstance(r, pd.Series):
        return -np.percentile(r, level)
    else:
        raise TypeError("Expected r to be Series or DataFrame")

def skewness(r):
    """
    Alternative to scipy.stats.skew()
    Computes the skewness of the supplied Series or DataFrame
    Returns a float or a Series
    """
    
    demeaned_r = r-r.mean()
    #Use the population standard deviation, so set dof=0
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**3).mean()
    return exp/sigma_r**3


def kurtosis(r):
    """
    Alternative to scipy.stats.kurtosis()
    Computes the kurtosis of the supplied Series or DataFrame
    Returns a float or a Series
    """
    
    demeaned_r = r-r.mean()
    #Use the population standard deviation, so set dof=0
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**4).mean()
    return exp/sigma_r**4

def var_gaussian(r, level=5, modified=False):
    """
    Returns the Parametric Gaussian VaR of a Series or DataFrame
    """
    
    #compute the Z score assuming it was gaussian
    z = norm.ppf(level/100)
    if modified:
        #Modify the Z scroe based on observed skewness and kurtosis
        s = skewness(r)
        k = kurtosis(r)
        z = (z + 
                (z**2 - 1)*s/6 +
                (z**3 - 3*z)*(k-3)/24 -
                (2*z**3 - 5*z)*(s**2)/36             
            )
    return -(r.mean() + z*r.std(ddof=0))

def cvar_historic(r, level=5):
    """
    Computes the Conditional VaR of Series or DataFrame
    """
    if isinstance(r, pd.Series):
        is_beyond = r <= -var_historic(r, level=level)
        return -r[is_beyond].mean()
    elif isinstance(r, pd.DataFrame):
        return r.aggregate(cvar_historic, level=level)
    else:
        raise TypeError("Expected r to be a Series or DataFrame")

def cvar_gaussian(r, level=5, modified=False):
    """
    Returns the Parametric Gaussian CVaR of a Series or DataFrame
    """
    # Compute the Z score assuming it was Gaussian
    z = norm.ppf(level / 100)
    
    if modified:
        # Modify the Z score based on observed skewness and kurtosis
        s = skewness(r)
        k = kurtosis(r)
        z = (z +
             (z**2 - 1) * s / 6 +
             (z**3 - 3*z) * (k - 3) / 24 -
             (2*z**3 - 5*z) * (s**2) / 36)
    
    # Compute the Gaussian CVaR
    mean = r.mean()
    std_dev = r.std(ddof=0)
    pdf_z = norm.pdf(z)  # PDF of the standard normal at z
    cvar = -(mean + std_dev * (pdf_z / (1 - level / 100)))
    
    return cvar


def drawdown(rets: pd.Series, start=100):
    '''
    Compute the drawdowns of an input pd.Series of returns. 
    The method returns a dataframe containing: 
    1. the associated wealth index (for an hypothetical starting investment of $100) 
    2. all previous peaks 
    3. the drawdowns
    '''
    wealth_index   = compound_returns(rets, start=start)
    previous_peaks = wealth_index.cummax()
    drawdowns      = (wealth_index - previous_peaks ) / previous_peaks
    df = pd.DataFrame({"Wealth": wealth_index, "Peaks": previous_peaks, "Drawdown": drawdowns} )
    return df


def monte_carlo_simulation(daily_returns, covariance_matrix, portfolio_weights, 
                           monte_carlo_sims=1000, t=100, initial_portfolio=1000):
    """
    Perform a Monte Carlo simulation to project portfolio value over time and return final portfolio values.

    Parameters:
    - daily_returns (numpy array): Historical daily returns of the assets.
    - covariance_matrix (numpy array): Covariance matrix of asset returns.
    - portfolio_weights (numpy array): Weights of the assets in the portfolio.
    - monte_carlo_sims (int): Number of Monte Carlo simulations (default: 1000).
    - t (int): Number of days to simulate (default: 100).
    - initial_portfolio (float): Initial portfolio value (default: 1000).

    Returns:
    - final_portfolio_values (numpy array): Final portfolio values from all Monte Carlo simulations.
    - None (generates and displays a plot of portfolio value paths).
    """
    # Calculate the mean returns of the assets
    mean_returns = daily_returns.mean()
    
    # Create a matrix of mean returns for the simulation period
    mean_m = np.full(shape=(t, len(portfolio_weights)), fill_value=mean_returns).T
    
    # Initialize an array to store the portfolio simulation results
    portfolio_sims = np.full(shape=(t, monte_carlo_sims), fill_value=0.0)

    # Monte Carlo Simulation Loop
    for m in range(monte_carlo_sims):
       
        Z = np.random.normal(size=(t, len(portfolio_weights)))
        L = np.linalg.cholesky(covariance_matrix)
        monte_carlo_daily_returns = mean_m + np.inner(L, Z)
        portfolio_sims[:, m] = np.cumprod(np.inner(portfolio_weights, monte_carlo_daily_returns.T) + 1) * initial_portfolio

    # Extract the final portfolio values
    final_portfolio_values = portfolio_sims[-1, :]  # Last row contains the final portfolio value for each simulation

    # Plot the results
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_sims, color="blue", alpha=0.4)  # Plot all simulation paths
    plt.ylabel("Portfolio Value ($)")
    plt.xlabel("Days")
    plt.title("Monte Carlo Simulation of a Stock Portfolio")
    plt.show()

    # Return the final portfolio values for further analysis
    return final_portfolio_values


def sharpe_ratio(s, risk_free_rate, periods_per_year, v=None):
    '''
    Computes the annualized sharpe ratio. 
    The variable periods_per_year can be, e.g., 12, 52, 252, in case of yearly, weekly, and daily data.
    The variable risk_free_rate is the annual one.
    The method takes in input either a DataFrame, a Series or a single number. 
    In the former case, it computes the annualized sharpe ratio of every column (Series) by using pd.aggregate. 
    In the latter case, s is the (allready annualized) return and v is the (already annualized) volatility 
    computed beforehand, for example, in case of a portfolio.
    '''
    if isinstance(s, pd.DataFrame):
        return s.aggregate( sharpe_ratio, risk_free_rate=risk_free_rate, periods_per_year=periods_per_year, v=None)
    elif isinstance(s, pd.Series):
        # convert the annual risk free rate to the period assuming that:
        # RFR_year = (1+RFR_period)^{periods_per_year} - 1. Hence:
        rf_to_period = (1 + risk_free_rate)**(1/periods_per_year) - 1        
        excess_return = s - rf_to_period
        # now, annualize the excess return
        ann_ex_rets = annualize_rets(excess_return, periods_per_year)
        # compute annualized volatility
        ann_vol = annualize_vol(s, periods_per_year)
        return ann_ex_rets / ann_vol
    elif isinstance(s, (int,float)) and v is not None:
        # Portfolio case: s is supposed to be the single (already annnualized) 
        # return of the portfolio and v to be the single (already annualized) volatility. 
        return (s - risk_free_rate) / v

