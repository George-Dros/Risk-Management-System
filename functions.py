import pandas as pd
import numpy as np
from scipy.stats import norm
from fbm import FBM

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



