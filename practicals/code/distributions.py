from scipy.stats import norm, entropy

def gaussian_pdf(x, mean, std):
    """Compute the class-conditional probability assuming a Gaussian distribution.

    Parameters
    ----------
    x : float
        The input value.
    mean : float
        The mean of the Gaussian distribution.
    std : float
        The standard deviation of the Gaussian distribution.
        
    Returns 
    -------
    float
        The class-conditional probability.
    """	
    return norm.pdf(x, mean, std)

def kl_divergence(p, q):
    """
    Compute the KL-divergence between two distributions.
    
    Parameters
    ----------
    p : array_like
        The first distribution.
    q : array_like
        The second distribution.
        
    Returns
    -------
    float
        The KL-divergence between the two distributions
    """
    return entropy(p, q)