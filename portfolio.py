import pandas as pd
from pypfopt import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt import plotting
from pypfopt import CLA


# Read in price data
df = pd.read_csv("dados/grupo1.csv", parse_dates=True, index_col="Data")

i = 1
max_return = 0
for i in range(1,11):
    path = "dados/grupo" + str(i) + ".csv"
    df = pd.read_csv(path, parse_dates=True, index_col="Data")
    # Calculate expected returns and sample covariance
    mu = expected_returns.mean_historical_return(df, frequency= 12)
    S = risk_models.sample_cov(df)

    # Optimise for maximal Sharpe ratio
    ef = EfficientFrontier(mu, S)
    raw_weights = ef.max_sharpe()
    cleaned_weights = ef.clean_weights()
    ef.save_weights_to_file("weights.csv")  # saves to file
    print(cleaned_weights)
    ef.portfolio_performance(verbose=True)

    print("---------")
    retorno = ef.portfolio_performance(verbose=True)
    print(retorno[2])
    print("---------")

    if retorno[0] > max_return:
        max_return = retorno[2]
        port = path


print(max_return)
print(port)


    # plotting.plot_covariance(S, plot_correlation=True)
    # cla = CLA(mu, S)
    # cla.max_sharpe()
    # cla.portfolio_performance(verbose=True)
    # ax = plotting.plot_efficient_frontier(cla)