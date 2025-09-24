"""
Recommendation engine prototype:
- Pulls real data via yfinance
- Computes signals (momentum, mean-reversion, VIX-adjusted expected returns)
- Produces combined expected returns
- Runs constrained mean-variance optimizer with turnover constraint
- Outputs buy / sell / hold suggestions with target weights and notional trades using an LLM

Notes:
- Replace ASSETS with your desired universe (ETFs, stocks, bonds tickers).
- For production, replace yfinance with institutional data feed and add robust error handling.
"""
