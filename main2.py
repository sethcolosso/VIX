import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
import cvxpy as cp
from datetime import datetime, timedelta
import scipy.linalg
import warnings

warnings.filterwarnings("ignore", message="You are solving a parameterized problem that is not DPP")
from transformers import pipeline
# Load lightweight Hugging Face model (only once)
nlp_model = pipeline("text2text-generation", model="google/flan-t5-small")
def generate_nlp_summary(instructions_df, vix_value):
    """
    Convert raw trade instructions + VIX into natural language buy/sell suggestions.
    """
    # Turn dataframe into plain text
    summary_text = f"VIX is {vix_value:.2f}. Portfolio trade instructions:\n"
    for _, row in instructions_df.iterrows():
        summary_text += f"{row['action']} {row['ticker']} worth {row['delta_usd']:.0f} USD ({row['delta_pct_of_portfolio']})\n"

    # Build prompt for the model
    prompt = (
        f"Summarize the key portfolio actions in simple terms and explain why:\n\n{summary_text}"
    )

    response = nlp_model(prompt, max_new_tokens=1500, do_sample=False)
    return response[0]["generated_text"]

ASSETS = ["SPY", "QQQ", "TLT", "IEF", "GLD", "NVDA", "MSFT", "TSLA"]  # example universe
START_DATE = (datetime.utcnow() - timedelta(days=5*365)).strftime("%Y-%m-%d")
END_DATE = datetime.utcnow().strftime("%Y-%m-%d")
RISK_FREE_RATE = 0.03

# Signal weighting (tune these)
W_MOM_12 = 0.4
W_MOM_3  = 0.2
W_MR_21  = 0.2   # mean-reversion short-term
W_VIX    = 0.2

# Optimization params
TARGET_RETURN_ANN = 0.06
MIN_WEIGHT = 0.0
MAX_WEIGHT = 0.4
TURNOVER_MAX = 0.2      # maximum fraction of portfolio turnover allowed this rebalance
TRANSACTION_COST_BPS = 5  # 5 bps per trade (0.0005) used for cost penalization
VOL_TARGET = 0.10
VIX_THRESHOLD = 18.0
ALPHA_VOL_BLEND = 0.6

# Trade execution params / portfolio
CURRENT_PORTFOLIO_VALUE = 2_000_000  # USD, example
CURRENT_HOLDINGS = {                 # example existing holdings (ticker -> USD notional)
    "SPY": 400_000,
    "QQQ": 300_000,
    "NVDA": 300_000,
    "MSFT":200_000,
    "TSLA":200_000,
    "TLT": 200_000,
    "GLD": 300_000
}

# -------------------------
# Utilities / Data fetch
# -------------------------
def fetch_price_data(tickers, start, end):
    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    if isinstance(data, pd.Series):
        data = data.to_frame(name=tickers[0])
    return data.dropna(how="all")

def fetch_vix(start, end):
    v = yf.download("^VIX", start=start, end=end, auto_adjust=True, progress=False)["Close"]
    v.name = "VIX"  # Fix: set the Series name directly
    return v

# -------------------------
# Signals
# -------------------------
def compute_returns(price_df):
    return price_df.pct_change().dropna()

def momentum(price_df, months=12):
    # price-based momentum: (price_now / price_n_months_ago - 1)
    lookback_days = int(months * 21)  # approximate trading days
    mom = price_df.pct_change(periods=lookback_days).iloc[-1]
    return mom

def short_term_mean_reversion_signal(price_df, window=21):
    # z-score of latest return vs trailing mean/std
    ret = compute_returns(price_df)
    rolling_mean = ret.rolling(window=window).mean().iloc[-1]
    rolling_std = ret.rolling(window=window).std(ddof=1).iloc[-1].replace(0, np.nan)
    latest_ret = ret.iloc[-1]
    z = (latest_ret - rolling_mean) / rolling_std
    # mean-reversion signal: negative z => buy (price dropped vs mean)
    mr_signal = -z
    return mr_signal.fillna(0.0)

def avg_daily_volume(tickers, start, end):
    df = yf.download(tickers, start=start, end=end, progress=False)["Volume"]
    if isinstance(df, pd.Series):
        df = df.to_frame(name=tickers[0])
    return df.mean().iloc[-1] if isinstance(df, pd.Series) else df.mean()

    # VIX-driven expected return: simple approach -> assets with higher beta to market suffer more when VIX rises.
def compute_beta(returns_df, market_col="SPY"):
    if market_col not in returns_df.columns:
        raise ValueError("Market proxy must be in returns dataframe for beta calc.")
    cov = returns_df.cov()
    var_mkt = returns_df[market_col].var()
    betas = cov.loc[:, market_col] / var_mkt
    return betas
# Build combined expected returns
def build_expected_returns(price_df, vix_series, w_mom12=W_MOM_12, w_mom3=W_MOM_3, w_mr=W_MR_21, w_vix=W_VIX):
    ret = compute_returns(price_df)
    # Momentum signals
    mom12 = momentum(price_df, months=12)
    mom3 = momentum(price_df, months=3)
    # Mean-reversion signal
    mr21 = short_term_mean_reversion_signal(price_df, window=21)
    # VIX adjusted expected excess return: negative when VIX > threshold for high beta assets
    vix_latest = vix_series.dropna().values[-1].item()
    betas = compute_beta(ret, market_col="SPY")
    # crude implied "penalty" = (vix_latest - vix_threshold)/vix_threshold scaled by beta
    vix_penalty = max(0.0, (vix_latest - VIX_THRESHOLD) / VIX_THRESHOLD)
    vix_effect = -betas * vix_penalty  # negative expected return for high-beta when VIX elevated
    
    # normalize signals
    def zscore(s):
        s = s.astype(float)
        return (s - s.mean()) / (s.std(ddof=1) if s.std(ddof=1) != 0 else 1.0)

    mom12_z = zscore(mom12)
    mom3_z  = zscore(mom3)
    mr21_z  = zscore(mr21)
    vix_z   = zscore(vix_effect)

    # combine into expected returns (annualized scale)
    combined = w_mom12 * mom12_z + w_mom3 * mom3_z + w_mr * mr21_z + w_vix * vix_z
    # convert scale to realistic expected annual returns by scaling to a target mean (e.g., 6% spread)
    # preserve sign and relative ranking
    target_mean = 0.06
    # scale so mean of positive signals maps to half of target_mean (conservative)
    pos_mean = combined[combined>0].mean() if (combined>0).any() else 1.0
    scale = (target_mean/2) / pos_mean if pos_mean and pos_mean != 0 else 1.0
    exp_ret_ann = combined * scale
    return exp_ret_ann.fillna(0.0), vix_latest

# -------------------------
# Covariance (adjusted with VIX)
# -------------------------
def build_forward_covariance(price_df, vix_series, alpha=ALPHA_VOL_BLEND, vix_thresh=VIX_THRESHOLD, gamma=0.7):
    returns = compute_returns(price_df)
    sigma_hist_daily = returns.std(ddof=1)
    sigma_hist_ann = sigma_hist_daily * np.sqrt(252)
    cov_daily = returns.cov()
    corr = returns.corr()

    vix_latest = vix_series.dropna().values[-1].item()  # <-- updated line
    vix_ann = vix_latest/100.0

    betas = compute_beta(returns, market_col="SPY")
    sigma_impl_ann = (betas.abs() * vix_ann).fillna(sigma_hist_ann)  # fallback to hist
    sigma_adj_ann = alpha * sigma_hist_ann + (1 - alpha) * sigma_impl_ann

    # correlation inflation
    if vix_latest > vix_thresh:
        excess = (vix_latest - vix_thresh) / vix_thresh
        corr_adj = corr * (1.0 + gamma * excess)
    else:
        corr_adj = corr
    corr_adj = corr_adj.clip(lower=-0.95, upper=0.95)
    D = np.diag(sigma_adj_ann.values)
    cov_adj_ann = D.dot(corr_adj.values).dot(D)
    return cov_adj_ann, sigma_adj_ann, vix_latest

def nearest_psd(mat):
    """Returns the nearest positive semidefinite matrix to mat."""
    # Symmetrize
    B = (mat + mat.T) / 2
    # Eigen-decomposition
    _, s, V = np.linalg.svd(B)
    H = np.dot(V.T, np.dot(np.diag(s), V))
    # Ensure PSD
    A2 = (B + H) / 2
    A3 = (A2 + A2.T) / 2
    # If still not PSD, add small jitter
    if np.all(np.linalg.eigvals(A3) > 0):
        return A3
    spacing = np.spacing(np.linalg.norm(mat))
    I = np.eye(mat.shape[0])
    k = 1
    while not np.all(np.linalg.eigvals(A3) > 0):
        mineig = np.min(np.linalg.eigvals(A3))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1
    return A3

# -------------------------
# Optimizer with turnover & tx cost
# -------------------------
def optimize_with_turnover(mu_ann, cov_ann, current_weights, lb=MIN_WEIGHT, ub=MAX_WEIGHT, turnover_max=TURNOVER_MAX, tx_cost_bps=TRANSACTION_COST_BPS):
    """
    Minimize variance - lambda * expected return + transaction cost penalty subject to:
      - sum(weights)=1
      - bounds
      - turnover constraint sum(|w - w_current|) <= turnover_max
    We convert tx_cost_bps to a linear penalty on absolute changes.
    """
    n = len(mu_ann)
    w = cp.Variable(n)
    w_curr = np.array(current_weights)

    Sigma = cp.Parameter((n,n), PSD=True)
    mu = cp.Parameter(n)
    Sigma.value = cov_ann
    mu.value = mu_ann

    lam = 0.0  # We solve min variance subject to achieving expected return >= some target instead of penalty
    # constraints
    constraints = [
        cp.sum(w) == 1,
        w >= lb,
        w <= ub,
        cp.norm1(w - w_curr) <= turnover_max
    ]
    # objective: minimize variance, subject to target return
    obj = cp.quad_form(w, Sigma)
    problem = cp.Problem(cp.Minimize(obj), constraints)

    problem.solve(solver=cp.OSQP, verbose=False)
    if w.value is None:
        # fallback: equal weight respecting bounds
        w_opt = np.ones(n) / n
    else:
        w_opt = np.array(w.value).flatten()
    return w_opt

# -------------------------
# Convert weights -> trades
# -------------------------
def compute_trade_instructions(target_weights, current_holdings, tickers, portfolio_value=CURRENT_PORTFOLIO_VALUE, tolerance_pct=0.01):
    # current_weights from holdings
    current_notional = np.array([current_holdings.get(t, 0.0) for t in tickers])
    curr_sum = current_notional.sum()
    if curr_sum == 0:
        current_weights = np.zeros(len(tickers))
    else:
        current_weights = current_notional / curr_sum

    target_notional = target_weights * portfolio_value
    current_notional_dollars = current_weights * portfolio_value

    delta = target_notional - current_notional_dollars

    instructions = []
    for t, td, cd in zip(tickers, delta, current_notional_dollars):
        pct_change = (td - cd) / (cd + 1e-9) if cd != 0 else np.sign(td) * 1.0
        if abs(td - cd) / (portfolio_value + 1e-9) < tolerance_pct:
            action = "HOLD"
        elif td > cd:
            action = "BUY"
        else:
            action = "SELL"
        instructions.append({
            "ticker": t,
            "action": action,
            "current_notional": cd,
            "target_notional": td,
            "delta_usd": td - cd,
            "delta_pct_of_portfolio": (td - cd) / portfolio_value
        })
    # sort by absolute delta descending
    instructions = sorted(instructions, key=lambda x: abs(x["delta_usd"]), reverse=True)
    return instructions

# -------------------------
# Main run
# -------------------------
def run_reco_pipeline(tickers=ASSETS, start=START_DATE, end=END_DATE, current_holdings_dict=CURRENT_HOLDINGS):
    price_df = fetch_price_data(tickers, start, end)
    vix = fetch_vix(start, end)
    if price_df.shape[0] < 250:
        raise RuntimeError("Not enough history for signals. Increase start date.")

    exp_ret_ann, vix_latest = build_expected_returns(price_df, vix)
    cov_adj_ann, sigma_adj_ann, vix_latest2 = build_forward_covariance(price_df, vix)
    cov_adj_ann = nearest_psd(cov_adj_ann)

    # current_weights from holdings
    tickers_list = list(price_df.columns)
    # map current holdings to same order
    curr_notional = np.array([current_holdings_dict.get(t, 0.0) for t in tickers_list])
    curr_sum = curr_notional.sum()
    curr_weights = (curr_notional / curr_sum) if curr_sum > 0 else np.zeros(len(tickers_list))

    # optimize with turnover constraint
    w_opt = optimize_with_turnover(exp_ret_ann.values, cov_adj_ann, curr_weights)

    # apply volatility targeting (scale to VOL_TARGET)
    portfolio_vol = np.sqrt(w_opt @ cov_adj_ann @ w_opt)
    if portfolio_vol > 0:
        leverage = min(3.0, VOL_TARGET / portfolio_vol)
    else:
        leverage = 1.0
    w_target = w_opt * leverage
    # renormalize (to sum to 1; if we allow leverage >1 we can let sum>1, here we renormalize to sum=1)
    w_target = np.maximum(w_target, 0.0)
    if w_target.sum() > 0:
        w_target = w_target / w_target.sum()
    else:
        w_target = np.ones_like(w_target) / len(w_target)

    # compute instructions
    instructions = compute_trade_instructions(w_target, current_holdings_dict, tickers_list, portfolio_value=CURRENT_PORTFOLIO_VALUE)
    # build output dataframe
    instr_df = pd.DataFrame(instructions)
    return {
        "tickers": tickers_list,
        "vix_latest": vix_latest,
        "expected_returns_ann": pd.Series(exp_ret_ann.values, index=tickers_list),
        "sigma_adj_ann": pd.Series(sigma_adj_ann.values, index=tickers_list),
        "target_weights": pd.Series(w_target, index=tickers_list),
        "instructions": instr_df
    }


if __name__ == "__main__":
    out = run_reco_pipeline()
    print(f"VIX latest: {out['vix_latest']:.2f}")

    # ---- Format instructions nicely ----
    instr_df = out["instructions"].copy()
    instr_df["current_notional"] = instr_df["current_notional"].round(0).astype(int)
    instr_df["target_notional"] = instr_df["target_notional"].round(0).astype(int)
    instr_df["delta_usd"] = instr_df["delta_usd"].round(0).astype(int)
    instr_df["delta_pct_of_portfolio"] = (instr_df["delta_pct_of_portfolio"] * 100).round(1).astype(str) + "%"

    print("\nExpected returns (ann, est):")
    print((out["expected_returns_ann"] * 100).round(2).astype(str) + "%")

    print("\nTarget weights:")
    print((out["target_weights"] * 100).round(2).astype(str) + "%")

    print("\nTop trade suggestions:")
    print(instr_df.head(20).to_string(index=False))

    # ---- Visualization ----
    # 1. Bar chart of expected returns
    (out["expected_returns_ann"] * 100).sort_values().plot(kind="barh", figsize=(8, 5), color="skyblue")
    plt.title("Expected Annual Returns (Model Estimates)")
    plt.xlabel("Expected Return (%)")
    plt.tight_layout()
    plt.show()

    # 2. Target portfolio weights
    (out["target_weights"] * 100).sort_values().plot(kind="barh", figsize=(8, 5), color="orange")
    plt.title("Optimized Target Weights (%)")
    plt.xlabel("Portfolio Weight (%)")
    plt.tight_layout()
    plt.show()

    # 3. Trade instructions (Buy/Sell/Hold)
    instr_plot = instr_df.copy()
    instr_plot.set_index("ticker")["delta_usd"].sort_values().plot(
        kind="barh", figsize=(8, 5),
        color=instr_plot["action"].map({"BUY": "green", "SELL": "red", "HOLD": "gray"})
    )
    plt.title("Trade Instructions (Δ USD Notional)")
    plt.xlabel("Trade Size (USD)")
    plt.tight_layout()
    plt.show()
        # ---- NLP Summary ----
    nlp_reco = generate_nlp_summary(instr_df, out["vix_latest"])
    print("\nAI Portfolio Summary:")
    print(nlp_reco)


