from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


import model.path_simulator as path_sim
from model.market import Market
from model.mc_pricer import price_american_ls_vector, price_european_naive_mc_vector
from model.option import OptionTrade
from model.path_simulator import simulate_gbm_paths_vector
from model.regression import design_matrix, ols_fit_predict
from typing import Any

# Configuration containers
@dataclass(frozen=True)
class MarketConfig:
    """Inputs needed to build the market object"""
    S0: float
    sigma: float
    r: float

@dataclass(frozen=True)
class TradeConfig:
    """Inputs needed to build the option trade object"""
    K: float
    is_call: bool
    is_american: bool
    pricing_date: dt.date
    maturity_date: dt.date
    div_amount: float = 0.0
    q: float = 0.0
    ex_div_date: dt.date | None = None

@dataclass(frozen=True)
class PricerConfig:
    """Inputs controlling path generation and pricing"""
    n_paths: int = 10_000
    n_steps: int = 50
    seed: int = 42
    antithetic: bool = False
    degree: int = 3


@dataclass(frozen=True)
class OutputConfig:
    """Output file names"""
    out_csv: str = "disc_avg_diagnostic.csv"
    out_xlsx: str = "disc_avg_diagnostic.xlsx"
    out_png: str = "disc_avg_diagnostic.png"

@dataclass(frozen=True)
class DiagnosticConfig:
    market: MarketConfig
    trade: TradeConfig
    pricer: PricerConfig = PricerConfig()
    output: OutputConfig = OutputConfig()


@dataclass(frozen=True)
class DiagnosticContext:
    """Runtime objects and labels derived from the user configuration"""
    market: Market
    trade: OptionTrade
    option_type: str
    exercise_label: str

def _terminal_payoff_table(S: np.ndarray, trade: OptionTrade, n_steps: int) -> tuple[np.ndarray, np.ndarray]:
    """Return terminal payoff and the allocated full value table"""
    V = trade.payoff_vector(S[:, -1])
    V_full = np.empty((S.shape[0], n_steps + 1), dtype=float)
    V_full[:, -1] = V
    return V, V_full

def _european_value_table(
    S: np.ndarray,
    trade: OptionTrade,
    market: Market,
    n_steps: int
) -> np.ndarray:
    """Return the full node value table for a European option"""
    dt_step = float(trade.T) / n_steps
    df = float(np.exp(-float(market.r) * dt_step))
    V, V_full = _terminal_payoff_table(S, trade, n_steps)

    for j in range(n_steps - 1, -1, -1):
        V = df * V
        V_full[:, j] = V
    return V_full

def _american_ls_value_table(
    S: np.ndarray,
    trade: OptionTrade,
    market: Market,
    n_steps: int,
    degree: int = 3,
    basis: str = "laguerre"
) -> np.ndarray:
    """Return the full node value table for an American option"""
    dt_step = float(trade.T) / n_steps
    df = float(np.exp(-float(market.r) * dt_step))
    strike = float(trade.strike)
    V, V_full = _terminal_payoff_table(S, trade, n_steps)

    for j in range(n_steps - 1, -1, -1):
        cont = df * V
        Sj = S[:, j]
        ex = trade.payoff_vector(Sj)
        itm = ex > 0.0

        if itm.any():
            X = design_matrix(Sj[itm], degree=degree, basis=basis, scale=strike)
            fitted_itm = ols_fit_predict(X, cont[itm], X)
            cont[itm] = np.where(ex[itm] >= fitted_itm, ex[itm], cont[itm])

        V_full[:, j] = cont
        V = cont

    return V_full

# Diagnostic Helper
def _build_context(config: DiagnosticConfig) -> DiagnosticContext:
    """Build model objects and display labels from the configuration"""
    market = Market(S0=config.market.S0, r=config.market.r, sigma=config.market.sigma)
    trade = OptionTrade(
        strike=config.trade.K,
        is_call=config.trade.is_call,
        exercise="american" if config.trade.is_american else "european",
        pricing_date=config.trade.pricing_date,
        maturity_date=config.trade.maturity_date,
        q=config.trade.q,
        ex_div_date=config.trade.ex_div_date,
        div_amount=config.trade.div_amount,
    )
    option_type = "Call" if config.trade.is_call else "Put"
    exercise_label = (
        "American (Longstaff-Schwartz)"
        if config.trade.is_american
        else "European"
    )
    return DiagnosticContext(
        market=market,
        trade=trade,
        option_type=option_type,
        exercise_label=exercise_label,
    )


def _print_header(config: DiagnosticConfig, context: DiagnosticContext) -> None:
    """Print a short execution summary before running the pricer"""
    print(f"\n{'=' * 60}")
    print(f"  {context.exercise_label} {context.option_type}")
    print(
        f"  S0={config.market.S0}  K={config.trade.K}  "
        f"r={config.market.r:.2%}  sigma={config.market.sigma:.2%}"
    )
    print(
        f"  N={config.pricer.n_paths:,} paths  "
        f"M={config.pricer.n_steps} steps  seed={config.pricer.seed}"
    )
    print(f"{'=' * 60}")


def _price_option(config: DiagnosticConfig, context: DiagnosticContext) -> float:
    """Run the requested pricing engine and return the time 0 MC price"""
    kwargs = {
        "market": context.market,
        "trade": context.trade,
        "n_paths": config.pricer.n_paths,
        "n_steps": config.pricer.n_steps,
        "seed": config.pricer.seed,
        "antithetic": config.pricer.antithetic,
    }

    if config.trade.is_american:
        price, _ = price_american_ls_vector(
            **kwargs,
            basis="laguerre",
            degree=config.pricer.degree,
        )
    else:
        price, _ = price_european_naive_mc_vector(**kwargs)

    print(f"  Monte Carlo Price (t=0) : {price:.6f}")
    return float(price)

def _load_or_resimulate_paths(
    config: DiagnosticConfig,
    context: DiagnosticContext
) -> tuple[np.ndarray, np.ndarray]:
    """Reuse cached paths when available, otherwise resimulate with the same seed"""
    last_paths = path_sim.LAST_PATHS
    if last_paths is not None and np.asarray(last_paths).ndim == 2:
        S = np.asarray(path_sim.LAST_PATHS, dtype=float)
        times = np.asarray(path_sim.LAST_TIMES, dtype=float)
        return times, S

    print("  [info] re-simulating paths with same seed for diagnostic")
    times, S = simulate_gbm_paths_vector(
        context.market,
        context.trade,
        config.pricer.n_paths,
        config.pricer.n_steps,
        config.pricer.seed,
        config.pricer.antithetic,
    )
    return np.asarray(times, dtype=float), np.asarray(S, dtype=float)

def _discount_vector(r: float, maturity: float, n_steps: int) -> np.ndarray:
    """Return cumulative discount factors from each step back to time 0"""
    dt_step = maturity / n_steps
    df_step = float(np.exp(-r * dt_step))
    return df_step ** np.arange(0, n_steps + 1, dtype=float)

def _build_value_table(
    S: np.ndarray,
    context: DiagnosticContext,
    pricer_config: PricerConfig
) -> np.ndarray:
    """Build the per-node option value table for the selected exercise style"""
    if context.trade.exercise == "american":
        return _american_ls_value_table(
            S,
            context.trade,
            context.market,
            pricer_config.n_steps,
            degree=pricer_config.degree,
        )

    return _european_value_table(S, context.trade, context.market, pricer_config.n_steps)

def _build_output_dataframe(
    times: np.ndarray,
    S: np.ndarray,
    V_table: np.ndarray,
    disc_to_t0: np.ndarray,
    n_steps: int
) -> tuple[np.ndarray, pd.DataFrame]:
    """Assemble the final summary table used for CSV, Excel, and plotting"""
    steps = np.arange(0, n_steps + 1)
    disc_avg_option_price = V_table.mean(axis=0) * disc_to_t0
    disc_avg_spot_price = S.mean(axis=0) * disc_to_t0

    df_out = pd.DataFrame(
        {
            "Step": steps,
            "Time (years)": np.round(times, 6),
            "Discounted Average Option Price": disc_avg_option_price,
            "Discounted Average Spot Price": disc_avg_spot_price,
        }
    )
    return steps, df_out

def _save_csv(df: pd.DataFrame, path: str) -> None:
    """Save the summary table to CSV"""
    df.to_csv(path, index=False, float_format="%.6f")


# Excel export
def _summary_sheet(
    exercise: str,
    option_type: str,
    price: float,
    n_paths: int,
    n_steps: int
) -> pd.DataFrame:
    """Build the compact summary sheet written to Excel"""
    return pd.DataFrame(
        {
            "Parameter": [
                "Exercise",
                "Option Type",
                "Monte Carlo Price (t=0)",
                "Number of Paths",
                "Number of Steps",
            ],
            "Value": [exercise, option_type, round(price, 6), n_paths, n_steps],
        }
    )

def _autofit_worksheet_columns(worksheet: Any) -> None:
    """Resize each worksheet column based on the longest displayed cell value."""
    for col in worksheet.columns:
        max_len = max(
            len(str(cell.value)) if cell.value is not None else 0
            for cell in col
        )
        worksheet.column_dimensions[col[0].column_letter].width = max_len + 4

def _save_excel(
    df: pd.DataFrame,
    path: str,
    exercise: str,
    option_type: str,
    price: float,
    n_paths: int,
    n_steps: int
) -> None:
    """Save summary and diagnostic tables to a formatted Excel workbook"""
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        summary = _summary_sheet(exercise, option_type, price, n_paths, n_steps)
        summary.to_excel(writer, sheet_name="Summary", index=False)
        df.to_excel(writer, sheet_name="Discounted Averages", index=False)

        # Make both sheets easier to read in Excel by resizing each column.
        for sheet_name in writer.sheets:
            worksheet = writer.sheets[sheet_name]
            _autofit_worksheet_columns(worksheet)



# Plot
def _configure_y_axis(axis: Any, label: str, color: str) -> None:
    """Apply consistent formatting to a y-axis"""
    axis.set_ylabel(label, color=color, fontsize=10)
    axis.tick_params(axis="y", labelcolor=color)
    axis.yaxis.set_major_formatter(mticker.ScalarFormatter(useOffset=False))
    axis.ticklabel_format(style="plain", axis="y")

def _plot_option_series(ax: Any, steps: np.ndarray, df_out: pd.DataFrame, price: float, color: str) -> None:
    """Plot the discounted option series and the time 0 MC reference line"""
    ax.plot(
        steps,
        df_out["Discounted Average Option Price"],
        color=color,
        linewidth=1.8,
        label="Discounted Average Option Price",
    )
    ax.axhline(
        price,
        color=color,
        linewidth=0.9,
        linestyle="-.",
        alpha=0.6,
        label=f"Monte Carlo Price (t=0) = {price:.4f}",
    )


def _plot_spot_series(ax: Any, steps: np.ndarray, df_out: pd.DataFrame, color: str) -> None:
    """Plot the discounted spot series on the secondary axis"""
    ax.plot(
        steps,
        df_out["Discounted Average Spot Price"],
        color=color,
        linewidth=1.6,
        linestyle="--",
        label="Discounted Average Spot Price",
    )

def _plot_title(is_american: bool, is_call: bool, n_paths: int, n_steps: int) -> str:
    """Build the plot title"""
    exercise = "American — Longstaff-Schwartz" if is_american else "European"
    option_type = "Call" if is_call else "Put"
    return (
        f"Discounted Average Price per Step — {exercise} {option_type}\n"
        f"N = {n_paths:,} paths,  M = {n_steps} steps"
    )

def _combine_legends(primary_axis: Any, secondary_axis: Any) -> None:
    """Merge legends from both y-axes into one box"""
    lines1, labels1 = primary_axis.get_legend_handles_labels()
    lines2, labels2 = secondary_axis.get_legend_handles_labels()
    primary_axis.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=9)


def _plot(
    df_out: pd.DataFrame,
    steps: np.ndarray,
    is_american: bool,
    is_call: bool,
    price: float,
    n_paths: int,
    n_steps: int,
    out_png: str
) -> None:
    """Save the dual-axis diagnostic plot as a PNG file"""
    fig, ax = plt.subplots(figsize=(12, 5))

    color_option = "#1f77b4"
    color_spot = "#ff7f0e"

    _plot_option_series(ax, steps, df_out, price, color_option)
    _configure_y_axis(ax, "Discounted Average Option Price (to t=0)", color_option)

    ax2 = ax.twinx()
    _plot_spot_series(ax2, steps, df_out, color_spot)
    _configure_y_axis(ax2, "Discounted Average Spot Price", color_spot)

    ax.set_title(_plot_title(is_american, is_call, n_paths, n_steps), fontsize=12)
    ax.set_xlabel("Step  j", fontsize=10)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.grid(True, linestyle="--", alpha=0.35)
    _combine_legends(ax, ax2)

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

# Main diagnostic

def run_diagnostic(config: DiagnosticConfig) -> pd.DataFrame:
    """Run the full diagnostic workflow and return the output DataFrame"""
    context = _build_context(config)
    _print_header(config, context)

    # Calculate price
    price = _price_option(config, context)
    times, S = _load_or_resimulate_paths(config, context)
    disc_to_t0 = _discount_vector(
        r=config.market.r,
        maturity=float(context.trade.T),
        n_steps=config.pricer.n_steps,
    )
    # Build Table Values
    V_table = _build_value_table(S, context, config.pricer)
    steps, df_out = _build_output_dataframe(
        times,
        S,
        V_table,
        disc_to_t0,
        config.pricer.n_steps,
    )

    print(f"{'=' * 60}\n")

    # Save to csv and excel
    _save_csv(df_out, config.output.out_csv)
    _save_excel(
        df_out,
        config.output.out_xlsx,
        context.exercise_label,
        context.option_type,
        price,
        config.pricer.n_paths,
        config.pricer.n_steps,
    )

    # Plot
    _plot(
        df_out=df_out,
        steps=steps,
        is_american=config.trade.is_american,
        is_call=config.trade.is_call,
        price=price,
        n_paths=config.pricer.n_paths,
        n_steps=config.pricer.n_steps,
        out_png=config.output.out_png,
    )
    return df_out

# Run main : example
if __name__ == "__main__":
    config = DiagnosticConfig(
        market=MarketConfig(
            S0=100.0,
            sigma=0.20,
            r=0.10
        ),
        # Trade Configuring
        trade=TradeConfig(
            K=100.0,
            is_call=True,
            is_american=False,
            pricing_date=dt.date(2026, 3, 1),
            maturity_date=dt.date(2026, 12, 25),
            div_amount=3.0,
            q=0.0,
            ex_div_date=dt.date(2026, 11, 30),
        ),
        # Pricer Configuring
        pricer=PricerConfig(
            n_paths=10_000,
            n_steps=250,
            seed=1,
            antithetic=True,
            degree=2,
        ),
        # Output
        output=OutputConfig(
            out_csv="disc_avg_diagnostic.csv",
            out_xlsx="disc_avg_diagnostic.xlsx",
            out_png="disc_avg_diagnostic.png",
        )
    )
    run_diagnostic(config)
