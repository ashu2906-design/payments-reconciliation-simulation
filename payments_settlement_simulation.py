from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st


def generate_transactions(
    num_transactions: int = 80,
    month_start: str = "2025-01-01",
    seed: int = 42,
) -> pd.DataFrame:
    """Generate payment platform transactions for a single month."""
    rng = np.random.default_rng(seed)

    start = pd.Timestamp(month_start)
    month_end = start + pd.offsets.MonthEnd(1)

    total_seconds = int((month_end + pd.Timedelta(days=1) - start).total_seconds())
    offsets = rng.integers(0, total_seconds, size=num_transactions)
    timestamps = start + pd.to_timedelta(offsets, unit="s")

    amounts = np.round(rng.uniform(5.0, 750.0, size=num_transactions), 2)
    tx_types = rng.choice(["payment", "refund"], size=num_transactions, p=[0.85, 0.15])

    transactions_df = pd.DataFrame(
        {
            "transaction_id": [f"TXN{i:05d}" for i in range(1, num_transactions + 1)],
            "amount": amounts,
            "timestamp": pd.to_datetime(timestamps),
            "type": tx_types,
        }
    ).sort_values("timestamp", ignore_index=True)

    return transactions_df


def generate_settlements(transactions_df: pd.DataFrame, seed: int = 99) -> pd.DataFrame:
    """Create a baseline settlement file that maps each transaction once."""
    rng = np.random.default_rng(seed)

    base = transactions_df[["transaction_id", "amount", "timestamp"]].copy()

    # Simulate 0-3 day settlement lag and keep baseline settlements within Jan 2025.
    lag_days = rng.integers(0, 4, size=len(base))
    settlement_dates = base["timestamp"] + pd.to_timedelta(lag_days, unit="D")
    jan_end = pd.Timestamp("2025-01-31")
    settlement_dates = settlement_dates.where(settlement_dates <= jan_end, jan_end)

    settlements_df = pd.DataFrame(
        {
            "settlement_id": [f"SET{i:05d}" for i in range(1, len(base) + 1)],
            "transaction_id": base["transaction_id"].values,
            "settled_amount": np.round(base["amount"].values, 2),
            "settlement_date": pd.to_datetime(settlement_dates).dt.normalize(),
        }
    )

    return settlements_df


def inject_discrepancies(
    transactions_df: pd.DataFrame, settlements_df: pd.DataFrame, seed: int = 7
) -> pd.DataFrame:
    """Inject intentional reconciliation edge cases."""
    rng = np.random.default_rng(seed)
    altered = settlements_df.copy()

    # 1) Delayed settlement: one Jan transaction settles in Feb 2025.
    delayed_idx = int(rng.integers(0, len(altered)))
    altered.loc[delayed_idx, "settlement_date"] = pd.Timestamp("2025-02-02")

    # 2) Rounding issue: tiny +0.01 shift keeps row plausible but changes totals.
    rounding_idx = (delayed_idx + 1) % len(altered)
    altered.loc[rounding_idx, "settled_amount"] = np.round(
        altered.loc[rounding_idx, "settled_amount"] + 0.01, 2
    )

    # 3) Duplicate settlement entry: append an exact duplicate row.
    duplicate_idx = (rounding_idx + 1) % len(altered)
    duplicate_row = altered.iloc[[duplicate_idx]].copy()
    altered = pd.concat([altered, duplicate_row], ignore_index=True)

    # 4) Orphan refund settlement: negative amount with no source transaction.
    orphan_row = pd.DataFrame(
        [
            {
                "settlement_id": "SET_ORPHAN_REFUND",
                "transaction_id": "TXN99999",
                "settled_amount": -37.42,
                "settlement_date": pd.Timestamp("2025-01-24"),
            }
        ]
    )
    altered = pd.concat([altered, orphan_row], ignore_index=True)

    return altered


def simulate_payments_and_settlements(
    num_transactions: int = 80,
    month_start: str = "2025-01-01",
    transaction_seed: int = 42,
    settlement_seed: int = 99,
    discrepancy_seed: int = 7,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate transactions and settlements with intentional discrepancies."""
    transactions_df = generate_transactions(
        num_transactions=num_transactions,
        month_start=month_start,
        seed=transaction_seed,
    )
    settlements_df = generate_settlements(transactions_df, seed=settlement_seed)
    settlements_df = inject_discrepancies(
        transactions_df=transactions_df,
        settlements_df=settlements_df,
        seed=discrepancy_seed,
    )

    return transactions_df, settlements_df


def _normalize_reconciliation_inputs(
    transactions_df: pd.DataFrame, settlements_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return copies with normalized dtypes used by reconciliation logic."""
    tx = transactions_df.copy()
    st = settlements_df.copy()

    tx["timestamp"] = pd.to_datetime(tx["timestamp"])
    st["settlement_date"] = pd.to_datetime(st["settlement_date"])

    tx["amount"] = tx["amount"].astype(float).round(2)
    st["settled_amount"] = st["settled_amount"].astype(float).round(2)

    return tx, st


def reconcile_transactions_and_settlements(
    transactions_df: pd.DataFrame,
    settlements_df: pd.DataFrame,
    rounding_tolerance: float = 0.05,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Reconcile transactions and settlements and return summary plus issue details."""
    tx, st = _normalize_reconciliation_inputs(transactions_df, settlements_df)

    tx_ids = set(tx["transaction_id"])
    st_ids = set(st["transaction_id"])
    details: List[Dict[str, str]] = []

    # a) Missing settlements
    missing_ids = sorted(tx_ids - st_ids)
    for transaction_id in missing_ids:
        details.append(
            {
                "issue_type": "missing_settlement",
                "transaction_id": transaction_id,
                "explanation": (
                    f"Transaction {transaction_id} exists in transactions but has no matching settlement entry."
                ),
            }
        )

    # b) Settlements with no matching transaction
    orphan_ids = sorted(st_ids - tx_ids)
    for transaction_id in orphan_ids:
        orphan_rows = st[st["transaction_id"] == transaction_id]
        orphan_total = float(orphan_rows["settled_amount"].sum())
        details.append(
            {
                "issue_type": "orphan_settlement",
                "transaction_id": transaction_id,
                "explanation": (
                    f"Settlement references unknown transaction {transaction_id}; "
                    f"orphan settled amount total is {orphan_total:.2f}."
                ),
            }
        )

    # c) Duplicate settlements (same transaction_id appears more than once).
    duplicate_counts = st.groupby("transaction_id").size()
    duplicate_ids = sorted(duplicate_counts[duplicate_counts > 1].index.tolist())
    for transaction_id in duplicate_ids:
        count = int(duplicate_counts.loc[transaction_id])
        details.append(
            {
                "issue_type": "duplicate_settlement",
                "transaction_id": transaction_id,
                "explanation": (
                    f"Transaction {transaction_id} appears {count} times in settlements; "
                    "expected one settlement record per transaction."
                ),
            }
        )

    # d) Amount mismatches for matched transaction IDs.
    matched_ids = sorted(tx_ids.intersection(st_ids))
    tx_amount_by_id = tx.set_index("transaction_id")["amount"]
    settled_amount_by_id = st.groupby("transaction_id")["settled_amount"].sum()

    amount_mismatch_ids: List[str] = []
    for transaction_id in matched_ids:
        tx_amount = float(tx_amount_by_id.loc[transaction_id])
        settled_amount = float(settled_amount_by_id.loc[transaction_id])
        if not np.isclose(tx_amount, settled_amount, atol=0.000001):
            amount_mismatch_ids.append(transaction_id)
            details.append(
                {
                    "issue_type": "amount_mismatch",
                    "transaction_id": transaction_id,
                    "explanation": (
                        f"Transaction {transaction_id} amount is {tx_amount:.2f} but total settled amount is "
                        f"{settled_amount:.2f} (difference {settled_amount - tx_amount:+.2f})."
                    ),
                }
            )

    # e) Transactions settled in a different month.
    tx_month_by_id = tx.set_index("transaction_id")["timestamp"].dt.to_period("M")
    st_month_by_id = st.groupby("transaction_id")["settlement_date"].min().dt.to_period("M")

    cross_month_ids: List[str] = []
    for transaction_id in matched_ids:
        tx_month = tx_month_by_id.loc[transaction_id]
        settlement_month = st_month_by_id.loc[transaction_id]
        if tx_month != settlement_month:
            cross_month_ids.append(transaction_id)
            details.append(
                {
                    "issue_type": "cross_month_settlement",
                    "transaction_id": transaction_id,
                    "explanation": (
                        f"Transaction {transaction_id} occurred in {tx_month} but earliest settlement is in "
                        f"{settlement_month}."
                    ),
                }
            )

    tx_total = float(tx["amount"].sum())
    st_total = float(st["settled_amount"].sum())
    total_difference = st_total - tx_total
    aggregate_rounding_discrepancy = (
        abs(total_difference) > 0 and abs(total_difference) <= rounding_tolerance
    )

    if aggregate_rounding_discrepancy:
        details.append(
            {
                "issue_type": "aggregate_rounding_discrepancy",
                "transaction_id": "AGGREGATE",
                "explanation": (
                    f"Aggregate totals differ by {total_difference:+.2f}, within rounding tolerance "
                    f"{rounding_tolerance:.2f}. This suggests a small rounding-level imbalance."
                ),
            }
        )

    details_df = pd.DataFrame(details, columns=["issue_type", "transaction_id", "explanation"])

    summary: Dict[str, Any] = {
        "total_transactions": int(len(tx)),
        "total_settlements": int(len(st)),
        "missing_settlement_count": len(missing_ids),
        "orphan_settlement_count": len(orphan_ids),
        "duplicate_settlement_count": len(duplicate_ids),
        "amount_mismatch_count": len(amount_mismatch_ids),
        "cross_month_settlement_count": len(cross_month_ids),
        "transaction_total_amount": round(tx_total, 2),
        "settlement_total_amount": round(st_total, 2),
        "total_amount_difference": round(total_difference, 2),
        "aggregate_rounding_discrepancy": aggregate_rounding_discrepancy,
        "rounding_tolerance": rounding_tolerance,
    }

    return summary, details_df


def run_streamlit_app() -> None:
    """Render a simple Streamlit UI for data generation and reconciliation."""
    st.set_page_config(page_title="Payments Reconciliation", layout="wide")
    st.title("Payments vs Bank Settlement Reconciliation")
    st.write("Generate sample data and inspect reconciliation results.")

    if st.button("Generate Data", type="primary"):
        transactions_df, settlements_df = simulate_payments_and_settlements()
        summary, details_df = reconcile_transactions_and_settlements(
            transactions_df=transactions_df,
            settlements_df=settlements_df,
        )

        st.session_state["transactions_df"] = transactions_df
        st.session_state["settlements_df"] = settlements_df
        st.session_state["summary"] = summary
        st.session_state["details_df"] = details_df

    if "summary" not in st.session_state:
        st.info("Click 'Generate Data' to run the simulation and reconciliation.")
        return

    summary = st.session_state["summary"]
    details_df = st.session_state["details_df"]

    st.subheader("1) Totals")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Transactions", int(summary["total_transactions"]))
    with col2:
        st.metric("Total Settlements", int(summary["total_settlements"]))

    st.subheader("2) Total Amounts Comparison")
    col3, col4, col5 = st.columns(3)
    with col3:
        st.metric("Transaction Total", f"{summary['transaction_total_amount']:.2f}")
    with col4:
        st.metric("Settlement Total", f"{summary['settlement_total_amount']:.2f}")
    with col5:
        st.metric("Difference", f"{summary['total_amount_difference']:+.2f}")

    st.subheader("3) Reconciliation Summary")
    st.json(summary)

    st.subheader("4) Detailed Discrepancies")
    if details_df.empty:
        st.success("No discrepancies found.")
    else:
        st.dataframe(details_df, use_container_width=True)


if __name__ == "__main__":
    run_streamlit_app()
