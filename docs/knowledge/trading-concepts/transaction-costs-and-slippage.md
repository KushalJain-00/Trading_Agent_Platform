# Transaction Costs and Slippage

## What It Is

Every trade incurs costs beyond the price of the asset itself:

**Transaction cost** (also called **commission**): A fee charged by the broker
for executing the trade. This project uses **5 basis points** (5 bps = 0.05%
of trade value).

**Bid-ask spread** (also called **slippage**): The difference between the
highest price a buyer will pay (bid) and the lowest price a seller will accept
(ask). When you buy, you pay the ask; when you sell, you get the bid. This
project models a **3 basis point spread** (3 bps = 0.03%), split equally
between entry and exit.

A **basis point** (bp) is 0.01%. So:
- 5 bps = 0.05% = ₹50 per ₹100,000 traded
- 100 bps = 1.00%

## Why It Matters for This Project

Transaction costs are the silent killer of high-frequency strategies. This
project discovered this the hard way:

**The cost-eating-P&L problem:**
- At confidence=0.65, holding=10, the strategy generated **75,001 trades**
- Total costs were **₹9.7 billion** against a ₹100M starting capital
- Costs were **251% of gross P&L** — meaning for every ₹1 of gross profit,
  ₹2.51 went to costs
- Net return: **-58.5%**

The high trade count (75K trades over ~1M bars) meant costs compounded rapidly.
Each trade costs:
```
cost = (exit_price × shares × 0.0005) + (shares × exit_price × 0.00015)
```

With confidence filtering (0.85) and minimum holding (75 bars):
- Trade count dropped from 75,001 to 13,269
- Costs dropped from ₹9.7B to ₹1.6B
- Cost % of gross P&L dropped from 251% to 16%
- Net return: **+85%**

## How It's Implemented Here

In `simulator.py`, costs are computed at entry and exit:

```python
# Entry cost (implicit in the entry price):
entry_price = close * (1 + half_spread)  # half_spread = 1.5 bps

# Exit cost (explicit):
exit_price = close * (1 - half_spread)
cost = abs(exit_price * size) * cost_frac + size * (exit_price * half_spread)
```

Where:
- `cost_frac = 5 / 10000` (5 bps transaction cost)
- `half_spread = 3 / 2 / 10000` (1.5 bps per side)
- The cost formula has two terms: commission on exit value + spread cost

The spread is modeled symmetrically: you buy at `close × 1.00015` and sell at
`close × 0.99985`. The commission is charged on the exit.

## Tunable Parameters

| Parameter | Default | What It Does | Effect of Increasing | Effect of Decreasing |
|-----------|---------|--------------|---------------------|---------------------|
| `cost_bps` | 5 | Transaction cost in basis points | Higher costs, more realistic for retail brokers | Lower costs, more optimistic |
| `spread_bps` | 3 | Bid-ask spread in basis points | Higher spread, worse fills | Lower spread, more optimistic |

**Realistic ranges:**
- Institutional: 1-2 bps cost, 1-2 bps spread
- Retail: 5-10 bps cost, 3-5 bps spread
- Illiquid stocks: 10-20 bps cost, 5-15 bps spread

Our defaults (5 bps cost, 3 bps spread) are conservative for retail trading
on NSE.

## Common Pitfalls

1. **Ignoring costs entirely.** A strategy that looks profitable without costs
   may be deeply unprofitable with them. Our initial backtest showed -59%
   before filtering — almost entirely due to costs.

2. **Using unrealistic cost assumptions.** Setting cost_bps=0 gives inflated
   results. Always use costs that match your actual broker and market.

3. **Not accounting for spread.** The spread is often larger than the
   commission, especially for less liquid stocks. Our symmetric spread model
   (1.5 bps per side) is a reasonable approximation.

4. **Costs scale with trade frequency.** A strategy that trades every bar will
   have massive costs. The holding period filter (min_holding_bars=75) reduces
   trade frequency from ~75K to ~13K trades, cutting costs by 83%.
