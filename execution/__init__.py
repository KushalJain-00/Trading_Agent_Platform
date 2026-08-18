"""Broker interface — stub for future real broker connection.

DO NOT CONNECT TO ANY REAL BROKER. This is paper-trading only.
Defines the interface a real broker adapter would need to implement.

PAPER TRADING ONLY — NO REAL MONEY.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class Order:
    order_id: str
    ticker: str
    side: str          # "BUY" or "SELL"
    quantity: float
    price: float
    order_type: str    # "MARKET" or "LIMIT"
    status: str        # "PENDING" | "FILLED" | "CANCELLED" | "REJECTED"
    filled_price: Optional[float] = None
    filled_quantity: Optional[float] = None
    timestamp: Optional[str] = None


@dataclass
class Position:
    ticker: str
    quantity: float
    avg_price: float
    current_price: float
    unrealized_pnl: float


@dataclass
class AccountBalance:
    total: float
    available: float
    used_margin: float
    unrealized_pnl: float


class BrokerInterface(ABC):
    """Abstract broker interface — implement for each real broker.

    PAPER TRADING ONLY. Do not use with real money.
    """

    @abstractmethod
    def place_order(self, ticker: str, side: str, quantity: float,
                    price: float = 0, order_type: str = "MARKET") -> Order:
        """Place a buy/sell order."""
        ...

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order."""
        ...

    @abstractmethod
    def get_positions(self) -> list[Position]:
        """Get all open positions."""
        ...

    @abstractmethod
    def get_account_balance(self) -> AccountBalance:
        """Get account balance and margin info."""
        ...

    @abstractmethod
    def get_order_status(self, order_id: str) -> Optional[Order]:
        """Check status of an order."""
        ...

    @abstractmethod
    def get_recent_trades(self, limit: int = 50) -> list[Order]:
        """Get recent filled trades."""
        ...


class PaperBroker(BrokerInterface):
    """In-memory paper trading broker — no real execution.

    PAPER TRADING ONLY. No real money at risk.
    """

    def __init__(self, initial_balance: float = 100_000_000):
        self._balance = initial_balance
        self._positions: dict[str, Position] = {}
        self._orders: list[Order] = []
        self._order_counter = 0

    def place_order(self, ticker, side, quantity, price=0, order_type="MARKET"):
        self._order_counter += 1
        order = Order(
            order_id=f"PAPER-{self._order_counter:06d}",
            ticker=ticker, side=side.upper(), quantity=quantity,
            price=price, order_type=order_type, status="FILLED",
            filled_price=price, filled_quantity=quantity,
        )
        self._orders.append(order)
        return order

    def cancel_order(self, order_id):
        return False

    def get_positions(self):
        return list(self._positions.values())

    def get_account_balance(self):
        return AccountBalance(self._balance, self._balance, 0.0, 0.0)

    def get_order_status(self, order_id):
        for o in self._orders:
            if o.order_id == order_id:
                return o
        return None

    def get_recent_trades(self, limit=50):
        return self._orders[-limit:]
