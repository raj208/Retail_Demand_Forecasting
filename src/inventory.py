from dataclasses import dataclass
import numpy as np
from scipy.stats import norm


@dataclass
class InventoryDecision:
    safety_stock: float
    reorder_point: float
    order_up_to: float
    order_qty: float
    z_value: float
    demand_lead_time: float
    demand_protection_period: float


def _pad_forecast(forecast: np.ndarray, needed: int) -> np.ndarray:
    """If we need more days than we have, extend by repeating last known forecast."""
    forecast = np.asarray(forecast, dtype=float)
    if len(forecast) >= needed:
        return forecast
    if len(forecast) == 0:
        return np.zeros(needed, dtype=float)
    pad_val = forecast[-1]
    pad = np.full(needed - len(forecast), pad_val, dtype=float)
    return np.concatenate([forecast, pad])


def inventory_decision(
    forecast_series: np.ndarray,
    sigma_daily: float,
    lead_time_days: int = 7,
    review_days: int = 7,
    service_level: float = 0.95,
    on_hand: float = 0.0,
    on_order: float = 0.0
) -> InventoryDecision:
    L = int(lead_time_days)
    R = int(review_days)
    if L < 1:
        raise ValueError("lead_time_days must be >= 1")
    if R < 0:
        raise ValueError("review_days must be >= 0")
    if not (0.5 < service_level < 0.9999):
        raise ValueError("service_level must be between ~0.5 and 0.9999")

    PP = L + R
    forecast_series = _pad_forecast(forecast_series, PP)

    z = float(norm.ppf(service_level))

    demand_L = float(np.sum(forecast_series[:L]))
    demand_PP = float(np.sum(forecast_series[:PP]))

    sigma_daily = float(max(sigma_daily, 0.0))
    sigma_L = (L ** 0.5) * sigma_daily
    sigma_PP = (PP ** 0.5) * sigma_daily

    rop = demand_L + z * sigma_L
    order_up_to = demand_PP + z * sigma_PP

    order_qty = max(0.0, order_up_to - (float(on_hand) + float(on_order)))

    return InventoryDecision(
        safety_stock=z * sigma_PP,
        reorder_point=rop,
        order_up_to=order_up_to,
        order_qty=order_qty,
        z_value=z,
        demand_lead_time=demand_L,
        demand_protection_period=demand_PP
    )
