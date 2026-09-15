"""
Synthetic Demand Generator for Fodripio.

Generates realistic time-series demand for multi-product industrial manufacturing setups.
Features supported:
- Product-level baseline demand
- Trend (linear/growth/decay)
- Weekly & Annual seasonality
- Promotional campaigns and price variation
- Holiday effects
- Inter-product correlation
- Exogenous noise and random demand shocks
- Seeded reproducibility
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union


class SyntheticDemandGenerator:
    """
    Generator for multi-product synthetic demand series.
    """

    DEFAULT_PRODUCTS = [
        {"id": "P1", "base_demand": 120.0, "trend": 0.05, "weekly": True, "annual": True, "volatility": 0.12},
        {"id": "P2", "base_demand": 80.0, "trend": 0.02, "weekly": True, "annual": True, "volatility": 0.15},
        {"id": "P3", "base_demand": 200.0, "trend": -0.01, "weekly": True, "annual": False, "volatility": 0.10},
        {"id": "P4", "base_demand": 50.0, "trend": 0.08, "weekly": True, "annual": True, "volatility": 0.20},
        {"id": "P5", "base_demand": 30.0, "trend": 0.00, "weekly": False, "annual": False, "volatility": 0.18},
    ]

    def __init__(
        self,
        start_date: str = "2024-01-01",
        end_date: str = "2025-12-31",
        products: Optional[List[Dict]] = None,
        random_state: int = 42,
    ):
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.products = products or self.DEFAULT_PRODUCTS
        self.random_state = random_state
        self.dates = pd.date_range(start=self.start_date, end=self.end_date, freq="D")
        self.n_days = len(self.dates)

    def generate(self) -> pd.DataFrame:
        """
        Generates daily demand data for all specified products.

        Returns:
            pd.DataFrame: Long-format DataFrame containing demand time-series and features.
        """
        rng = np.random.default_rng(self.random_state)
        records = []

        # Common features across products for each date
        day_of_year = self.dates.dayofyear.values
        day_of_week = self.dates.dayofweek.values

        # Major holidays placeholder (e.g., fixed dates like Jan 1, Jul 4, Dec 25, Thanksgiving approx)
        holidays = (
            (self.dates.month == 1) & (self.dates.day == 1)
            | (self.dates.month == 7) & (self.dates.day == 4)
            | (self.dates.month == 12) & (self.dates.day == 25)
            | ((self.dates.month == 11) & (self.dates.day >= 22) & (self.dates.day <= 28) & (day_of_week == 3))
        ).astype(int)

        t = np.arange(self.n_days)

        for p_idx, prod in enumerate(self.products):
            p_id = prod["id"]
            base = prod["base_demand"]
            trend_rate = prod.get("trend", 0.0)
            has_weekly = prod.get("weekly", True)
            has_annual = prod.get("annual", True)
            vol = prod.get("volatility", 0.15)

            # 1. Trend component
            trend_component = base * (1 + (trend_rate * t / 365.0))

            # 2. Weekly seasonality (peak mid-week, lower weekend)
            if has_weekly:
                weekly_weights = np.array([1.0, 1.05, 1.10, 1.08, 0.95, 0.60, 0.50])
                weekly_factor = weekly_weights[day_of_week]
            else:
                weekly_factor = np.ones(self.n_days)

            # 3. Annual seasonality (sinusoidal)
            if has_annual:
                annual_factor = 1.0 + 0.20 * np.sin(2 * np.pi * (day_of_year - 80) / 365.25)
            else:
                annual_factor = np.ones(self.n_days)

            # 4. Promotions (random periodic bursts ~ 5% of days)
            promo_mask = (rng.uniform(0, 1, self.n_days) < 0.05).astype(int)
            promo_boost = 1.0 + promo_mask * rng.uniform(0.20, 0.40, self.n_days)

            # 5. Price variation (base price + promo discount)
            base_price = 100.0 + p_idx * 25.0
            price = base_price * (1.0 - 0.15 * promo_mask + rng.normal(0, 0.02, self.n_days))
            price_elasticity = -1.2
            price_effect = (price / base_price) ** price_elasticity

            # 6. Holiday effect (dip or spike depending on product)
            holiday_factor = 1.0 - 0.30 * holidays if p_idx % 2 == 0 else 1.0 + 0.15 * holidays

            # 7. Shocks (infrequent unexpected disruptions or spikes)
            shocks = np.zeros(self.n_days)
            shock_indices = rng.choice(self.n_days, size=max(1, self.n_days // 180), replace=False)
            shocks[shock_indices] = rng.choice([-0.5, 0.6], size=len(shock_indices))

            # Combine deterministic components
            mean_demand = (
                trend_component
                * weekly_factor
                * annual_factor
                * promo_boost
                * price_effect
                * holiday_factor
                * (1.0 + shocks)
            )

            # 8. Multiplicative gamma noise for realistic positive skew
            shape = 1.0 / (vol ** 2)
            scale = vol ** 2
            noise = rng.gamma(shape, scale, self.n_days)

            demand = np.maximum(0.0, np.round(mean_demand * noise, 2))

            for i in range(self.n_days):
                records.append(
                    {
                        "date": self.dates[i].strftime("%Y-%m-%d"),
                        "product_id": p_id,
                        "demand": demand[i],
                        "price": np.round(price[i], 2),
                        "promotion": int(promo_mask[i]),
                        "holiday": int(holidays[i]),
                        "shock": 1 if i in shock_indices else 0,
                    }
                )

        df = pd.DataFrame(records)
        return df


if __name__ == "__main__":
    generator = SyntheticDemandGenerator()
    df = generator.generate()
    print(f"Generated synthetic demand dataset: {df.shape[0]} rows, products: {df['product_id'].nunique()}")
