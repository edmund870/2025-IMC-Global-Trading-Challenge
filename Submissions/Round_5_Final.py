from datamodel import (
    OrderDepth,
    UserId,
    TradingState,
    Order,
    Listing,
    Observation,
    ProsperityEncoder,
    Symbol,
    Trade,
)

from collections import deque
import copy
import numpy as np
import math
import json

from typing import Any


empty_dict_cache = {
    "KELP": deque(maxlen=100),
    "RAINFOREST_RESIN": deque(maxlen=200),
    "SQUID_INK": deque(maxlen=100),
    "CROISSANTS": deque(maxlen=1000),
    "JAMS": deque(maxlen=1000),
    "PICNIC_BASKET1": deque(maxlen=1000),
    "PICNIC_BASKET2": deque(maxlen=1000),
    "DJEMBES": deque(maxlen=1000),
    "VOLCANIC_ROCK": deque(maxlen=1000),
    "VOLCANIC_ROCK_VOUCHER_9500": deque(maxlen=1000),
    "VOLCANIC_ROCK_VOUCHER_9750": deque(maxlen=1000),
    "VOLCANIC_ROCK_VOUCHER_10000": deque(maxlen=1000),
    "VOLCANIC_ROCK_VOUCHER_10250": deque(maxlen=1000),
    "VOLCANIC_ROCK_VOUCHER_10500": deque(maxlen=1000),
    "MAGNIFICENT_MACARONS": deque(maxlen=1000),
}


class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(
        self,
        state: TradingState,
        orders: dict[Symbol, list[Order]],
        conversions: int,
        trader_data: str,
    ) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(
                        state, self.truncate(state.traderData, max_item_length)
                    ),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])

        return compressed

    def compress_order_depths(
        self, order_depths: dict[Symbol, OrderDepth]
    ) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[: max_length - 3] + "..."


logger = Logger()


class tradable_product:
    RAINFOREST_RESIN = "RAINFOREST_RESIN"
    KELP = "KELP"
    SQUID_INK = "SQUID_INK"
    CROISSANTS = "CROISSANTS"
    JAMS = "JAMS"
    PICNIC_BASKET1 = "PICNIC_BASKET1"
    PICNIC_BASKET2 = "PICNIC_BASKET2"
    DJEMBES = "DJEMBES"
    VOLCANIC_ROCK = "VOLCANIC_ROCK"
    VOLCANIC_ROCK_VOUCHER_9500 = "VOLCANIC_ROCK_VOUCHER_9500"
    VOLCANIC_ROCK_VOUCHER_9750 = "VOLCANIC_ROCK_VOUCHER_9750"
    VOLCANIC_ROCK_VOUCHER_10000 = "VOLCANIC_ROCK_VOUCHER_10000"
    VOLCANIC_ROCK_VOUCHER_10250 = "VOLCANIC_ROCK_VOUCHER_10250"
    VOLCANIC_ROCK_VOUCHER_10500 = "VOLCANIC_ROCK_VOUCHER_10500"
    MAGNIFICENT_MACARONS = "MAGNIFICENT_MACARONS"


class position_limit:
    KELP = 50
    RAINFOREST_RESIN = 50
    SQUID_INK = 50
    CROISSANTS = 250
    JAMS = 350
    PICNIC_BASKET1 = 60
    PICNIC_BASKET2 = 100
    DJEMBES = 60
    VOLCANIC_ROCK = 400
    VOLCANIC_ROCK_VOUCHER_9500 = 200
    VOLCANIC_ROCK_VOUCHER_9750 = 200
    VOLCANIC_ROCK_VOUCHER_10000 = 200
    VOLCANIC_ROCK_VOUCHER_10250 = 200
    VOLCANIC_ROCK_VOUCHER_10500 = 200
    MAGNIFICENT_MACARONS = 75
    MAGNIFICENT_MACARONS_CONVERSION = 10


class traders:
    Caesar = "Caesar"
    Camilla = "Camilla"
    Charlie = "Charlie"
    Gary = "Gary"
    Gina = "Gina"
    Olga = "Olga"
    Olivia = "Olivia"
    Pablo = "Pablo"
    Paris = "Paris"
    Penelope = "Penelope"
    Peter = "Peter"


class quotes:
    bid = "bid"
    ask = "ask"


class VECM:
    def __init__(self, endog: np.ndarray) -> None:
        self.y = endog.T
        self.k_ar_diff = 1
        self.coint_rank = 1

    def _sij(
        self, delta_x: np.ndarray, delta_y_1_T: np.ndarray, y_lag1: np.ndarray
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        nobs = y_lag1.shape[1]
        r0, r1 = self._r_matrices(delta_y_1_T, y_lag1, delta_x)

        if r0 is None:
            return None, None, None, None, None, None, None

        s00 = np.dot(r0, r0.T) / nobs
        s01 = np.dot(r0, r1.T) / nobs
        s10 = s01.T
        s11 = np.dot(r1, r1.T) / nobs

        if np.linalg.det(self._mat_sqrt(s11)) == 0:
            return None, None, None, None, None, None, None

        s11_ = np.linalg.inv(self._mat_sqrt(s11))
        s01_s11_ = np.dot(s01, s11_)

        if np.linalg.det(s00) == 0:
            return None, None, None, None, None, None, None
        elif np.linalg.det(s01_s11_.T @ np.linalg.inv(s00) @ s01_s11_) == 0:
            return None, None, None, None, None, None, None

        eig = np.linalg.eig(s01_s11_.T @ np.linalg.inv(s00) @ s01_s11_)
        lambd = eig[0]
        v = eig[1]
        lambd_order = np.argsort(lambd)[::-1]
        lambd = lambd[lambd_order]
        v = v[:, lambd_order]
        return s00, s01, s10, s11, s11_, lambd, v

    def _mat_sqrt(self, _2darray: np.ndarray) -> np.ndarray:
        u_, s_, v_ = np.linalg.svd(_2darray, full_matrices=False)
        s_ = np.sqrt(s_)
        return u_.dot(s_[:, None] * v_)

    def _r_matrices(
        self, delta_y_1_T: np.ndarray, y_lag1: np.ndarray, delta_x: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if np.linalg.det(delta_x.dot(delta_x.T)) == 0:
            return None, None
        nobs = y_lag1.shape[1]
        m = np.identity(nobs) - (
            delta_x.T.dot(np.linalg.inv(delta_x.dot(delta_x.T))).dot(delta_x)
        )
        r0 = delta_y_1_T.dot(m)
        r1 = y_lag1.dot(m)
        return r0, r1

    def _endog_matrices(
        self,
        endog: np.ndarray,
        diff_lags: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        p = diff_lags + 1
        y = endog
        K = y.shape[0]
        y_1_T = y[:, p:]
        T = y_1_T.shape[1]
        delta_y = np.diff(y)
        delta_y_1_T = delta_y[:, p - 1 :]

        y_lag1 = y[:, p - 1 : -1]
        y_lag1_stack = [y_lag1]
        y_lag1 = np.row_stack(y_lag1_stack)

        delta_x = np.zeros((diff_lags * K, T))
        if diff_lags > 0:
            for j in range(delta_x.shape[1]):
                delta_x[:, j] = delta_y[
                    :, j + p - 2 : None if j - 1 < 0 else j - 1 : -1
                ].T.reshape(K * (p - 1))
        delta_x_stack = [delta_x]
        delta_x = np.row_stack(delta_x_stack)

        return y_1_T, delta_y_1_T, y_lag1, delta_x

    def _estimate_vecm_ml(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        y_1_T, delta_y_1_T, y_lag1, delta_x = self._endog_matrices(
            self.y,
            self.k_ar_diff,
        )

        _, s01, _, s11, s11_, _, v = self._sij(delta_x, delta_y_1_T, y_lag1)

        if any([i is None for i in [_, s01, _, s11, s11_, _, v]]):
            return None, None, None

        beta_tilde = (v[:, : self.coint_rank].T.dot(s11_)).T
        beta_tilde = np.real_if_close(beta_tilde)
        beta_tilde = np.dot(beta_tilde, np.linalg.inv(beta_tilde[: self.coint_rank]))
        alpha_tilde = s01.dot(beta_tilde).dot(
            np.linalg.inv(beta_tilde.T.dot(s11).dot(beta_tilde))
        )
        gamma_tilde = (
            (delta_y_1_T - alpha_tilde.dot(beta_tilde.T).dot(y_lag1))
            .dot(delta_x.T)
            .dot(np.linalg.inv(np.dot(delta_x, delta_x.T)))
        )

        return alpha_tilde, beta_tilde, gamma_tilde


class options:
    def __init__(self, S: int, K: int, r: float, sigma: float, T: float) -> None:
        self.S, self.K, self.r, self.sigma, self.T = S, K, r, sigma, T
        self.S_cache = S
        self.iv_cache = sigma

    def phi(self, x: float) -> float:
        return (1.0 + math.erf(x / np.sqrt(2.0))) / 2.0

    def black_scholes_call(
        self,
    ) -> float:
        d1 = (np.log(self.S / self.K) + (self.r + self.sigma**2 / 2) * self.T) / (
            self.sigma * np.sqrt(self.T)
        )
        d2 = d1 - self.sigma * np.sqrt(self.T)
        return self.S * self.phi(d1) - self.K * np.exp(-self.r * self.T) * self.phi(d2)

    def call_delta(
        self,
    ) -> float:
        d1 = (np.log(self.S / self.K) + (self.r + self.sigma**2 / 2) * self.T) / (
            self.sigma * np.sqrt(self.T)
        )
        return self.phi(d1)

    def vega(
        self,
    ) -> float:
        d1 = (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma**2) * self.T) / (
            self.sigma * np.sqrt(self.T)
        )
        return self.S * np.exp(-0.5 * d1**2) / np.sqrt(2 * np.pi) * np.sqrt(self.T)

    def solve_iv(
        self,
        call_price: int,
        initial_guess: float = 0.2,
        max_iter: int = 1000,
        tol: float = 1e-6,
    ) -> float:
        self.sigma = initial_guess
        for _ in range(max_iter):
            f_val = self.black_scholes_call() - call_price
            f_prime_val = self.vega()
            if abs(f_prime_val) < tol:
                break
            self.sigma = self.sigma - f_val / f_prime_val
            if abs(f_val) < tol:
                break
        return self.sigma


class orderbook:
    def __init__(self, order_depth: OrderDepth) -> None:
        """
        orderbook variables

        Parameters
        ----------
        order_depth : OrderDepth
            order book containing bids and asks
        """

        self.order_depth = order_depth
        self.bids = order_depth.buy_orders.items()
        self.asks = order_depth.sell_orders.items()

    def top_of_book(self, quote: quotes) -> tuple[int, int]:
        if quote == quotes.bid:
            return list(self.bids)[0] if len(self.bids) > 0 else [0, 0]
        if quote == quotes.ask:
            return list(self.asks)[0] if len(self.asks) > 0 else [1e6, 0]

    def get_level(self, quote: quotes, level) -> list[int, int]:
        if quote == quotes.bid:
            return list(self.bids)[level] if len(self.bids) > 0 else [0, 0]
        if quote == quotes.ask:
            return list(self.asks)[level] if len(self.asks) > 0 else [1e6, 0]

    def high_volume_quotes(self, quote: quotes) -> tuple[int, int]:
        if quote == quotes.bid:
            return (
                max(self.bids, key=lambda tup: tup[1]) if len(self.bids) > 0 else [0, 0]
            )
        if quote == quotes.ask:
            return (
                min(self.asks, key=lambda tup: tup[1])
                if len(self.asks) > 0
                else [1e6, 0]
            )

    def spread(self) -> int:
        return abs(
            self.top_of_book(quote=quotes.bid)[0]
            - self.top_of_book(quote=quotes.ask)[0]
        )

    def most_micro_price(self) -> float:
        most_ask, most_ask_vol = self.high_volume_quotes(quote=quotes.ask)
        most_bid, most_bid_vol = self.high_volume_quotes(quote=quotes.bid)
        return (most_ask * most_bid_vol + most_bid * abs(most_ask_vol)) / (
            abs(most_ask_vol) + most_bid_vol
        )

    def micro_price(self) -> float:
        best_ask, best_ask_vol = self.top_of_book(quote=quotes.ask)
        best_bid, best_bid_vol = self.top_of_book(quote=quotes.bid)
        return (best_ask * best_bid_vol + best_bid * abs(best_ask_vol)) / (
            abs(best_ask_vol) + best_bid_vol
        )

    def most_mid_price(self) -> float:
        most_ask, _ = self.high_volume_quotes(quote=quotes.ask)
        most_bid, _ = self.high_volume_quotes(quote=quotes.bid)
        return (most_ask + most_bid) / 2

    def mid_price(self) -> float:
        best_ask, _ = self.top_of_book(quote=quotes.ask)
        best_bid, _ = self.top_of_book(quote=quotes.bid)
        return (best_ask + best_bid) / 2

    def total_volume(self, quote: quotes) -> int:
        volume = 0
        if quote == quotes.bid:
            for _, vol in self.bids:
                volume += vol

        if quote == quotes.ask:
            for _, vol in self.asks:
                volume += abs(vol)

        return volume

    def skew(self) -> float:
        bid_vol = self.total_volume(quote=quotes.bid)
        ask_vol = self.total_volume(quote=quotes.ask)
        return (bid_vol - ask_vol) / (ask_vol + bid_vol)

    def volume_ratio(self) -> float:
        bid_vol = self.total_volume(quote=quotes.bid)
        ask_vol = self.total_volume(quote=quotes.ask)
        return bid_vol / ask_vol

    def depth(self, quote: quotes) -> int:
        if quote == quotes.bid:
            return len(self.bids)
        if quote == quotes.ask:
            return len(self.asks)


class fair_value:
    def __init__(
        self, product: tradable_product, product_params, order_book: orderbook
    ) -> None:
        self.product = product
        self.product_params = product_params
        self.order_book = order_book

    def RAINFOREST_RESIN_FV(
        self,
    ) -> int:
        base = 10_000
        return base

    def SQUID_FV(
        self, curr_mid_kelp: int, mid_cache_kelp: deque, mid_cache_squid: deque
    ) -> tuple[float, float]:
        if len(mid_cache_kelp) < 10 or len(mid_cache_squid) < 10:
            return 1, np.ones(len(mid_cache_kelp))

        mid_kelp = np.array(mid_cache_kelp)
        y = np.array(mid_cache_squid)

        if len(mid_kelp) != len(y):
            mid_kelp = mid_kelp[:-1]

        x = np.column_stack((np.ones(mid_kelp.shape[0]), mid_kelp))

        coeffs = np.linalg.inv(x.T @ x) @ x.T @ y

        spread = self.order_book.mid_price() - curr_mid_kelp * coeffs[1] + coeffs[0]
        preds = y - mid_kelp * coeffs[1] + coeffs[0]

        return spread, preds

    def PICNIC1_FV(
        self, basket_order_depth: dict[tradable_product, orderbook]
    ) -> float:
        cross = (
            basket_order_depth[tradable_product.CROISSANTS].mid_price()
            * self.product_params[tradable_product.CROISSANTS]
        )

        jams = (
            basket_order_depth[tradable_product.JAMS].mid_price()
            * self.product_params[tradable_product.JAMS]
        )

        dj = (
            basket_order_depth[tradable_product.DJEMBES].mid_price()
            * self.product_params[tradable_product.DJEMBES]
        )

        return cross + jams + dj

    def PICNIC2_FV(
        self, basket_order_depth: dict[tradable_product, orderbook]
    ) -> float:
        cross = (
            basket_order_depth[tradable_product.CROISSANTS].mid_price()
            * self.product_params[tradable_product.CROISSANTS]
        )
        jams = (
            basket_order_depth[tradable_product.JAMS].mid_price()
            * self.product_params[tradable_product.JAMS]
        )

        return cross + jams

    def VECM(
        self,
        alpha: np.ndarray,
        beta: np.ndarray,
        gamma: np.ndarray,
        current_price: int,
        prev_mid: int,
    ) -> np.ndarray:
        return alpha @ (beta @ current_price) + gamma @ (current_price - prev_mid)

    def VECM_FORECAST(
        self,
        prod_1: tradable_product,
        prod_1_mid: int,
        prod_2: tradable_product,
        prod_2_mid: int,
        mid_cache: dict[tradable_product, deque],
    ) -> tuple[np.ndarray, np.ndarray]:
        most_mid_price = np.array([prod_1_mid, prod_2_mid])

        alpha, beta, gamma = (
            self.product_params["alpha"],
            self.product_params["beta"],
            self.product_params["gamma"],
        )

        if len(mid_cache[self.product]) < 1:
            prev_mid = most_mid_price
        else:
            prev_mid = np.array(
                [
                    mid_cache[prod_1][-1],
                    mid_cache[prod_2][-1],
                ]
            )

        delta = self.VECM(
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            current_price=most_mid_price,
            prev_mid=prev_mid,
        )

        prod_1_forecast = prod_1_mid + delta[0]
        prod_2_forecast = prod_2_mid + delta[1]

        forecast = np.array([prod_1_forecast, prod_2_forecast])

        return delta, forecast

    def CALL_LB(self, payoff: int, call_price: int) -> int:
        return call_price - payoff  # if call price < payoff arbv

    def THEO_PRICE(
        self, S: int, K: int, r: float, TTE: float, mt_cache: deque, coeffs: np.ndarray
    ) -> tuple[float, float]:
        if coeffs is not None:
            x = np.array([mt_cache[-1] ** 2, mt_cache[-1], 1])

            iv_interpolated = sum(coeffs * x)

            option = options(S=S, K=K, r=r, sigma=iv_interpolated, T=TTE)

            price = option.black_scholes_call()
            delta = option.call_delta()

            return price, delta
        else:
            return self.order_book.mid_price(), 0

    def VECM_MACAROONS(
        self,
        mid: float,
        pristine_mid: float,
        sunlight: float,
        mid_cache: dict[tradable_product, deque],
        pristine_mid_cache: deque,
        sunlight_cache: deque,
    ) -> tuple[float, float, float]:
        most_mid_price = np.array([mid, pristine_mid, sunlight])

        alpha, beta, gamma = (
            self.product_params["alpha"],
            self.product_params["beta"],
            self.product_params["gamma"],
        )

        if len(mid_cache[self.product]) < 1:
            prev_mid = most_mid_price
        else:
            prev_mid = np.array(
                [
                    mid_cache[self.product][-1],
                    pristine_mid_cache[-1],
                    sunlight_cache[-1],
                ]
            )

        delta = self.VECM(
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            current_price=most_mid_price,
            prev_mid=prev_mid,
        )

        prod_1_forecast = mid + delta[0]
        prod_2_forecast = pristine_mid + delta[1]

        return delta[0], prod_1_forecast, prod_2_forecast


class trading_strategy:
    def __init__(
        self,
        product: tradable_product,
        product_params: dict,
        position: int,
        position_limit: int,
        order_book: orderbook,
        fair_value: int | float,
        buyer: dict[str, float],
        seller: dict[str, float],
    ) -> None:
        self.product = product
        self.position = position
        self.product_params = product_params
        self.position_pct = position / position_limit
        self.position_limit = position_limit
        self.max_buy = position_limit - position
        self.max_sell = position_limit + position
        self.fair_value = fair_value
        self.order_book = order_book
        self.max_buy_price = fair_value + product_params.get("ask_slip", 0)
        self.max_sell_price = fair_value + product_params.get("bid_slip", 0)

        self.z = 0
        self.threshold = self.product_params.get("z_threshold", 0)
        self.exit_threshold = self.product_params.get("exit_threshold", 0)
        self.orders = []

        self.buyer = buyer
        self.seller = seller
        self.copy_buy = any(
            [
                b
                for b, q in buyer.items()
                if b in self.product_params["trade_with"].keys()
                and q in self.product_params["trade_with"].get(b, [])
            ]
        ) or any(
            [
                s
                for s, q in seller.items()
                if s in self.product_params["trade_against"].keys()
                and q in self.product_params["trade_against"].get(s, [])
            ]
        )

        self.copy_sell = any(
            [
                s
                for s, q in seller.items()
                if s in self.product_params["trade_with"].keys()
                and q in self.product_params["trade_with"].get(s, [])
            ]
        ) or any(
            [
                b
                for b, q in buyer.items()
                if b in self.product_params["trade_against"].keys()
                and q in self.product_params["trade_against"].get(b, [])
            ]
        )

    def update_FV(self, FV: float) -> None:
        self.fair_value = FV
        self.max_buy_price = FV + self.product_params.get("ask_slip", 0)
        self.max_sell_price = FV + self.product_params.get("bid_slip", 0)

    def z_score(self, value: float, sample: np.ndarray) -> float:
        return (value - sample.mean()) / (sample.std() + 1e-16)

    def ladder(self, qty: int, n: int, decay: float) -> list[int]:
        raw_values = [math.exp(-decay * i) for i in range(1, n + 1)]
        sum_raw = sum(raw_values)
        ladder = [math.floor(qty * value / sum_raw) for value in raw_values]
        return ladder

    def ladder_orders(
        self, max_price: float, highest_trade: float, quote: quotes, decay: float
    ) -> None:
        if quote == quotes.bid:
            highest_bid = self.order_book.get_level(quote=quotes.bid, level=0)[0]
            worst_buy = max(max_price, highest_trade, highest_bid)
            best_buy = min(max_price, highest_trade, highest_bid)
            bid_range = int(abs(best_buy - worst_buy))
            if bid_range == 0:
                bid_range = self.order_book.spread()
            bid_qty = self.ladder(self.max_buy, n=bid_range, decay=decay)
            for i in range(bid_range):
                if bid_qty[i] != 0:
                    self.orders.append(
                        Order(self.product, int(best_buy + i), bid_qty[i])
                    )

        if quote == quotes.ask:
            lowest_ask = self.order_book.get_level(quote=quotes.ask, level=0)[0]
            best_sell = max(max_price, highest_trade, lowest_ask)
            worst_sell = min(max_price, highest_trade, lowest_ask)
            ask_range = int(abs(worst_sell - best_sell))
            if ask_range == 0:
                ask_range = self.order_book.spread()
            ask_qty = self.ladder(qty=self.max_sell, n=ask_range, decay=decay)
            for i in range(ask_range):
                if ask_qty[i] != 0:
                    self.orders.append(
                        Order(self.product, int(best_sell - i), -ask_qty[i])
                    )

    def fv_arb(self, quote: quotes) -> None:
        if quote == quotes.bid:
            for bid, bid_vol in self.order_book.bids:
                bid_vol = abs(bid_vol)
                if bid >= self.max_sell_price:  # if bid > fv want to sell
                    sell_amount = min(bid_vol, self.max_sell)
                    self.orders += [Order(self.product, bid, -sell_amount)]
                    self.max_sell -= sell_amount
                else:
                    break

        if quote == quotes.ask:
            for ask, ask_vol in self.order_book.asks:
                ask_vol = abs(ask_vol)
                if ask <= self.max_buy_price:  # if ask < fv want to buy
                    buy_amount = min(ask_vol, self.max_buy)
                    self.orders += [Order(self.product, ask, buy_amount)]
                    self.max_buy -= buy_amount
                else:
                    break

    def rest_order(self, quote: quotes, price: float, quantity: int) -> None:
        if quote == quotes.bid:
            # resting bids
            final_buy_price = int(min(self.fair_value, price))
            self.orders += [Order(self.product, final_buy_price, quantity)]

        if quote == quotes.ask:
            # resting ask
            final_sell_price = int(max(self.fair_value, price))
            self.orders += [
                Order(
                    self.product,
                    final_sell_price,
                    -quantity,
                )
            ]

    def take_orders(self, quote: quotes) -> None:
        if quote == quotes.ask:
            for i in range(0, self.order_book.depth(quote=quotes.ask) - 1):
                price, ask_vol = self.order_book.get_level(quote=quotes.ask, level=i)
                ask_vol = min(abs(ask_vol), self.max_buy)
                self.orders += [Order(self.product, price, ask_vol)]
                self.max_buy -= ask_vol
        elif quote == quotes.bid:
            for i in range(0, self.order_book.depth(quote=quotes.bid) - 1):
                price, bid_vol = self.order_book.get_level(quote=quotes.bid, level=i)
                bid_vol = min(bid_vol, self.max_sell)
                self.orders += [Order(self.product, price, -bid_vol)]
                self.max_sell -= bid_vol

    def market_make_KELP(self) -> None:
        if self.copy_buy:
            self.take_orders(quote=quotes.ask)
        if self.copy_sell:
            self.take_orders(quote=quotes.bid)

        if self.max_buy > 0:
            highest_buy = self.order_book.high_volume_quotes(quote=quotes.bid)[0] + 1
            if self.order_book.depth(quote=quotes.ask) > self.order_book.depth(
                quote=quotes.bid
            ) and self.position_pct <= -self.product_params.get("threshold", 0):
                self.ladder_orders(
                    max_price=self.max_buy_price,
                    highest_trade=highest_buy,
                    quote=quotes.bid,
                    decay=self.product_params.get("decay", 0),
                )

            else:
                self.rest_order(
                    quote=quotes.bid, price=highest_buy, quantity=self.max_buy
                )

        if self.max_sell > 0 or self.copy_sell:
            highest_sell = self.order_book.high_volume_quotes(quote=quotes.ask)[0] - 1
            if self.order_book.depth(quote=quotes.ask) < self.order_book.depth(
                quote=quotes.bid
            ) and self.position_pct >= self.product_params.get("threshold", 0):
                self.ladder_orders(
                    max_price=self.max_sell_price,
                    highest_trade=highest_sell,
                    quote=quotes.ask,
                    decay=self.product_params.get("decay", 0),
                )

            else:
                self.rest_order(
                    quote=quotes.ask, price=highest_sell, quantity=self.max_sell
                )

    def market_make_RESIN(self) -> None:
        if self.copy_buy:
            self.take_orders(quote=quotes.ask)
        if self.copy_sell:
            self.take_orders(quote=quotes.bid)

        self.fv_arb(quote=quotes.ask)
        self.fv_arb(quote=quotes.bid)
        if self.max_buy > 0:
            highest_buy = max(
                [0]
                + [
                    self.order_book.get_level(quote=quotes.bid, level=i)[0]
                    for i in range(self.order_book.depth(quote=quotes.bid))
                    if self.order_book.get_level(quote=quotes.bid, level=i)[0]
                    < self.fair_value - 1
                ]
            )
            if self.position_pct >= -self.product_params.get("threshold", 0):
                if highest_buy >= self.fair_value - 2:
                    highest_buy = self.fair_value - 3

            self.rest_order(
                quote=quotes.bid,
                price=highest_buy + 1,
                quantity=self.max_buy,
            )

        if self.max_sell > 0:
            highest_sell = min(
                [1e6]
                + [
                    self.order_book.get_level(quote=quotes.ask, level=i)[0]
                    for i in range(self.order_book.depth(quote=quotes.ask))
                    if self.order_book.get_level(quote=quotes.ask, level=i)[0]
                    > self.fair_value + 1
                ]
            )
            if self.position_pct <= self.product_params.get("threshold", 0):
                if highest_sell <= self.fair_value + 2:
                    highest_sell = self.fair_value + 3

            self.rest_order(
                quote=quotes.ask,
                price=highest_sell - 1,
                quantity=self.max_sell,
            )

    def trade_SQUID(self, spread: float, preds: np.ndarray) -> None:
        if len(preds) > 2:
            self.z = self.z_score(spread, preds)

        if self.z < -self.threshold or self.copy_buy:
            self.take_orders(quote=quotes.ask)

        if self.z > self.threshold or self.copy_sell:
            self.take_orders(quote=quotes.bid)

    def trade_PICNIC1(
        self,
        delta: float,
        coin_forecast_cache_ts: deque,
        prem_disc: float,
        prem_disc_cache: deque,
    ) -> None:
        z2 = 0

        if len(prem_disc_cache) > 2:
            prem_disc_cache = np.array(prem_disc_cache)
            self.z = self.z_score(prem_disc, prem_disc_cache)
            z2 = self.z_score(delta, np.array(coin_forecast_cache_ts))

        threshold2 = self.product_params.get("z_threshold2", 0)

        if self.copy_buy:
            self.take_orders(quote=quotes.ask)
        if self.copy_sell:
            self.take_orders(quote=quotes.bid)

        if self.z < -self.threshold and z2 > threshold2 and self.max_buy > 0:
            self.take_orders(quote=quotes.ask)
        if self.z > self.threshold and z2 < -threshold2 and self.max_sell > 0:
            self.take_orders(quote=quotes.bid)

        if self.z < -self.exit_threshold and self.position < 0:
            for price, ask_vol in self.order_book.asks:
                ask_vol = min(abs(ask_vol), self.max_buy)
                self.orders += [Order(self.product, price, ask_vol)]
                self.max_buy -= ask_vol

        if self.z > self.exit_threshold and self.position > 0:
            for price, bid_vol in self.order_book.bids:
                bid_vol = min(bid_vol, self.max_sell)
                self.orders += [Order(self.product, price, -bid_vol)]
                self.max_sell -= bid_vol

    def trade_PICNIC2(
        self,
        prem_disc: float,
        prem_disc_cache: deque,
    ) -> None:
        if len(prem_disc_cache) > 2:
            prem_disc_cache = np.array(prem_disc_cache)
            self.z = self.z_score(prem_disc, prem_disc_cache)

        if self.z < -self.threshold or self.copy_buy:
            self.take_orders(quote=quotes.ask)
        if self.z > self.threshold or self.copy_sell:
            self.take_orders(quote=quotes.bid)

        if self.z < -self.exit_threshold and self.position < 0:
            for price, ask_vol in self.order_book.asks:
                ask_vol = min(abs(ask_vol), self.max_buy)
                self.orders += [Order(self.product, price, ask_vol)]
                self.max_buy -= ask_vol

        if self.z > self.exit_threshold and self.position > 0:
            for price, bid_vol in self.order_book.bids:
                bid_vol = min(bid_vol, self.max_sell)
                self.orders += [Order(self.product, price, -bid_vol)]
                self.max_sell -= bid_vol

    def trade_JAMS(self, coin_forecast_cache_ts: deque) -> None:
        if len(coin_forecast_cache_ts) > 2:
            self.z = self.z_score(self.fair_value, np.array(coin_forecast_cache_ts))

        if self.z > self.threshold or self.copy_buy:
            self.take_orders(quote=quotes.ask)

        if self.z < -self.threshold or self.copy_sell:
            self.take_orders(quote=quotes.bid)

    def trade_CROISSANTS(self, coin_forecast_cache_ts: deque) -> None:
        if len(coin_forecast_cache_ts) > 2:
            self.z = self.z_score(self.fair_value, np.array(coin_forecast_cache_ts))

        if self.z > self.threshold or self.copy_buy:
            self.take_orders(quote=quotes.ask)

        if self.z < -self.threshold or self.copy_sell:
            self.take_orders(quote=quotes.bid)

    def trade_DJEMBES(self, coin_forecast_cache_ts: deque) -> None:
        if len(coin_forecast_cache_ts) > 2:
            self.z = self.z_score(self.fair_value, np.array(coin_forecast_cache_ts))

        if self.z > self.threshold or self.copy_buy:
            self.take_orders(quote=quotes.ask)

        if self.z < -self.threshold or self.copy_sell:
            self.take_orders(quote=quotes.bid)

    def CALL_LB_ARB(
        self,
    ) -> None:
        if self.fair_value < 0 and self.max_buy > 0:
            for ask, ask_vol in self.order_book.asks:
                buy_amount = min(abs(ask_vol), self.max_buy)
                self.orders += [Order(self.product, ask, buy_amount)]
                self.max_buy -= buy_amount

    def TRADE_OPTIONS(self, base_iv_cache: deque) -> None:
        if len(base_iv_cache) > 2:
            base_iv_cache = np.array(base_iv_cache)
            self.z = self.z_score(base_iv_cache[-1], base_iv_cache[:-1])

        if self.copy_buy:
            self.take_orders(quote=quotes.ask)

        if self.copy_sell:
            self.take_orders(quote=quotes.bid)

        if self.z < -self.threshold:
            self.fv_arb(quote=quotes.ask)
        else:
            highest_buy = self.order_book.high_volume_quotes(quote=quotes.bid)[0]
            self.rest_order(quote=quotes.bid, price=highest_buy, quantity=self.max_buy)

        if self.z > self.threshold:
            self.fv_arb(quote=quotes.bid)
        else:
            highest_sell = self.order_book.high_volume_quotes(quote=quotes.ask)[0]
            self.rest_order(
                quote=quotes.ask, price=highest_sell, quantity=self.max_sell
            )

    def TRADE_VOLCANIC_ROCK(self, base_iv_cache: deque) -> None:
        to_hedge = -int(self.position - self.product_params["hedge_pos"])

        if len(base_iv_cache) > 2:
            base_iv_cache = np.array(base_iv_cache)
            self.z = self.z_score(base_iv_cache[-1], base_iv_cache[:-1])

        if self.copy_buy or (to_hedge > 0 and self.z > self.threshold):
            for price, ask_vol in self.order_book.asks:
                ask_vol = min(abs(ask_vol), self.max_buy, abs(to_hedge))
                self.orders += [Order(self.product, price, ask_vol)]
                self.max_buy -= ask_vol
                to_hedge -= ask_vol
        else:
            highest_buy = self.order_book.high_volume_quotes(quote=quotes.bid)[0]
            self.rest_order(quote=quotes.bid, price=highest_buy, quantity=self.max_buy)

        if self.copy_sell or (to_hedge < 0 and self.z > self.threshold):
            for price, bid_vol in self.order_book.bids:
                bid_vol = min(bid_vol, self.max_sell, abs(to_hedge))
                self.orders += [Order(self.product, price, -bid_vol)]
                self.max_sell -= bid_vol
                to_hedge += bid_vol
        else:
            highest_sell = self.order_book.high_volume_quotes(quote=quotes.ask)[0]
            self.rest_order(
                quote=quotes.ask, price=highest_sell, quantity=self.max_sell
            )

    def TRADE_MACARONS(
        self,
        conv_sunlight: float,
        forecast: float,
        pristine_forecast_bid: float,
        pristine_forecast_ask: float,
        pristine_buy: float,
        pristine_sell: float,
        coin_forecast_cache_ts: deque,
    ) -> int:
        conversion = 0

        if len(coin_forecast_cache_ts) > 2:
            self.z = self.z_score(self.fair_value, np.array(coin_forecast_cache_ts))

        max_conversion = min(
            abs(self.position), position_limit.MAGNIFICENT_MACARONS_CONVERSION
        )

        if self.z < -self.threshold or self.copy_buy or conv_sunlight < 50:
            for bid, bid_vol in self.order_book.bids:
                if bid >= forecast - self.order_book.spread() / 2:
                    if (
                        conversion != -max_conversion
                        and bid < pristine_sell
                        and self.position > 0
                    ):
                        conversion = -max_conversion
                        self.max_sell += conversion
                    else:
                        sell_amount = min(abs(bid_vol), self.max_sell)
                        self.orders += [Order(self.product, bid, -sell_amount)]
                        self.max_sell -= sell_amount

        if self.z > self.threshold or self.copy_sell:
            for ask, ask_vol in self.order_book.asks:
                if ask <= forecast + self.order_book.spread() / 2:
                    if (
                        conversion != max_conversion
                        and ask > pristine_buy
                        and self.position < 0
                    ):
                        conversion = max_conversion
                        self.max_buy -= conversion
                    else:
                        buy_amount = min(abs(ask_vol), self.max_buy)
                        self.orders += [Order(self.product, ask, buy_amount)]
                        self.max_buy -= buy_amount

        if self.max_buy > 0:
            highest_buy = int(pristine_forecast_bid)
            self.orders += [Order(self.product, highest_buy, self.max_buy)]

        if self.max_sell > 0:
            highest_sell = int(pristine_forecast_ask)
            self.orders += [Order(self.product, highest_sell, -self.max_sell)]

        return conversion


class Trader:
    def __init__(self, params: dict = None):
        if params is not None:
            self.params = params
        else:
            self.params = {
                tradable_product.RAINFOREST_RESIN: {
                    "ask_slip": 1,
                    "bid_slip": -1,
                    "decay": 0.1,
                    "threshold": 0.0,
                    "trade_with": {
                        traders.Pablo: [4, 5],
                        traders.Caesar: [1],
                        traders.Charlie: [7],
                        traders.Penelope: [8],
                    },
                    "trade_against": {
                        traders.Paris: [1],
                        traders.Charlie: [5, 8],
                        traders.Penelope: [1, 6, 7],
                    },
                },
                tradable_product.KELP: {
                    "alpha": np.array([[-1.93181848e-04], [0.0]]),
                    "beta": np.array([[1.0, -0.9341909]]),
                    "gamma": np.array(
                        [
                            [-2.37870411e-02, -3.37620662e-01],
                            [0.0, -4.70692887e-01],
                        ]
                    ),
                    "decay": 0.6,
                    "threshold": 0.8,
                    "trade_with": {
                        traders.Gina: [i for i in range(11)],
                        traders.Paris: [1],
                        traders.Camilla: [1],
                        traders.Pablo: [i for i in range(6)],
                        traders.Caesar: [i for i in range(2, 7)],
                        traders.Penelope: [2, 3, 4, 5, 14],
                    },
                    "trade_against": {
                        traders.Paris: [2],
                        traders.Caesar: [1],
                        traders.Charlie: [i for i in range(10)],
                        traders.Penelope: [6, 10, 11],
                    },
                },
                tradable_product.SQUID_INK: {
                    "alpha": np.array([[-1.93181848e-04], [0.0]]),
                    "beta": np.array([[1.0, -0.9341909]]),
                    "gamma": np.array(
                        [
                            [-2.37870411e-02, -3.37620662e-01],
                            [0.0, -4.70692887e-01],
                        ]
                    ),
                    "z_threshold": 0,
                    "trade_with": {
                        traders.Gina: [i for i in range(11)],
                        traders.Paris: [1],
                        traders.Camilla: [1, 2, 6],
                        traders.Olivia: [
                            1,
                            14,
                            15,
                        ],
                        traders.Pablo: [i for i in range(6)],
                        traders.Caesar: [2, 3, 4, 5, 6],
                        traders.Charlie: [
                            13,
                            11,
                        ],
                        traders.Gary: [1],
                        traders.Penelope: [1, 2, 3, 4, 12, 14],
                    },
                    "trade_against": {
                        traders.Paris: [2],
                        traders.Caesar: [1],
                        traders.Charlie: [
                            1,
                            7,
                            2,
                            5,
                            9,
                            4,
                            6,
                            3,
                            10,
                            14,
                        ],
                        traders.Gary: [12, 13],
                        traders.Penelope: [6, 10, 11, 15],
                    },
                },
                tradable_product.PICNIC_BASKET1: {
                    tradable_product.CROISSANTS: 6,
                    tradable_product.JAMS: 3,
                    tradable_product.DJEMBES: 1,
                    "alpha": np.array([[-5.61674869e-05], [0.0]]),
                    "beta": np.array([[1.0, -0.07318334]]),
                    "gamma": np.array(
                        [[-0.11347602, 0.00577044], [0.39925558, -0.03051414]]
                    ),
                    "z_threshold": 1.96,
                    "z_threshold2": 1.64,
                    "exit_threshold": 2.33,
                    "trade_with": {
                        traders.Pablo: [
                            1,
                            7,
                            2,
                            5,
                            6,
                            3,
                            8,
                        ],
                        traders.Penelope: [8],
                    },
                    "trade_against": {
                        traders.Camilla: [1, 2, 3, 5, 6, 7, 8],
                        traders.Caesar: [1, 2, 5, 7, 8, 10],
                    },
                },
                tradable_product.PICNIC_BASKET2: {
                    tradable_product.CROISSANTS: 4,
                    tradable_product.JAMS: 2,
                    tradable_product.DJEMBES: 0,
                    "z_threshold": 1.96,
                    "exit_threshold": 2.33,
                    "trade_with": {
                        traders.Pablo: [i for i in range(8)],
                        traders.Caesar: [1],
                        traders.Charlie: [10],
                    },
                    "trade_against": {
                        traders.Camilla: [
                            1,
                            7,
                            8,
                            2,
                            6,
                            3,
                        ],
                        traders.Caesar: [5, 8],
                        traders.Charlie: [i for i in range(8)],
                    },
                },
                tradable_product.CROISSANTS: {
                    "alpha": np.array([[-5.61674869e-05], [0.0]]),
                    "beta": np.array([[1.0, -0.07318334]]),
                    "gamma": np.array(
                        [[-0.11347602, 0.00577044], [0.39925558, -0.03051414]]
                    ),
                    "z_threshold": 6,
                    "trade_with": {
                        traders.Paris: [7, 8],
                        traders.Camilla: [i for i in range(11)],
                        traders.Olivia: [3],
                        traders.Caesar: [3],
                    },
                    "trade_against": {
                        traders.Paris: [5],
                        traders.Caesar: [i for i in range(4, 11)],
                    },
                },
                tradable_product.JAMS: {
                    "alpha": np.array([[-2.56244811e-04], [0.0]]),
                    "beta": np.array([[1.0, -0.11116249]]),
                    "gamma": np.array([[0.0, 0.00482966], [0.26875689, -0.03614998]]),
                    "z_threshold": 3.8,
                    "trade_with": {traders.Camilla: [i for i in range(12)]},
                    "trade_against": {traders.Caesar: [i for i in range(12)]},
                },
                tradable_product.DJEMBES: {
                    "alpha": np.array([[-9.90008420e-05], [1.71206399e-03]]),
                    "beta": np.array([[1.0, -0.22870816]]),
                    "gamma": np.array([[0.03207502, 0.0], [0.11032108, -0.02182072]]),
                    "z_threshold": 1,
                    "trade_with": {
                        traders.Paris: [5],
                        traders.Camilla: [i for i in range(3, 12)],
                        traders.Caesar: [2],
                    },
                    "trade_against": {
                        traders.Paris: [
                            2,
                        ],
                        traders.Caesar: [i for i in range(3, 11)],
                    },
                },
                tradable_product.VOLCANIC_ROCK: {
                    "hedge_pos": 0,
                    "z_threshold": 0.55,
                    "trade_with": {
                        traders.Pablo: [
                            1,
                        ]
                    },
                    "trade_against": {
                        traders.Caesar: [
                            1,
                        ]
                        + [i for i in range(5, 11)]
                    },
                },
                tradable_product.VOLCANIC_ROCK_VOUCHER_9500: {
                    "strike": 9500,
                    "dte": 3e6,
                    "z_threshold": 0.55,
                    "trade_with": {
                        traders.Pablo: [i for i in range(21)],
                        traders.Caesar: [
                            3,
                            5,
                        ]
                        + [i for i in range(10, 21)],
                        traders.Penelope: [1, 2, 11, 15, 16],
                    },
                    "trade_against": {
                        traders.Camilla: [i for i in range(21)],
                        traders.Caesar: [1, 2, 6, 7],
                        traders.Penelope: [
                            12,
                        ],
                    },
                },
                tradable_product.VOLCANIC_ROCK_VOUCHER_9750: {
                    "strike": 9750,
                    "dte": 3e6,
                    "z_threshold": 0.55,
                    "trade_with": {
                        traders.Pablo: [i for i in range(21)],
                        traders.Caesar: [
                            6,
                        ]
                        + [i for i in range(10, 21)],
                        traders.Penelope: [1, 2, 6, 7, 9, 8, 11, 15],
                    },
                    "trade_against": {
                        traders.Camilla: [i for i in range(21)],
                        traders.Caesar: [
                            1,
                            7,
                            2,
                            4,
                            6,
                        ],
                        traders.Penelope: [1, 4, 10, 12, 19],
                    },
                },
                tradable_product.VOLCANIC_ROCK_VOUCHER_10000: {
                    "strike": 10_000,
                    "dte": 3e6,
                    "z_threshold": 0.55,
                    "trade_with": {
                        traders.Camilla: [
                            3,
                        ],
                        traders.Pablo: [i for i in range(20)],
                        traders.Caesar: [5, 6] + [i for i in range(19, 21)],
                        traders.Penelope: [1, 5, 6, 7, 15, 11],
                    },
                    "trade_against": {
                        traders.Camilla: [2] + [i for i in range(5, 21)],
                        traders.Caesar: [
                            1,
                            7,
                            2,
                        ],
                        traders.Penelope: [12],
                    },
                },
                tradable_product.VOLCANIC_ROCK_VOUCHER_10250: {
                    "strike": 10_250,
                    "dte": 3e6,
                    "z_threshold": 0.55,
                    "trade_with": {
                        traders.Pablo: [i for i in range(21)],
                        traders.Caesar: [i for i in range(10, 21)],
                        traders.Penelope: [
                            3,
                            6,
                        ],
                    },
                    "trade_against": {
                        traders.Camilla: [
                            2,
                        ]
                        + [i for i in range(5, 21)],
                        traders.Caesar: [
                            1,
                            7,
                            2,
                            5,
                            6,
                            3,
                        ],
                    },
                },
                tradable_product.VOLCANIC_ROCK_VOUCHER_10500: {
                    "strike": 10_500,
                    "dte": 3e6,
                    "z_threshold": 0.55,
                    "trade_with": {
                        traders.Pablo: [i for i in range(21)],
                        traders.Caesar: [i for i in range(10, 21)],
                    },
                    "trade_against": {
                        traders.Camilla: [i for i in range(5, 21)],
                        traders.Caesar: [
                            1,
                            2,
                            5,
                            7,
                        ],
                    },
                },
                tradable_product.MAGNIFICENT_MACARONS: {
                    "alpha": np.array([[-0.11809086], [0.02087891], [-0.00018677]]),
                    "beta": np.array([[1.0, -1.00619334, -0.04519032]]),
                    "gamma": np.array(
                        [
                            [-4.43562692e-01, 4.41261197e-01, -5.54291956e00],
                            [-1.28555496e-02, 9.24099084e-03, -3.89429168e00],
                            [1.43503499e-04, -1.58118913e-04, 6.27232006e-01],
                        ],
                    ),
                    "z_threshold": 1.96,
                    "trade_with": {
                        traders.Paris: [i for i in range(1, 5)],
                        traders.Charlie: [i for i in range(4, 8)],
                    },
                    "trade_against": {
                        traders.Paris: [i for i in range(6, 8)],
                        traders.Camilla: [i for i in range(2, 8)],
                        traders.Caesar: [
                            2,
                            5,
                            4,
                            6,
                        ],
                        traders.Charlie: [1, 2],
                    },
                },
            }

    options = [
        tradable_product.VOLCANIC_ROCK_VOUCHER_9500,
        tradable_product.VOLCANIC_ROCK_VOUCHER_9750,
        tradable_product.VOLCANIC_ROCK_VOUCHER_10000,
        tradable_product.VOLCANIC_ROCK_VOUCHER_10250,
        tradable_product.VOLCANIC_ROCK_VOUCHER_10500,
    ]

    mid_cache = copy.deepcopy(empty_dict_cache)
    forecast_cache = copy.deepcopy(empty_dict_cache)

    coin_forecast_cache_ts = copy.deepcopy(empty_dict_cache)
    prem_disc_cache = copy.deepcopy(empty_dict_cache)
    pred_prem_disc_cache = copy.deepcopy(empty_dict_cache)

    iv_cache = copy.deepcopy(empty_dict_cache)
    mt_cache = copy.deepcopy(empty_dict_cache)
    smile_coeff = None
    base_iv_cache = deque(maxlen=1000)

    sunlight_cache = deque(maxlen=50)
    sugar_cache = deque(maxlen=50)
    pristine_mid_cache = deque(maxlen=50)

    def process_traders(
        self, trades: list[Trade], timestamp: list[int]
    ) -> tuple[dict[str, int], dict[str, int]]:
        buyer, seller = {}, {}

        if trades:
            for i in trades:
                if i.timestamp in timestamp:
                    buyer[i.buyer] = i.quantity
                    seller[i.seller] = i.quantity
        return buyer, seller

    def winsorize_percentile(
        self, data: np.ndarray, lower_pct: int = 5, upper_pct: int = 95
    ) -> np.ndarray:
        lower = np.percentile(data, lower_pct)
        upper = np.percentile(data, upper_pct)
        return np.clip(data, lower, upper)

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        conversions = 0
        trader_data = ""
        timestamp = state.timestamp
        logger.print(timestamp)

        result = {}  # Orders to be placed on exchange matching engine

        basket_order_depth = {
            tradable_product.CROISSANTS: orderbook(
                (state.order_depths[tradable_product.CROISSANTS])
            ),
            tradable_product.JAMS: orderbook(
                (state.order_depths[tradable_product.JAMS])
            ),
            tradable_product.DJEMBES: orderbook(
                (state.order_depths[tradable_product.DJEMBES])
            ),
        }

        s, r, t = (
            orderbook(
                order_depth=state.order_depths[tradable_product.VOLCANIC_ROCK]
            ).mid_price(),
            0.0,
            self.params[tradable_product.VOLCANIC_ROCK_VOUCHER_10000]["dte"]
            / (365 * 1e6),
        )

        x = np.array([])
        y = np.array([])
        for option in self.options:
            k = self.params[option]["strike"]
            self.mt_cache[option].append(np.log(s / k) / t)

            ob = orderbook(order_depth=state.order_depths[option])

            option_ = options(
                S=s,
                K=k,
                r=r,
                sigma=0,
                T=t,
            )

            sigma = option_.solve_iv(
                call_price=ob.mid_price(),
            )

            if sigma > 1 or sigma < 0:
                if len(self.iv_cache[option]) == 0:
                    sigma = 0
                else:
                    sigma = self.iv_cache[option][-1]
            self.iv_cache[option].append(sigma)

            x = np.concatenate((x, np.array(self.mt_cache[option])))
            y = np.concatenate((y, np.array(self.iv_cache[option])))

        y = self.winsorize_percentile(y)

        try:
            self.smile_coeff = np.polyfit(x=x, y=y, deg=2)
            self.base_iv_cache.append(self.smile_coeff[-1])
        except:
            self.smile_coeff = None
            self.base_iv_cache.append(0)

        for product in state.order_depths:
            curr_pos = state.position.get(product, 0)

            # get order & trade data
            order_depth: OrderDepth = state.order_depths[product]
            order_book = orderbook(order_depth=order_depth)

            trades = state.market_trades.get(product, None)
            if product in [
                tradable_product.DJEMBES,
                tradable_product.CROISSANTS,
                tradable_product.SQUID_INK,
            ]:
                buyer, seller = self.process_traders(
                    trades=trades, timestamp=[timestamp, timestamp - 100]
                )
            else:
                buyer, seller = self.process_traders(
                    trades=trades, timestamp=[timestamp]
                )

            product_params = self.params[product]

            product_fv = fair_value(
                product=product,
                product_params=product_params,
                order_book=order_book,
            )

            if product == tradable_product.RAINFOREST_RESIN:
                resin_FV = product_fv.RAINFOREST_RESIN_FV()

                resin_strat = trading_strategy(
                    product=product,
                    product_params=product_params,
                    position=curr_pos,
                    position_limit=position_limit.RAINFOREST_RESIN,
                    order_book=order_book,
                    fair_value=resin_FV,
                    buyer=buyer,
                    seller=seller,
                )

                resin_strat.market_make_RESIN()

                result[product] = resin_strat.orders

            if product == tradable_product.KELP:
                delta, forecast = product_fv.VECM_FORECAST(
                    prod_1=tradable_product.KELP,
                    prod_1_mid=orderbook(
                        order_depth=state.order_depths[tradable_product.KELP]
                    ).mid_price(),
                    prod_2=tradable_product.SQUID_INK,
                    prod_2_mid=orderbook(
                        order_depth=state.order_depths[tradable_product.SQUID_INK]
                    ).mid_price(),
                    mid_cache=self.mid_cache,
                )

                kelp_strat = trading_strategy(
                    product=product,
                    product_params=product_params,
                    position=curr_pos,
                    position_limit=position_limit.KELP,
                    order_book=order_book,
                    fair_value=forecast[1],
                    buyer=buyer,
                    seller=seller,
                )

                kelp_strat.fv_arb(quote=quotes.ask)
                kelp_strat.fv_arb(quote=quotes.bid)

                kelp_strat.market_make_KELP()
                result[product] = kelp_strat.orders

            if product == tradable_product.SQUID_INK:
                spread, pred = product_fv.SQUID_FV(
                    curr_mid_kelp=orderbook(
                        state.order_depths[tradable_product.KELP]
                    ).mid_price(),
                    mid_cache_kelp=self.mid_cache[tradable_product.KELP],
                    mid_cache_squid=self.mid_cache[tradable_product.SQUID_INK],
                )
                squid_strat = trading_strategy(
                    product=product,
                    product_params=product_params,
                    position=curr_pos,
                    position_limit=position_limit.SQUID_INK,
                    order_book=order_book,
                    fair_value=0,
                    buyer=buyer,
                    seller=seller,
                )

                squid_strat.trade_SQUID(spread=spread, preds=pred)

                result[product] = squid_strat.orders

            if product == tradable_product.PICNIC_BASKET1:
                basket_price = product_fv.PICNIC1_FV(basket_order_depth)

                delta, forecast = product_fv.VECM_FORECAST(
                    prod_1=tradable_product.CROISSANTS,
                    prod_1_mid=orderbook(
                        order_depth=state.order_depths[tradable_product.CROISSANTS]
                    ).mid_price(),
                    prod_2=tradable_product.PICNIC_BASKET1,
                    prod_2_mid=orderbook(
                        order_depth=state.order_depths[tradable_product.PICNIC_BASKET1]
                    ).mid_price(),
                    mid_cache=self.mid_cache,
                )

                prem_disc = order_book.mid_price() - basket_price

                picnic1_strat = trading_strategy(
                    product=product,
                    product_params=product_params,
                    position=curr_pos,
                    position_limit=position_limit.PICNIC_BASKET1,
                    order_book=order_book,
                    fair_value=basket_price,
                    buyer=buyer,
                    seller=seller,
                )

                picnic1_strat.trade_PICNIC1(
                    delta=delta[1],
                    coin_forecast_cache_ts=self.coin_forecast_cache_ts[product],
                    prem_disc=prem_disc,
                    prem_disc_cache=self.prem_disc_cache[product],
                )

                result[product] = picnic1_strat.orders

                self.prem_disc_cache[product].append(prem_disc)
                self.coin_forecast_cache_ts[product].append(delta[1])

            if product == tradable_product.PICNIC_BASKET2:
                basket_price = product_fv.PICNIC2_FV(basket_order_depth)

                prem_disc = order_book.mid_price() - basket_price

                picnic2_strat = trading_strategy(
                    product=product,
                    product_params=product_params,
                    position=curr_pos,
                    position_limit=position_limit.PICNIC_BASKET2,
                    order_book=order_book,
                    fair_value=basket_price,
                    buyer=buyer,
                    seller=seller,
                )

                picnic2_strat.trade_PICNIC2(
                    prem_disc=prem_disc,
                    prem_disc_cache=self.prem_disc_cache[product],
                )

                result[product] = picnic2_strat.orders

                self.prem_disc_cache[product].append(prem_disc)

            if product == tradable_product.CROISSANTS:
                delta, forecast = product_fv.VECM_FORECAST(
                    prod_1=tradable_product.CROISSANTS,
                    prod_1_mid=orderbook(
                        order_depth=state.order_depths[tradable_product.CROISSANTS]
                    ).mid_price(),
                    prod_2=tradable_product.PICNIC_BASKET1,
                    prod_2_mid=orderbook(
                        order_depth=state.order_depths[tradable_product.PICNIC_BASKET1]
                    ).mid_price(),
                    mid_cache=self.mid_cache,
                )

                croissants_strat = trading_strategy(
                    product=product,
                    product_params=product_params,
                    position=curr_pos,
                    position_limit=position_limit.CROISSANTS,
                    order_book=order_book,
                    fair_value=delta[0],
                    buyer=buyer,
                    seller=seller,
                )

                croissants_strat.trade_CROISSANTS(
                    coin_forecast_cache_ts=self.coin_forecast_cache_ts[product]
                )

                result[product] = croissants_strat.orders

                self.coin_forecast_cache_ts[product].append(delta[0])

            if product == tradable_product.JAMS:
                delta, forecast = product_fv.VECM_FORECAST(
                    prod_1=tradable_product.JAMS,
                    prod_1_mid=orderbook(
                        order_depth=state.order_depths[tradable_product.JAMS]
                    ).mid_price(),
                    prod_2=tradable_product.PICNIC_BASKET1,
                    prod_2_mid=orderbook(
                        order_depth=state.order_depths[tradable_product.PICNIC_BASKET1]
                    ).mid_price(),
                    mid_cache=self.mid_cache,
                )

                jams_strat = trading_strategy(
                    product=product,
                    product_params=product_params,
                    position=curr_pos,
                    position_limit=position_limit.JAMS,
                    order_book=order_book,
                    fair_value=delta[0],
                    buyer=buyer,
                    seller=seller,
                )
                jams_strat.trade_JAMS(
                    coin_forecast_cache_ts=self.coin_forecast_cache_ts[product]
                )

                result[product] = jams_strat.orders
                self.coin_forecast_cache_ts[product].append(delta[0])

            if product == tradable_product.DJEMBES:
                delta, forecast = product_fv.VECM_FORECAST(
                    prod_1=tradable_product.DJEMBES,
                    prod_1_mid=orderbook(
                        order_depth=state.order_depths[tradable_product.DJEMBES]
                    ).mid_price(),
                    prod_2=tradable_product.PICNIC_BASKET1,
                    prod_2_mid=orderbook(
                        order_depth=state.order_depths[tradable_product.PICNIC_BASKET1]
                    ).mid_price(),
                    mid_cache=self.mid_cache,
                )

                djembes_strat = trading_strategy(
                    product=product,
                    product_params=product_params,
                    position=curr_pos,
                    position_limit=position_limit.DJEMBES,
                    order_book=order_book,
                    fair_value=delta[0],
                    buyer=buyer,
                    seller=seller,
                )
                djembes_strat.trade_DJEMBES(
                    coin_forecast_cache_ts=self.coin_forecast_cache_ts[product]
                )

                result[product] = djembes_strat.orders
                self.coin_forecast_cache_ts[product].append(delta[0])

            if product in self.options:
                sigma = self.iv_cache[product][-1]

                option_strat = trading_strategy(
                    product=product,
                    product_params=product_params,
                    position=curr_pos,
                    position_limit=position_limit.VOLCANIC_ROCK_VOUCHER_9500,
                    order_book=order_book,
                    fair_value=0,
                    buyer=buyer,
                    seller=seller,
                )

                # MISPRICING
                call_price, delta = product_fv.THEO_PRICE(
                    S=s,
                    K=k,
                    r=r,
                    TTE=t,
                    mt_cache=self.mt_cache[product],
                    coeffs=self.smile_coeff,
                )

                option_strat.update_FV(FV=call_price)
                option_strat.TRADE_OPTIONS(base_iv_cache=self.base_iv_cache)

                # CALL LB ARB
                call_lb = product_fv.CALL_LB(
                    payoff=max(s - k, 0), call_price=order_book.mid_price()
                )
                option_strat.update_FV(FV=call_lb)
                option_strat.CALL_LB_ARB()

                self.params[tradable_product.VOLCANIC_ROCK]["hedge_pos"] += (
                    curr_pos * -delta
                )

                all_orders = {}
                for order in option_strat.orders:
                    if order.price not in all_orders.keys():
                        all_orders[order.price] = 0
                    all_orders[order.price] += order.quantity
                    self.params[tradable_product.VOLCANIC_ROCK]["hedge_pos"] += (
                        order.quantity * -delta
                    )

                agg_orders = []
                for price, qty in all_orders.items():
                    agg_orders += [Order(product, price, qty)]

                result[product] = agg_orders
                product_params["dte"] -= 100

            if product == tradable_product.VOLCANIC_ROCK:
                rock_strat = trading_strategy(
                    product=product,
                    product_params=product_params,
                    position=curr_pos,
                    position_limit=position_limit.VOLCANIC_ROCK,
                    order_book=order_book,
                    fair_value=0,
                    buyer=buyer,
                    seller=seller,
                )

                rock_strat.TRADE_VOLCANIC_ROCK(base_iv_cache=self.base_iv_cache)

                self.params[tradable_product.VOLCANIC_ROCK]["hedge_pos"] = 0

                result[product] = rock_strat.orders

            if product == tradable_product.MAGNIFICENT_MACARONS:
                obs = state.observations.conversionObservations[product]
                conv_bid = obs.bidPrice
                conv_ask = obs.askPrice
                conv_import = obs.importTariff
                conv_export = obs.exportTariff
                conv_transport = obs.transportFees
                conv_sugar = obs.sugarPrice
                conv_sunlight = obs.sunlightIndex

                pristine_buy = conv_ask + conv_import + conv_transport + 0.1
                pristine_sell = conv_bid - conv_transport - conv_export - 0.1

                pristine_mid = (pristine_buy + pristine_sell) / 2

                delta, forecast, pristine_forecast = product_fv.VECM_MACAROONS(
                    mid=order_book.mid_price(),
                    pristine_mid=pristine_mid,
                    sunlight=conv_sunlight,
                    mid_cache=self.mid_cache,
                    pristine_mid_cache=self.pristine_mid_cache,
                    sunlight_cache=self.sunlight_cache,
                )

                pristine_forecast_bid = (
                    pristine_forecast - (pristine_buy - pristine_sell) / 2
                )
                pristine_forecast_ask = (
                    pristine_forecast + (pristine_buy - pristine_sell) / 2
                )

                macarons_strat = trading_strategy(
                    product=product,
                    product_params=product_params,
                    position=curr_pos,
                    position_limit=position_limit.MAGNIFICENT_MACARONS,
                    order_book=order_book,
                    fair_value=delta,
                    buyer=buyer,
                    seller=seller,
                )

                conversions = macarons_strat.TRADE_MACARONS(
                    conv_sunlight=conv_sunlight,
                    forecast=forecast,
                    pristine_forecast_bid=pristine_forecast_bid,
                    pristine_forecast_ask=pristine_forecast_ask,
                    pristine_buy=pristine_buy,
                    pristine_sell=pristine_sell,
                    coin_forecast_cache_ts=self.coin_forecast_cache_ts[product],
                )

                result[product] = macarons_strat.orders

                self.sunlight_cache.append(conv_sunlight)
                self.sugar_cache.append(conv_sugar)
                self.pristine_mid_cache.append(pristine_mid)
                self.coin_forecast_cache_ts[product].append(delta)

        for product in state.order_depths:
            self.mid_cache[product].append(
                orderbook(order_depth=state.order_depths[product]).mid_price()
            )

        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data
