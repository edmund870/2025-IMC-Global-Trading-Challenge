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
from typing import List
import copy
import numpy as np
import math
import json

from typing import Any


empty_dict = {
    "KELP": 0,
    "RAINFOREST_RESIN": 0,
    "SQUID_INK": 0,
    "CROISSANTS": 0,
    "JAMS": 0,
    "PICNIC_BASKET1": 0,
    "PICNIC_BASKET2": 0,
    "DJEMBES": 0,
    "VOLCANIC_ROCK": 0,
    "VOLCANIC_ROCK_VOUCHER_9500": 0,
    "VOLCANIC_ROCK_VOUCHER_9750": 0,
    "VOLCANIC_ROCK_VOUCHER_10000": 0,
    "VOLCANIC_ROCK_VOUCHER_10250": 0,
    "VOLCANIC_ROCK_VOUCHER_10500": 0,
}

empty_dict_cache = {
    "KELP": deque(maxlen=101),
    "RAINFOREST_RESIN": deque(maxlen=200),
    "SQUID_INK": deque(maxlen=100),
    "CROISSANTS": deque(maxlen=1_000),
    "JAMS": deque(maxlen=1_000),
    "PICNIC_BASKET1": deque(maxlen=1_000),
    "PICNIC_BASKET2": deque(maxlen=1_000),
    "DJEMBES": deque(maxlen=1_000),
    "VOLCANIC_ROCK": deque(maxlen=1_000),
    "VOLCANIC_ROCK_VOUCHER_9500": deque(maxlen=1_000),
    "VOLCANIC_ROCK_VOUCHER_9750": deque(maxlen=1_000),
    "VOLCANIC_ROCK_VOUCHER_10000": deque(maxlen=1_000),
    "VOLCANIC_ROCK_VOUCHER_10250": deque(maxlen=1_000),
    "VOLCANIC_ROCK_VOUCHER_10500": deque(maxlen=1_000),
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


class trade_direction:
    buy = "buy"
    sell = "sell"


class quotes:
    bid = "bid"
    ask = "ask"


class orderbook:
    def __init__(self, order_depth: OrderDepth):
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
            return max(self.bids, key=lambda tup: tup[1])
        if quote == quotes.ask:
            return min(self.asks, key=lambda tup: tup[1])

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
    ):
        self.product = product
        self.product_params = product_params
        self.order_book = order_book

    def RAINFOREST_RESIN_FV(
        self,
    ) -> int:
        base = 10_000
        return base

    def SQUID_FV(self, curr_mid_kelp, mid_cache_kelp, mid_cache_squid):
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

    def PICNIC1_FV(self, basket_order_depth):
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

    def PICNIC2_FV(self, basket_order_depth):
        cross = (
            basket_order_depth[tradable_product.CROISSANTS].mid_price()
            * self.product_params[tradable_product.CROISSANTS]
        )
        jams = (
            basket_order_depth[tradable_product.JAMS].mid_price()
            * self.product_params[tradable_product.JAMS]
        )

        return cross + jams

    def VECM(self, current_price, prev_mid):
        return self.product_params["alpha"] @ (
            self.product_params["beta"] @ current_price
        ) + self.product_params["gamma"] @ (current_price - prev_mid)

    def VECM_FORECAST(self, prod_1, prod_1_mid, prod_2, prod_2_mid, mid_cache):
        most_mid_price = np.array([prod_1_mid, prod_2_mid])

        if len(mid_cache[self.product]) < 1:
            prev_mid = most_mid_price
        else:
            prev_mid = np.array(
                [
                    mid_cache[prod_1][-1],
                    mid_cache[prod_2][-1],
                ]
            )

        delta = self.VECM(current_price=most_mid_price, prev_mid=prev_mid)

        prod_1_forecast = prod_1_mid + delta[0]
        prod_2_forecast = prod_2_mid + delta[1]

        forecast = np.array([prod_1_forecast, prod_2_forecast])

        return delta, forecast

    def phi(self, x):
        return (1.0 + math.erf(x / np.sqrt(2.0))) / 2.0

    def black_scholes_call(self, S, K, r, sigma, T):
        d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return S * self.phi(d1) - K * np.exp(-r * T) * self.phi(d2)

    def call_delta(self, S, K, r, sigma, T):
        d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
        return self.phi(d1)

    def vega(self, S, K, r, sigma, T):
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        return S * np.exp(-0.5 * d1**2) / np.sqrt(2 * np.pi) * np.sqrt(T)

    def solve_iv(
        self, call_price, S, K, r, T, initial_guess=0.2, max_iter=1000, tol=1e-6
    ):
        solution = initial_guess
        for _ in range(max_iter):
            f_val = self.black_scholes_call(S, K, r, solution, T) - call_price
            f_prime_val = self.vega(S, K, r, solution, T)
            if abs(f_prime_val) < tol:
                break
            solution = solution - f_val / f_prime_val
            if abs(f_val) < tol:
                break
        return solution

    def CALL_LB(self, payoff, call_price):
        return call_price - payoff  # if call price < payoff arb

    def BUTTERFLY(
        self,
        lower,
        mid,
        upper,
    ):
        spread = -mid + 0.5 * (lower + upper)
        return spread  # cannot be < 0

    def BULL_CALL_SPREAD(self, lower, upper, lower_strike, upper_strike):
        premium = lower - upper
        strike_diff = upper_strike - lower_strike

        return strike_diff - premium  # < 0 = arb


class trading_strategy:
    def __init__(
        self,
        product: tradable_product,
        product_params: dict,
        position: int,
        position_limit: int,
        order_book: orderbook,
        fair_value: int | float,
    ):
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

    def update_FV(self, FV):
        self.fair_value = FV
        self.max_buy_price = FV + self.product_params.get("ask_slip", 0)
        self.max_sell_price = FV + self.product_params.get("bid_slip", 0)

    def z_score(self, value, sample):
        return (value - sample.mean()) / (sample.std() + 1e-16)

    def hurst(self, ts):
        lags = range(2, 100)

        tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]

        poly = np.polyfit(np.log(lags), np.log(tau), 1)

        return poly[0] * 2.0

    def ladder(self, qty, n, decay) -> list[int]:
        raw_values = [math.exp(-decay * i) for i in range(1, n + 1)]
        sum_raw = sum(raw_values)
        ladder = [math.floor(qty * value / sum_raw) for value in raw_values]
        return ladder

    def ladder_orders(self, max_price, highest_trade, quote, decay) -> list[Order]:
        orders = []
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
                    orders.append(Order(self.product, int(best_buy + i), bid_qty[i]))

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
                    orders.append(Order(self.product, int(best_sell - i), -ask_qty[i]))

        return orders

    def fv_arb(self, quote) -> list[Order]:
        orders = []
        if quote == quotes.bid:
            for bid, bid_vol in self.order_book.bids:
                bid_vol = abs(bid_vol)
                if bid >= self.max_sell_price:  # if bid > fv want to sell
                    sell_amount = min(bid_vol, self.max_sell)
                    orders.append(Order(self.product, bid, -sell_amount))
                    self.max_sell -= sell_amount
                else:
                    break

        if quote == quotes.ask:
            for ask, ask_vol in self.order_book.asks:
                ask_vol = abs(ask_vol)
                if ask <= self.max_buy_price:  # if ask < fv want to buy
                    buy_amount = min(ask_vol, self.max_buy)
                    orders.append(Order(self.product, ask, buy_amount))
                    self.max_buy -= buy_amount
                else:
                    break

        return orders

    def rest_order(self, quote, price, quantity) -> list[Order]:
        orders = []
        if quote == quotes.bid:
            # resting bids
            final_buy_price = int(min(self.fair_value, price))
            orders.append(Order(self.product, final_buy_price, quantity))

        if quote == quotes.ask:
            # resting ask
            final_sell_price = int(max(self.fair_value, price))
            orders.append(
                Order(
                    self.product,
                    final_sell_price,
                    -quantity,
                )
            )
        return orders

    def market_make_KELP(self) -> list[Order]:
        orders = []

        if self.max_buy > 0:
            highest_buy = self.order_book.high_volume_quotes(quote=quotes.bid)[0] + 1
            if self.order_book.depth(quote=quotes.ask) > self.order_book.depth(
                quote=quotes.bid
            ) and self.position_pct <= -self.product_params.get("threshold", 0):
                orders += self.ladder_orders(
                    max_price=self.max_buy_price,
                    highest_trade=highest_buy,
                    quote=quotes.bid,
                    decay=self.product_params.get("decay", 0),
                )

            else:
                orders += self.rest_order(
                    quote=quotes.bid, price=highest_buy, quantity=self.max_buy
                )

        if self.max_sell > 0:
            highest_sell = self.order_book.high_volume_quotes(quote=quotes.ask)[0] - 1
            if self.order_book.depth(quote=quotes.ask) < self.order_book.depth(
                quote=quotes.bid
            ) and self.position_pct >= self.product_params.get("threshold", 0):
                orders += self.ladder_orders(
                    max_price=self.max_sell_price,
                    highest_trade=highest_sell,
                    quote=quotes.ask,
                    decay=self.product_params.get("decay", 0),
                )

            else:
                orders += self.rest_order(
                    quote=quotes.ask, price=highest_sell, quantity=self.max_sell
                )
        return orders

    def market_make_RESIN(self) -> list[Order]:
        orders = []

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

            orders += self.rest_order(
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

            orders += self.rest_order(
                quote=quotes.ask,
                price=highest_sell - 1,
                quantity=self.max_sell,
            )

        return orders

    def trade_SQUID(self, spread, preds):
        orders = []

        if len(preds) < 2:
            return orders

        z = self.z_score(spread, preds)
        threshold = self.product_params.get("z_threshold", 0)

        if z < -threshold and self.max_buy > 0:
            for i in range(0, self.order_book.depth(quote=quotes.ask) - 1):
                if self.max_buy < 0:
                    break
                price, ask_vol = self.order_book.get_level(quote=quotes.ask, level=i)
                ask_vol = min(abs(ask_vol), self.max_buy)
                orders += [Order(self.product, price, ask_vol)]
                self.max_buy -= ask_vol

        if z > threshold and self.max_sell > 0:
            for i in range(0, self.order_book.depth(quote=quotes.bid) - 1):
                price, bid_vol = self.order_book.get_level(quote=quotes.bid, level=i)
                bid_vol = min(bid_vol, self.max_sell)
                orders += [Order(self.product, price, -bid_vol)]
                self.max_sell -= bid_vol

        return orders

    def trade_PICNIC(
        self,
        delta,
        coin_forecast_cache_ts,
        prem_disc,
        prem_disc_cache,
    ) -> list[Order]:
        orders = []

        if len(prem_disc_cache) < 2:
            return orders

        prem_disc_cache = np.array(prem_disc_cache)
        prem_disc_diff = np.diff(prem_disc_cache)
        z = self.z_score((prem_disc - prem_disc_cache[-1]), prem_disc_diff)
        z2 = self.z_score(delta, np.array(coin_forecast_cache_ts))

        threshold = self.product_params.get("z_threshold", 0)
        threshold2 = self.product_params.get("z_threshold2", 0)

        if z < -threshold and z2 > threshold2 and self.max_buy > 0:
            orders += self.fv_arb(quote=quotes.ask)
        if z > threshold and z2 < -threshold2 and self.max_sell > 0:
            orders += self.fv_arb(quote=quotes.bid)

        if self.product == tradable_product.PICNIC_BASKET1:
            if self.max_buy > 0:
                highest_buy = (
                    self.order_book.high_volume_quotes(quote=quotes.bid)[0] + 1
                )
                orders += self.rest_order(
                    quote=quotes.bid, price=highest_buy, quantity=self.max_buy
                )

            if self.max_sell > 0:
                highest_sell = (
                    self.order_book.high_volume_quotes(quote=quotes.ask)[0] - 1
                )
                orders += self.rest_order(
                    quote=quotes.ask, price=highest_sell, quantity=self.max_sell
                )

        return orders

    def trade_JAMS(self, coin_forecast_cache_ts):
        orders = []
        if len(coin_forecast_cache_ts) < 2:
            return orders

        threshold = self.product_params.get("z_threshold", 0)
        z = self.z_score(self.fair_value, np.array(coin_forecast_cache_ts))

        if z > threshold:
            for ask, ask_vol in self.order_book.asks:
                buy_amount = min(abs(ask_vol), self.max_buy)
                orders += [Order(self.product, ask, buy_amount)]
                self.max_buy -= buy_amount

        if z < -threshold:
            for bid, bid_vol in self.order_book.bids:
                sell_amount = min(bid_vol, self.max_sell)
                orders += [Order(self.product, bid, -sell_amount)]
                self.max_sell -= sell_amount

        return orders

    def trade_CROISSANTS(self, coin_forecast_cache_ts):
        orders = []
        if len(coin_forecast_cache_ts) < 2:
            return orders

        threshold = self.product_params.get("z_threshold", 0)
        z = self.z_score(self.fair_value, np.array(coin_forecast_cache_ts))

        if z > threshold:
            highest_buy = self.order_book.top_of_book(quote=quotes.ask)[0]
            orders += [Order(self.product, highest_buy, self.max_buy)]
        if z < -threshold:
            highest_sell = self.order_book.top_of_book(quote=quotes.bid)[0]
            orders += [Order(self.product, highest_sell, -self.max_sell)]

        return orders

    def trade_DJEMBES(self, coin_forecast_cache_ts):
        orders = []
        if len(coin_forecast_cache_ts) < 2:
            return orders

        threshold = self.product_params.get("z_threshold", 0)
        z = self.z_score(self.fair_value, np.array(coin_forecast_cache_ts))

        if z > threshold:
            highest_buy = self.order_book.top_of_book(quote=quotes.ask)[0]
            orders += [Order(self.product, highest_buy, self.max_buy)]
        if z < -threshold:
            highest_sell = self.order_book.top_of_book(quote=quotes.bid)[0]
            orders += [Order(self.product, highest_sell, -self.max_sell)]

        return orders

    def CALL_LB_ARB(
        self,
    ):
        orders = []
        if self.fair_value < 0 and self.max_buy > 0:
            for ask, ask_vol in self.order_book.asks:
                buy_amount = min(abs(ask_vol), self.max_buy)
                orders += [Order(self.product, ask, buy_amount)]
                self.max_buy -= buy_amount
        return orders

    def BUTTERFLY_ARB(self, leg):
        orders = []

        if leg == "SHORT":
            if self.fair_value < 0 and self.max_sell > 0:
                for bid, bid_vol in self.order_book.bids:
                    sell_amount = min(bid_vol, self.max_sell)
                    orders += [Order(self.product, bid, -sell_amount)]
                    self.max_sell -= sell_amount

        if leg == "LONG":
            if self.fair_value < 0 and self.max_buy > 0:
                for ask, ask_vol in self.order_book.asks:
                    buy_amount = min(abs(ask_vol), self.max_buy)
                    orders += [Order(self.product, ask, buy_amount)]
                    self.max_buy -= buy_amount

        return orders

    def BCP_ARB(self, leg):
        orders = []

        if leg == "LONG":
            if self.fair_value < 0 and self.max_buy > 0:
                for ask, ask_vol in self.order_book.asks:
                    buy_amount = min(abs(ask_vol), self.max_buy)
                    orders += [Order(self.product, ask, buy_amount)]
                    self.max_buy -= buy_amount

        if leg == "SHORT":
            if self.fair_value < 0 and self.max_sell > 0:
                for bid, bid_vol in self.order_book.bids:
                    sell_amount = min(bid_vol, self.max_sell)
                    orders += [Order(self.product, bid, -sell_amount)]
                    self.max_sell -= sell_amount

        return orders

    def TRADE_VOL(self, iv, iv_cache):
        orders = []
        if len(iv_cache) < 10:
            return orders

        iv_cache = np.array(iv_cache)
        iv_cache_diff = np.diff(iv_cache)
        curr_diff = iv - iv_cache[-1]

        threshold = self.product_params.get("z_threshold", 0)
        z = self.z_score(curr_diff, iv_cache_diff)

        if z < -threshold and self.max_buy > 0:
            ask, ask_vol = self.order_book.top_of_book(quote=quotes.ask)
            buy_amount = min(abs(ask_vol), self.max_buy)
            orders += [Order(self.product, ask, buy_amount)]
            self.max_buy -= buy_amount
        if z > threshold and self.max_sell > 0:
            bid, bid_vol = self.order_book.top_of_book(quote=quotes.bid)
            sell_amount = min(bid_vol, self.max_sell)
            orders += [Order(self.product, bid, -sell_amount)]
            self.max_sell -= sell_amount

        if z > threshold and self.position_limit > 0 and self.max_sell > 0:
            bid, bid_vol = self.order_book.top_of_book(quote=quotes.bid)
            sell_amount = min(bid_vol, self.max_sell)
            orders += [Order(self.product, bid, -sell_amount)]
            self.max_sell -= sell_amount
        if z < -threshold and self.position_pct < 0 and self.max_buy > 0:
            ask, ask_vol = self.order_book.top_of_book(quote=quotes.ask)
            buy_amount = min(abs(ask_vol), self.max_buy)
            orders += [Order(self.product, ask, buy_amount)]
            self.max_buy -= buy_amount

        return orders

    def SHORT_THETA(
        self,
    ):
        orders = []
        if self.fair_value < 0.2 or self.fair_value > 0.8:
            for bid, bid_vol in self.order_book.bids:
                sell_amount = min(bid_vol, self.max_sell)
                orders += [Order(self.product, bid, -sell_amount)]
                self.max_sell -= sell_amount
        return orders

    def HEDGE(self, to_hedge):
        orders = []

        hedge_amt = -(self.position - to_hedge)

        if hedge_amt > 0 and self.max_buy > 0:
            for ask, ask_vol in self.order_book.asks:
                buy_amount = min(abs(ask_vol), self.max_buy, hedge_amt)
                orders += [Order(self.product, ask, buy_amount)]
                hedge_amt -= ask_vol
                self.max_buy -= buy_amount
                if hedge_amt <= 0:
                    break

        elif hedge_amt < 0 and self.max_sell > 0:
            hedge_amt = abs(hedge_amt)
            for bid, bid_vol in self.order_book.bids:
                sell_amount = min(bid_vol, self.max_sell, hedge_amt)
                orders += [Order(self.product, bid, -sell_amount)]
                hedge_amt += sell_amount
                self.max_sell -= sell_amount
                if hedge_amt >= 0:
                    break
        return orders


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
                },
                tradable_product.KELP: {
                    "alpha": np.array([[-6.20579905e-04], [3.58373569e-05]]),
                    "beta": np.array([[1.0, -0.91325515]]),
                    "gamma": np.array(
                        [[0.00538188, -0.34236735], [0.00595464, -0.48509457]]
                    ),
                    "decay": 0.6,
                    "threshold": 0.8,
                },
                tradable_product.SQUID_INK: {
                    "z_threshold": 1.7,
                },
                tradable_product.PICNIC_BASKET1: {
                    tradable_product.CROISSANTS: 6,
                    tradable_product.JAMS: 3,
                    tradable_product.DJEMBES: 1,
                    "alpha": np.array([[-0.00077897], [0.0]]),
                    "beta": np.array([[1.0, -0.07278235]]),
                    "gamma": np.array(
                        [[-0.13665529, 0.00551131], [0.46266129, -0.0553495]]
                    ),
                    "z_threshold": 2.2,
                    "z_threshold2": 2.0,
                },
                tradable_product.PICNIC_BASKET2: {
                    tradable_product.CROISSANTS: 4,
                    tradable_product.JAMS: 2,
                    tradable_product.DJEMBES: 0,
                    "alpha": np.array([[-0.00018354], [0.00021965]]),
                    "beta": np.array([[1.0, -1.944405]]),
                    "gamma": np.array(
                        [[-0.03897751, -0.00675144], [-0.0137392, -0.01803694]]
                    ),
                    "z_threshold": 1.4,
                    "z_threshold2": 2,
                },
                tradable_product.CROISSANTS: {
                    "alpha": np.array([[-0.00077897], [0.0]]),
                    "beta": np.array([[1.0, -0.07278235]]),
                    "gamma": np.array(
                        [[-0.13665529, 0.00551131], [0.46266129, -0.0553495]]
                    ),
                    "z_threshold": 3.0,
                },
                tradable_product.JAMS: {
                    "alpha": np.array([[0.0], [0.0]]),
                    "beta": np.array([[1.0, -0.11138969]]),
                    "gamma": np.array(
                        [[-0.05566674, 0.00370577], [0.19408842, -0.04752362]]
                    ),
                    "z_threshold": 5.6,
                },
                tradable_product.DJEMBES: {
                    "alpha": np.array([[0.0], [0.00141277]]),
                    "beta": np.array([[1.0, -0.22807497]]),
                    "gamma": np.array(
                        [[0.01875615, 0.00280019], [0.05341494, -0.04265734]]
                    ),
                    "z_threshold": 3.6,
                },
                tradable_product.VOLCANIC_ROCK: {"pos": 0, "to_hedge": 0},
                tradable_product.VOLCANIC_ROCK_VOUCHER_9500: {
                    "strike": 9500,
                    "dte": 5e6,
                    "z_threshold": 1.059128940308789,
                },
                tradable_product.VOLCANIC_ROCK_VOUCHER_9750: {
                    "strike": 9750,
                    "dte": 5e6,
                    "z_threshold": 0.8600954832723592,
                },
                tradable_product.VOLCANIC_ROCK_VOUCHER_10000: {
                    "strike": 10_000,
                    "dte": 5e6,
                    "z_threshold": 1.2651093313307173,
                },
                tradable_product.VOLCANIC_ROCK_VOUCHER_10250: {
                    "strike": 10_250,
                    "dte": 5e6,
                    "z_threshold": 1.3923953674290663,
                },
                tradable_product.VOLCANIC_ROCK_VOUCHER_10500: {
                    "strike": 10_500,
                    "dte": 5e6,
                    "z_threshold": 1.6777865342072642,
                },
            }

    mid_cache = copy.deepcopy(empty_dict_cache)
    forecast_cache = copy.deepcopy(empty_dict_cache)

    coin_forecast_cache_kelp = []
    coin_forecast_cache_squid = []
    coin_forecast_cache_croissants = []
    coin_forecast_cache_p1 = []
    coin_forecast_cache_p2 = []

    coin_forecast_cache_ts = copy.deepcopy(empty_dict_cache)
    prem_disc_cache = copy.deepcopy(empty_dict_cache)
    pred_prem_disc_cache = copy.deepcopy(empty_dict_cache)
    iv_cache = copy.deepcopy(empty_dict_cache)

    def process_traders(self, trades) -> tuple[dict, dict]:
        buyer, seller = {}, {}
        if trades:
            for i in trades:
                buyer[i.buyer] = i.quantity
                seller[i.seller] = i.quantity
        return buyer, seller

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

        options = [
            tradable_product.VOLCANIC_ROCK_VOUCHER_9500,
            tradable_product.VOLCANIC_ROCK_VOUCHER_9750,
            tradable_product.VOLCANIC_ROCK_VOUCHER_10000,
            tradable_product.VOLCANIC_ROCK_VOUCHER_10250,
            tradable_product.VOLCANIC_ROCK_VOUCHER_10500,
        ]

        butterfly = [
            tradable_product.VOLCANIC_ROCK_VOUCHER_9750,
            tradable_product.VOLCANIC_ROCK_VOUCHER_10000,
            tradable_product.VOLCANIC_ROCK_VOUCHER_10250,
        ]

        BCP = [
            tradable_product.VOLCANIC_ROCK_VOUCHER_9500,
            tradable_product.VOLCANIC_ROCK_VOUCHER_9750,
            tradable_product.VOLCANIC_ROCK_VOUCHER_10000,
            tradable_product.VOLCANIC_ROCK_VOUCHER_10250,
        ]

        self.params[tradable_product.VOLCANIC_ROCK]["pos"] = state.position.get(
            tradable_product.VOLCANIC_ROCK, 0
        )

        for product in state.order_depths:
            curr_pos = state.position.get(product, 0)

            # get order & trade data
            order_depth: OrderDepth = state.order_depths[product]
            order_book = orderbook(order_depth=order_depth)

            trades = state.market_trades.get(product, None)
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
                )

                fv_arb_buy_orders = resin_strat.fv_arb(quote=quotes.ask)
                fv_arb_sell_orders = resin_strat.fv_arb(quote=quotes.bid)

                MM_orders = resin_strat.market_make_RESIN()

                result[product] = fv_arb_buy_orders + fv_arb_sell_orders + MM_orders

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
                )

                fv_arb_buy_orders = kelp_strat.fv_arb(quote=quotes.ask)
                fv_arb_sell_orders = kelp_strat.fv_arb(quote=quotes.bid)

                MM_orders = kelp_strat.market_make_KELP()
                result[product] = fv_arb_buy_orders + fv_arb_sell_orders + MM_orders

                self.coin_forecast_cache_kelp = forecast

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
                )

                squid_orders = squid_strat.trade_SQUID(spread=spread, preds=pred)

                result[product] = squid_orders

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
                )

                picnic1_orders = picnic1_strat.trade_PICNIC(
                    delta=delta[1],
                    coin_forecast_cache_ts=self.coin_forecast_cache_ts[product],
                    prem_disc=prem_disc,
                    prem_disc_cache=self.prem_disc_cache[product],
                )

                result[product] = picnic1_orders

                self.prem_disc_cache[product].append(prem_disc)
                self.coin_forecast_cache_ts[product].append(delta[1])

            if product == tradable_product.PICNIC_BASKET2:
                basket_price = product_fv.PICNIC2_FV(basket_order_depth)

                delta, forecast = product_fv.VECM_FORECAST(
                    prod_1=tradable_product.PICNIC_BASKET1,
                    prod_1_mid=orderbook(
                        order_depth=state.order_depths[tradable_product.PICNIC_BASKET1]
                    ).mid_price(),
                    prod_2=tradable_product.PICNIC_BASKET2,
                    prod_2_mid=orderbook(
                        order_depth=state.order_depths[tradable_product.PICNIC_BASKET2]
                    ).mid_price(),
                    mid_cache=self.mid_cache,
                )

                prem_disc = order_book.mid_price() - basket_price

                picnic2_strat = trading_strategy(
                    product=product,
                    product_params=product_params,
                    position=curr_pos,
                    position_limit=position_limit.PICNIC_BASKET2,
                    order_book=order_book,
                    fair_value=basket_price,
                )

                picnic2_orders = picnic2_strat.trade_PICNIC(
                    delta=delta[1],
                    coin_forecast_cache_ts=self.coin_forecast_cache_ts[product],
                    prem_disc=prem_disc,
                    prem_disc_cache=self.prem_disc_cache[product],
                )

                result[product] = picnic2_orders

                self.prem_disc_cache[product].append(prem_disc)
                self.coin_forecast_cache_ts[product].append(delta[1])

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
                )

                croissants_orders = croissants_strat.trade_CROISSANTS(
                    coin_forecast_cache_ts=self.coin_forecast_cache_ts[product]
                )

                croissants_strat.update_FV(FV=forecast[0])

                croissants_orders += croissants_strat.fv_arb(quote=quotes.bid)
                croissants_orders += croissants_strat.fv_arb(quote=quotes.ask)

                result[product] = croissants_orders

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
                )
                jams_orders = jams_strat.trade_JAMS(
                    coin_forecast_cache_ts=self.coin_forecast_cache_ts[product]
                )

                jams_strat.update_FV(FV=forecast[0])

                jams_orders += jams_strat.fv_arb(quote=quotes.bid)
                jams_orders += jams_strat.fv_arb(quote=quotes.ask)

                result[product] = jams_orders
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
                )
                djembes_orders = djembes_strat.trade_DJEMBES(
                    coin_forecast_cache_ts=self.coin_forecast_cache_ts[product]
                )

                result[product] = djembes_orders
                self.coin_forecast_cache_ts[product].append(delta[0])

            if product in options:
                butterfly_orders = []
                bcp_orders = []
                call_lb_orders = []
                short_theta = []
                idx = options.index(product)

                s, k, r, t = (
                    orderbook(
                        order_depth=state.order_depths[tradable_product.VOLCANIC_ROCK]
                    ).mid_price(),
                    product_params["strike"],
                    0.0,
                    product_params["dte"] / (365 * 1e6),
                )

                sigma = product_fv.solve_iv(
                    call_price=order_book.mid_price(), S=s, K=k, r=r, T=t
                )

                if sigma > 1 or sigma < 0:
                    if len(self.iv_cache[product]) == 0:
                        sigma = 0
                    else:
                        sigma = self.iv_cache[product][-1]

                option_strat = trading_strategy(
                    product=product,
                    product_params=product_params,
                    position=curr_pos,
                    position_limit=position_limit.VOLCANIC_ROCK_VOUCHER_9500,
                    order_book=order_book,
                    fair_value=0,
                )

                delta = product_fv.call_delta(S=s, K=k, r=r, sigma=sigma, T=t)
                option_strat.update_FV(FV=delta)
                short_theta = option_strat.SHORT_THETA()

                call_lb = product_fv.CALL_LB(
                    payoff=max(s - k, 0), call_price=order_book.mid_price()
                )

                option_strat.update_FV(FV=call_lb)

                call_lb_orders += option_strat.CALL_LB_ARB()

                if product in butterfly:
                    butterfly_spread = product_fv.BUTTERFLY(
                        lower=orderbook(
                            order_depth=state.order_depths[options[idx - 1]]
                        ).mid_price(),
                        mid=order_book.mid_price(),
                        upper=orderbook(
                            order_depth=state.order_depths[options[idx + 1]]
                        ).mid_price(),
                    )

                    option_strat.update_FV(FV=butterfly_spread)
                    butterfly_orders += option_strat.BUTTERFLY_ARB(leg="SHORT")

                for i in range(idx, len(options)):
                    if i + 2 < len(options):
                        butterfly_spread = product_fv.BUTTERFLY(
                            lower=order_book.mid_price(),
                            mid=orderbook(
                                order_depth=state.order_depths[options[i + 1]]
                            ).mid_price(),
                            upper=orderbook(
                                order_depth=state.order_depths[options[i + 2]]
                            ).mid_price(),
                        )

                        option_strat.update_FV(FV=butterfly_spread)
                        butterfly_orders += option_strat.BUTTERFLY_ARB(leg="LONG")

                for i in range(idx, len(options)):
                    if i - 2 >= 0:
                        butterfly_spread = product_fv.BUTTERFLY(
                            lower=orderbook(
                                order_depth=state.order_depths[options[i - 2]]
                            ).mid_price(),
                            mid=orderbook(
                                order_depth=state.order_depths[options[i - 1]]
                            ).mid_price(),
                            upper=order_book.mid_price(),
                        )

                        option_strat.update_FV(FV=butterfly_spread)
                        butterfly_orders += option_strat.BUTTERFLY_ARB(leg="LONG")

                if product in BCP:
                    for i in range(idx + 1, len(BCP)):
                        spread = product_fv.BULL_CALL_SPREAD(
                            lower=order_book.mid_price(),
                            upper=orderbook(
                                order_depth=state.order_depths[options[i]]
                            ).mid_price(),
                            lower_strike=self.params[product]["strike"],
                            upper_strike=self.params[options[i]]["strike"],
                        )

                        option_strat.update_FV(FV=spread)
                        bcp_orders += option_strat.BCP_ARB(leg="LONG")

                for i in range(idx + 1, len(BCP)):
                    if i - 1 >= 1:
                        spread = product_fv.BULL_CALL_SPREAD(
                            lower=orderbook(
                                order_depth=state.order_depths[options[i - 1]]
                            ).mid_price(),
                            upper=order_book.mid_price(),
                            lower_strike=self.params[product]["strike"],
                            upper_strike=self.params[options[i]]["strike"],
                        )

                        option_strat.update_FV(FV=spread)
                        bcp_orders += option_strat.BCP_ARB(leg="SHORT")

                option_strat.update_FV(FV=sigma)

                option_trades = option_strat.TRADE_VOL(
                    iv=sigma,
                    iv_cache=self.iv_cache[product],
                )

                delta = product_fv.call_delta(S=s, K=k, r=r, sigma=sigma, T=t)

                self.params[tradable_product.VOLCANIC_ROCK]["to_hedge"] += sum(
                    [
                        -math.floor(order.quantity * delta)
                        for order in option_trades
                        if option_trades
                    ]
                    + [
                        -math.floor(order.quantity * delta)
                        for order in short_theta
                        if short_theta
                    ]
                )

                all_orders = {}
                for order in (
                    short_theta
                    + call_lb_orders
                    + butterfly_orders
                    + bcp_orders
                    + option_trades
                ):
                    if order.price not in all_orders.keys():
                        all_orders[order.price] = 0
                    all_orders[order.price] += order.quantity

                agg_orders = []
                for price, qty in all_orders.items():
                    agg_orders += [Order(product, price, qty)]

                result[product] = agg_orders
                product_params["dte"] -= 100

                self.iv_cache[product].append(sigma)

        if product == tradable_product.VOLCANIC_ROCK:
            option_strat = trading_strategy(
                product=product,
                product_params=product_params,
                position=curr_pos,
                position_limit=position_limit.VOLCANIC_ROCK,
                order_book=order_book,
                fair_value=0,
            )

            hedge_orders = option_strat.HEDGE(
                to_hedge=product_params["to_hedge"],
            )
            result[product] = hedge_orders

            logger.print("HEDGEEEE", product_params["to_hedge"])

            self.params[tradable_product.VOLCANIC_ROCK]["to_hedge"] = 0

        for product in state.order_depths:
            self.mid_cache[product].append(
                orderbook(order_depth=state.order_depths[product]).mid_price()
            )

        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data
