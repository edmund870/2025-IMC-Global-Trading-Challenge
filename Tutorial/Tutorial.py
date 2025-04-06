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


empty_dict = {"KELP": 0, "RAINFOREST_RESIN": 0}

empty_dict_cache = {"KELP": deque(maxlen=1_001), "RAINFOREST_RESIN": deque(maxlen=200)}


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


class position_limit:
    KELP = 50
    RAINFOREST_RESIN = 50


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
            return max(self.bids, key=lambda tup: tup[0])
        if quote == quotes.ask:
            return min(self.asks, key=lambda tup: tup[0])

    def get_level(self, quote: quotes, level):
        if quote == quotes.bid:
            return list(self.bids)[level]
        if quote == quotes.ask:
            return list(self.asks)[level]

    def high_volume_quotes(self, quote: quotes) -> tuple[int, int]:
        if quote == quotes.bid:
            return max(self.bids, key=lambda tup: tup[1])
        if quote == quotes.ask:
            return min(self.asks, key=lambda tup: tup[1])

    def spread(self):
        return abs(
            self.top_of_book(quote=quotes.bid)[0]
            - self.top_of_book(quote=quotes.ask)[0]
        )

    def most_micro_price(self):
        most_ask, most_ask_vol = self.high_volume_quotes(quote=quotes.ask)
        most_bid, most_bid_vol = self.high_volume_quotes(quote=quotes.bid)
        return (most_ask * most_bid_vol + most_bid * abs(most_ask_vol)) / (
            abs(most_ask_vol) + most_bid_vol
        )

    def micro_price(self):
        best_ask, best_ask_vol = self.top_of_book(quote=quotes.ask)
        best_bid, best_bid_vol = self.top_of_book(quote=quotes.bid)
        return (best_ask * best_bid_vol + best_bid * abs(best_ask_vol)) / (
            abs(best_ask_vol) + best_bid_vol
        )

    def most_mid_price(self):
        most_ask, _ = self.high_volume_quotes(quote=quotes.ask)
        most_bid, _ = self.high_volume_quotes(quote=quotes.bid)
        return (most_ask + most_bid) / 2

    def mid_price(self):
        best_ask, _ = self.top_of_book(quote=quotes.ask)
        best_bid, _ = self.top_of_book(quote=quotes.bid)
        return (best_ask + best_bid) / 2

    def total_volume(self, quote: quotes):
        volume = 0
        if quote == quotes.bid:
            for _, vol in self.bids:
                volume += vol

        if quote == quotes.ask:
            for _, vol in self.asks:
                volume += abs(vol)

        return volume

    def skew(self):
        bid_vol = self.total_volume(quote=quotes.bid)
        ask_vol = self.total_volume(quote=quotes.ask)
        return (bid_vol - ask_vol) / (ask_vol + bid_vol)

    def volume_ratio(self):
        bid_vol = self.total_volume(quote=quotes.bid)
        ask_vol = self.total_volume(quote=quotes.ask)
        return bid_vol / ask_vol

    def depth(self, quote: quotes):
        if quote == quotes.bid:
            return len(self.bids)
        if quote == quotes.ask:
            return len(self.asks)


class fair_value:
    def __init__(self, product: tradable_product, order_book: orderbook):
        self.product = product
        self.order_book = order_book

    def RAINFOREST_RESIN_FV(
        self,
    ) -> int:
        base = 10_000
        return base

    def log_likelihood(self, sigma2, y_diff, residual):
        T = len(y_diff)

        ll = -T / 2 * np.log(2 * np.pi * sigma2) - (1 / (2 * sigma2)) * np.sum(
            residual**2
        )

        return -ll

    def gradient(self, mu, sigma2, params, y_diff, X1_diff, X2_diff):
        theta_1, beta_1, beta_2 = params
        y_diff_prev, X1_diff, X2_diff = (
            y_diff[:-1],
            X1_diff[1:],
            X2_diff[1:],
        )
        residual = (
            y_diff[1:]
            - mu
            - theta_1 * y_diff_prev
            - beta_1 * X1_diff
            - beta_2 * X2_diff
        )

        dL_dtheta_1 = np.sum(y_diff_prev * (residual)) / sigma2
        dL_dbeta_1 = np.sum(X1_diff * (residual)) / sigma2
        dL_dbeta_2 = np.sum(X2_diff * (residual)) / sigma2

        return np.array([-dL_dtheta_1, -dL_dbeta_1, -dL_dbeta_2]), residual

    def gradient_descent(
        self,
        mu,
        sigma2,
        y_diff,
        X1_diff,
        X2_diff,
        initial_params,
        learning_rate=np.array([12e-7, 2e-6, 16e-7]),
        max_iter=100,
        tolerance=1e-5,
    ):
        params = np.array(initial_params)
        prev_loss = 1e10
        for _ in range(max_iter):
            grad, residual = self.gradient(mu, sigma2, params, y_diff, X1_diff, X2_diff)
            loss = self.log_likelihood(sigma2=sigma2, y_diff=y_diff, residual=residual)
            params -= learning_rate * grad

            if prev_loss - loss < tolerance:
                break
            else:
                prev_loss = loss
        return params

    def KELP_FV(
        self,
        product_params: dict,
        mid_cache: deque,
        micro_cache: deque,
        most_mid_cache: deque,
        forecast_cache: deque,
    ) -> tuple[int, int, float]:
        most_mid = self.order_book.most_mid_price()
        micro_price = self.order_book.micro_price()
        mid_price = self.order_book.mid_price()

        if len(micro_cache) > 1:
            calib_params = self.gradient_descent(
                mu=product_params["mu"],
                sigma2=product_params["sigma2"],
                y_diff=np.diff(mid_cache),
                X1_diff=np.diff(most_mid_cache),
                X2_diff=np.diff(micro_cache),
                initial_params=[-0.603, 0.1, 0.237],
            )

            theta_1 = calib_params[0]
            alpha_1 = calib_params[1]
            alpha_2 = calib_params[2]
        else:
            theta_1 = product_params["MA1_coeff"]
            alpha_1 = product_params["most_mid_coeff"]
            alpha_2 = product_params["micro_price_coeff"]

        if len(micro_cache) >= 2:
            micro_delta = micro_cache[-1] - micro_cache[-2]
            most_mid_delta = most_mid_cache[-1] - most_mid_cache[-2]
        else:
            micro_delta = 0
            most_mid_delta = 0

        prev_forecast = forecast_cache[-1] if forecast_cache else mid_price
        forecast_error = mid_price - prev_forecast
        forecast = (
            mid_price
            + theta_1 * forecast_error
            + alpha_1 * most_mid_delta
            + alpha_2 * micro_delta
        )

        forecast = mid_price if math.isnan(forecast) else forecast

        return mid_price, micro_price, most_mid, forecast


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
        self.product_params = product_params
        self.position_pct = position / position_limit
        self.max_buy = position_limit - position
        self.max_sell = position_limit + position
        self.fair_value = fair_value
        self.order_book = order_book
        self.max_buy_price = fair_value + product_params.get("ask_slip", 0)
        self.max_sell_price = fair_value + product_params.get("bid_slip", 0)

    def ladder(self, qty, n, decay):
        raw_values = [math.exp(-decay * i) for i in range(1, n + 1)]
        sum_raw = sum(raw_values)
        ladder = [math.floor(qty * value / sum_raw) for value in raw_values]
        return ladder

    def ladder_orders(self, max_price, highest_trade, quote, decay):
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

    def fv_arb(self, quote, avg_price):
        orders = []
        if quote == quotes.bid:
            for bid, bid_vol in self.order_book.bids:
                bid_vol = abs(bid_vol)
                if (
                    bid >= self.max_sell_price and bid >= avg_price
                ):  # if bid > fv want to sell
                    sell_amount = min(bid_vol, self.max_sell)
                    orders.append(Order(self.product, bid, -sell_amount))
                    self.max_sell -= sell_amount
                else:
                    break

        if quote == quotes.ask:
            for ask, ask_vol in self.order_book.asks:
                ask_vol = abs(ask_vol)
                if (
                    ask <= self.max_buy_price and ask <= avg_price
                ):  # if ask < fv want to buy
                    buy_amount = min(ask_vol, self.max_buy)
                    orders.append(Order(self.product, ask, buy_amount))
                    self.max_buy -= buy_amount
                else:
                    break

        return orders

    def rest_order(self, quote, price, quantity):
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

        # logger.print(self.product, self.fair_value)
        # logger.print(orders)
        return orders

    def directional(
        self,
    ):
        pass


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
                    "mu": 0,
                    "sigma2": 0.3798141,
                    "MA1_coeff": -0.56866306,
                    "most_mid_coeff": 0.12107209,
                    "micro_price_coeff": 0.27218619,
                    "decay": 0.6,
                    "threshold": 0.8,
                },
            }

    position_cache = copy.deepcopy(empty_dict_cache)
    avg_long_price_cache = copy.deepcopy(empty_dict_cache)
    avg_short_price_cache = copy.deepcopy(empty_dict_cache)
    long_qty_cache = copy.deepcopy(empty_dict_cache)
    short_qty_cache = copy.deepcopy(empty_dict_cache)
    long_price_cache = copy.deepcopy(empty_dict_cache)
    short_price_cache = copy.deepcopy(empty_dict_cache)

    mid_cache = copy.deepcopy(empty_dict_cache)
    micro_cache = copy.deepcopy(empty_dict_cache)
    most_mid_cache = copy.deepcopy(empty_dict_cache)
    forecast_cache = copy.deepcopy(empty_dict_cache)

    def compute_avg_cost(self, own_trades, timestamp, product, curr_pos):
        if own_trades:
            for trade in own_trades:
                if trade.timestamp == timestamp - 100:
                    if trade.seller == "SUBMISSION":  # sell
                        self.short_qty_cache[product].append(-trade.quantity)
                        self.short_price_cache[product].append(trade.price)
                    else:
                        self.short_qty_cache[product].append(0)
                        self.short_price_cache[product].append(0)
                    if trade.buyer == "SUBMISSION":
                        self.long_qty_cache[product].append(trade.quantity)
                        self.long_price_cache[product].append(trade.price)
                    else:
                        self.long_qty_cache[product].append(0)
                        self.long_price_cache[product].append(0)

        idx = -1
        long_qty = self.long_qty_cache[product]
        short_qty = self.short_qty_cache[product]
        long_price = self.long_price_cache[product]
        short_price = self.short_price_cache[product]
        long_avg, short_avg = 0, 0
        if curr_pos != 0:
            temp_pos = 0
            for i in range(len(long_qty) - 1, -1, -1):
                temp_pos += short_qty[i] + long_qty[i]
                if curr_pos > 0 and temp_pos >= curr_pos:
                    idx = i
                    break
                elif curr_pos < 0 and temp_pos <= curr_pos:
                    idx = i
                    break
                else:
                    continue
        if idx != -1:
            qty = np.array(long_qty)[idx:]
            price = np.array(long_price)[idx:]
            if sum(qty) == 0:
                long_avg = 0
            else:
                long_avg = sum(price * qty) / sum(qty)

            qty = np.array(short_qty)[idx:]
            price = np.array(short_price)[idx:]

            if sum(qty) == 0:
                short_avg = 1e6
            else:
                short_avg = sum(price * qty) / sum(qty)
        return long_avg, short_avg

    def process_traders(self, trades):
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
        logger.print(self.params)

        result = {}  # Orders to be placed on exchange matching engine
        for product in state.order_depths:
            curr_pos = state.position.get(product, 0)
            self.position_cache[product].append(curr_pos)

            own_trades = state.own_trades.get(product)

            long_avg, short_avg = self.compute_avg_cost(
                own_trades=own_trades,
                timestamp=timestamp,
                product=product,
                curr_pos=curr_pos,
            )

            # get order & trade data
            order_depth: OrderDepth = state.order_depths[product]
            order_book = orderbook(order_depth=order_depth)

            trades = state.market_trades.get(product, None)

            if product == tradable_product.RAINFOREST_RESIN:
                product_params = self.params[product]

                resin_FV = fair_value(product=product, order_book=order_book)

                resin_strat = trading_strategy(
                    product=product,
                    product_params=product_params,
                    position=curr_pos,
                    position_limit=position_limit.RAINFOREST_RESIN,
                    order_book=order_book,
                    fair_value=resin_FV.RAINFOREST_RESIN_FV(),
                )

                fv_arb_buy_orders = resin_strat.fv_arb(
                    quote=quotes.ask, avg_price=short_avg
                )
                fv_arb_sell_orders = resin_strat.fv_arb(
                    quote=quotes.bid, avg_price=long_avg
                )

                MM_orders = resin_strat.market_make_RESIN()

                result[product] = fv_arb_buy_orders + fv_arb_sell_orders + MM_orders

            if product == tradable_product.KELP:
                product_params = self.params[product]

                kelp_FV = fair_value(product=product, order_book=order_book)
                mid_price, micro_price, most_mid_price, forecast = kelp_FV.KELP_FV(
                    product_params=product_params,
                    mid_cache=self.mid_cache[product],
                    micro_cache=self.micro_cache[product],
                    most_mid_cache=self.most_mid_cache[product],
                    forecast_cache=self.forecast_cache[product],
                )

                kelp_strat = trading_strategy(
                    product=product,
                    product_params=product_params,
                    position=curr_pos,
                    position_limit=position_limit.KELP,
                    order_book=order_book,
                    fair_value=forecast,
                )

                fv_arb_buy_orders = kelp_strat.fv_arb(
                    quote=quotes.ask, avg_price=short_avg
                )
                fv_arb_sell_orders = kelp_strat.fv_arb(
                    quote=quotes.bid, avg_price=long_avg
                )

                MM_orders = kelp_strat.market_make_KELP()

                result[product] = fv_arb_buy_orders + fv_arb_sell_orders + MM_orders

                self.forecast_cache[product].append(forecast)
                self.mid_cache[product].append(mid_price)
                self.micro_cache[product].append(micro_price)
                self.most_mid_cache[product].append(most_mid_price)

        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data
