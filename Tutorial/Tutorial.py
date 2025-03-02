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

empty_dict_cache = {"KELP": deque(maxlen=50), "RAINFOREST_RESIN": deque(maxlen=50)}


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
            compressed.append(
                [listing["symbol"], listing["product"], listing["denomination"]]
            )

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


class trade_direction:
    buy = "buy"
    sell = "sell"


class trading_strategy:
    def __init__(
        self,
        product,
        position,
        position_limit,
        bids,
        asks,
        fair_value,
    ):
        self.product = product
        self.max_buy = position_limit - position
        self.max_sell = position_limit + position
        self.fair_value = fair_value
        self.bids, self.asks = bids, asks

    def market_make(self, ask_slip, bid_slip):
        orders = []

        for ask, ask_vol in self.asks:
            ask_vol = abs(ask_vol)
            if ask <= self.fair_value + ask_slip:  # if ask < fv want to buy
                buy_amount = min(ask_vol, self.max_buy)
                orders.append(Order(self.product, ask, buy_amount))
                self.max_buy -= buy_amount

        if self.max_buy > 0:
            highest_buy = max(self.bids, key=lambda tup: tup[1])[0] + 1
            orders.append(
                Order(
                    self.product, int(min(self.fair_value, highest_buy)), self.max_buy
                )
            )

        for bid, bid_vol in self.bids:
            bid_vol = abs(bid_vol)
            if bid >= self.fair_value + bid_slip:  # if bid > fv want to sell
                sell_amount = min(bid_vol, self.max_sell)
                orders.append(Order(self.product, bid, -sell_amount))
                self.max_sell -= sell_amount

        if self.max_sell > 0:
            highest_sell = min(self.asks, key=lambda tup: tup[1])[0] - 1
            orders.append(
                Order(
                    self.product,
                    int(max(self.fair_value, highest_sell)),
                    -self.max_sell,
                )
            )
        logger.print(orders)
        return orders


class Trader:
    def __init__(self, params=None):
        if params is not None:
            self.params = params
        else:
            self.params = {
                tradable_product.RAINFOREST_RESIN: {
                    "fair_value": 10_000,
                    "ask_slip": -2,
                    "bid_slip": 1,
                },
                tradable_product.KELP: {
                    "MA1_coeff": -0.6130858530887017,
                    "window": 27,
                    "ask_slip": -2,
                    "bid_slip": 0,
                },
            }

    position = copy.deepcopy(empty_dict)
    POSITION_LIMIT = {"KELP": 50, "RAINFOREST_RESIN": 50}

    spread_cache = copy.deepcopy(empty_dict_cache)
    bid_cache = copy.deepcopy(empty_dict_cache)
    ask_cache = copy.deepcopy(empty_dict_cache)
    mid_cache = copy.deepcopy(empty_dict_cache)
    forecast_cache = copy.deepcopy(empty_dict_cache)

    def directional(
        self,
        product,
        position,
        position_limit,
        bids,
        asks,
        fair_value,
        up_width,
        down_width,
        bid_spread,
        ask_spread,
    ):
        pass

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        conversions = 0
        trader_data = ""
        timestamp = state.timestamp
        logger.print(self.params)

        for key, val in state.position.items():
            self.position[key] = val

        result = {}  # Orders to be placed on exchange matching engine
        for product in state.order_depths:
            # get order & trade data
            order_depth: OrderDepth = state.order_depths[product]
            trades = (
                state.market_trades[product]
                if product in list(state.market_trades.keys())
                else None
            )
            buyer, seller = {}, {}
            if trades is not None:
                for i in trades:
                    buyer[i.buyer] = i.quantity
                    seller[i.seller] = i.quantity

            if product == tradable_product.RAINFOREST_RESIN:
                product_params = self.params[tradable_product.RAINFOREST_RESIN]
                strat = trading_strategy(
                    product=product,
                    position=self.position[product],
                    position_limit=self.POSITION_LIMIT[product],
                    bids=order_depth.buy_orders.items(),
                    asks=order_depth.sell_orders.items(),
                    fair_value=product_params["fair_value"],
                )

                orders = strat.market_make(
                    ask_slip=product_params["ask_slip"],
                    bid_slip=product_params["bid_slip"],
                )

                result[product] = orders

            if product == tradable_product.KELP:
                product_params = self.params[tradable_product.KELP]
                best_ask, _ = (
                    list(order_depth.sell_orders.items())[0]
                    if len(list(order_depth.sell_orders.items())) > 0
                    else [0, 0]
                )
                best_bid, _ = (
                    list(order_depth.buy_orders.items())[0]
                    if len(list(order_depth.buy_orders.items())) > 0
                    else [0, 0]
                )
                mid_price = (best_bid + best_ask) / 2

                if len(self.mid_cache[product]) > product_params["window"]:
                    ma_window = np.array(self.mid_cache[product])[
                        -product_params["window"] :
                    ]
                    ma_window = np.diff(ma_window)
                    gamma_0 = np.var(ma_window)
                    gamma_1 = np.cov(ma_window[:-1], ma_window[1:])[0, 1]
                    theta_1 = gamma_1 / gamma_0
                else:
                    theta_1 = product_params["MA1_coeff"]

                prev_forecast = (
                    self.forecast_cache[product][-1]
                    if self.forecast_cache[product]
                    else mid_price
                )
                forecast_error = mid_price - prev_forecast
                forecast = mid_price + theta_1 * forecast_error
                forecast = round(forecast * 2) / 2

                # logger.print(f"params: {product_params['window']}")
                # logger.print(f"thetea: {theta_1}")
                # logger.print(f"mid: {mid_price}")
                # logger.print(f"prev_forecast: {prev_forecast}")
                # logger.print(f"forecast_error: {forecast_error}")
                # logger.print(f"forecast: {forecast}")

                strat = trading_strategy(
                    product=product,
                    position=self.position[product],
                    position_limit=self.POSITION_LIMIT[product],
                    bids=order_depth.buy_orders.items(),
                    asks=order_depth.sell_orders.items(),
                    fair_value=forecast,
                )

                orders = strat.market_make(
                    ask_slip=product_params["ask_slip"],
                    bid_slip=product_params["bid_slip"],
                )

                result[product] = orders

                self.forecast_cache[product].append(forecast)
                self.mid_cache[product].append(mid_price)

            result[product] = orders

        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data
