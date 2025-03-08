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

empty_dict_cache = {"KELP": deque(maxlen=100), "RAINFOREST_RESIN": deque(maxlen=50)}


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
        """
        Top of book bids and asks

        Parameters
        ----------
        quote : quotes
            bid / ask

        Returns
        -------
        tuple[int, int]
            top of book price, top of book volume
        """
        if quote == quotes.bid:
            return max(self.bids, key=lambda tup: tup[0])
        if quote == quotes.ask:
            return min(self.asks, key=lambda tup: tup[0])

    def high_volume_quotes(self, quote: quotes) -> tuple[int, int]:
        """
        retrive quotes with highest volume

        Parameters
        ----------
        quote : quotes
            bid / ask

        Returns
        -------
        tuple[int, int]
            highest volume price, highest volume volume
        """
        if quote == quotes.bid:
            return max(self.bids, key=lambda tup: tup[1])
        if quote == quotes.ask:
            return min(self.asks, key=lambda tup: tup[1])


class fair_value:
    def __init__(self, product: tradable_product, order_book: orderbook):
        """
        Compute fair value of products

        Parameters
        ----------
        product : tradable_product
            product type
        order_book : orderbook
            orderbook class
        """
        self.product = product
        self.order_book = order_book

    def RAINFOREST_RESIN_FV(self) -> int:
        """
        FV of rainforest resin

        Returns
        -------
        int
            FV
        """
        return 10_000

    def KELP_FV(
        self,
        product_params: dict,
        mid_cache: deque,
        micro_cache: deque,
        forecast_cache: deque,
    ) -> tuple[int, int, float]:
        """
        FV of KELP

        Parameters
        ----------
        product_params : dict
            product specific parameters
        mid_cache : deque
            cache of midprices
        micro_cache : deque
            cache of microprices
        forecast_cache : deque
            cache of forecast

        Returns
        -------
        tuple[int, int, float]
            midprice, microprice, forecast (aka FV)
        """
        best_ask, best_ask_vol = self.order_book.top_of_book(quote=quotes.ask)
        best_bid, best_bid_vol = self.order_book.top_of_book(quote=quotes.bid)

        micro_price = (best_ask * best_bid_vol + best_bid * abs(best_ask_vol)) / (
            abs(best_ask_vol) + best_bid_vol
        )
        mid_price = (best_ask + best_bid) / 2

        if len(mid_cache) > product_params["window"]:
            endog_window = np.array(mid_cache)[-product_params["window"] :]
            endog_window = np.diff(endog_window)
            gamma_0 = np.var(endog_window)
            gamma_1 = np.cov(endog_window[:-1], endog_window[1:])[0, 1]
            theta_1 = gamma_1 / gamma_0
        else:
            theta_1 = product_params["MA1_coeff"]

        if len(micro_cache) >= 2:
            micro_delta = micro_cache[-1] - micro_cache[-2]
        else:
            micro_delta = 0

        prev_forecast = forecast_cache[-1] if forecast_cache else mid_price
        forecast_error = mid_price - prev_forecast
        forecast = (
            mid_price
            + product_params["MA1_coeff"] * forecast_error
            + product_params["exog_coeff"] * micro_delta
        )

        return mid_price, micro_price, forecast


class trading_strategy:
    def __init__(
        self,
        product: tradable_product,
        position: int,
        position_limit: int,
        order_book: orderbook,
        fair_value: int | float,
    ):
        """
        trading strategy

        Parameters
        ----------
        product : tradable_product
            product type
        position : int
            current position
        position_limit : int
            max position
        order_book : orderbook
            orderbook of bids and asks
        fair_value : int | float
            fair value of product
        """
        self.product = product
        self.max_buy = position_limit - position
        self.max_sell = position_limit + position
        self.fair_value = fair_value
        self.order_book = order_book

    def market_make(self, ask_slip: int, bid_slip: int) -> list[Order]:
        """
        Market making strategy
        done by quoting bids and asks below / above fair value

        Parameters
        ----------
        ask_slip : int
            how much below fair value am I willing to buy
        bid_slip : int
            how much above fair value am I willing to sell

        Returns
        -------
        list[Order]
            list of orders
        """

        orders = []
        spread = abs(
            self.order_book.top_of_book(quote=quotes.bid)[0]
            - self.order_book.top_of_book(quote=quotes.ask)[0]
        )

        for ask, ask_vol in self.order_book.asks:
            ask_vol = abs(ask_vol)
            if ask <= self.fair_value + ask_slip:  # if ask < fv want to buy
                buy_amount = min(ask_vol, self.max_buy)
                orders.append(Order(self.product, ask, buy_amount))
                self.max_buy -= buy_amount

        if self.max_buy > 0:
            highest_buy = self.order_book.high_volume_quotes(quote=quotes.bid)[0] + 1
            final_buy_price = int(min(self.fair_value, highest_buy))
            if self.product == tradable_product.RAINFOREST_RESIN and 6 <= spread <= 8:
                final_buy_price += 1
            orders.append(Order(self.product, final_buy_price, self.max_buy))

        for bid, bid_vol in self.order_book.bids:
            bid_vol = abs(bid_vol)
            if bid >= self.fair_value + bid_slip:  # if bid > fv want to sell
                sell_amount = min(bid_vol, self.max_sell)
                orders.append(Order(self.product, bid, -sell_amount))
                self.max_sell -= sell_amount

        if self.max_sell > 0:
            highest_sell = self.order_book.high_volume_quotes(quote=quotes.ask)[0] - 1
            final_sell_price = int(max(self.fair_value, highest_sell))
            if self.product == tradable_product.RAINFOREST_RESIN and 6 <= spread <= 8:
                final_sell_price -= 1
            orders.append(
                Order(
                    self.product,
                    final_sell_price,
                    -self.max_sell,
                )
            )
        logger.print(orders)
        return orders


class Trader:
    def __init__(self, params: dict = None):
        """
        On initialization, check for product parameters

        Parameters
        ----------
        params : dict, optional
            product parameters, by default None
        """
        if params is not None:
            self.params = params
        else:
            self.params = {
                tradable_product.RAINFOREST_RESIN: {
                    "ask_slip": -2,
                    "bid_slip": 1,
                },
                tradable_product.KELP: {
                    "MA1_coeff": -0.56871383,  # -0.6130858530887017,
                    "exog_coeff": 0.30679742,
                    "window": 15,
                    "ask_slip": -2,
                    "bid_slip": 1,
                },
            }

    position = copy.deepcopy(empty_dict)
    POSITION_LIMIT = {"KELP": 50, "RAINFOREST_RESIN": 50}

    spread_cache = copy.deepcopy(empty_dict_cache)
    bid_cache = copy.deepcopy(empty_dict_cache)
    ask_cache = copy.deepcopy(empty_dict_cache)
    mid_cache = copy.deepcopy(empty_dict_cache)
    micro_cache = copy.deepcopy(empty_dict_cache)
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
            order_book = orderbook(order_depth=order_depth)
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
                product_params = self.params[product]
                FV = fair_value(product=product, order_book=order_book)
                strat = trading_strategy(
                    product=product,
                    position=self.position[product],
                    position_limit=self.POSITION_LIMIT[product],
                    order_book=order_book,
                    fair_value=FV.RAINFOREST_RESIN_FV(),
                )

                orders = strat.market_make(
                    ask_slip=product_params["ask_slip"],
                    bid_slip=product_params["bid_slip"],
                )

                result[product] = orders

            if product == tradable_product.KELP:
                product_params = self.params[product]
                FV = fair_value(product=product, order_book=order_book)
                mid_price, micro_price, forecast = FV.KELP_FV(
                    product_params=product_params,
                    mid_cache=self.mid_cache[product],
                    micro_cache=self.micro_cache[product],
                    forecast_cache=self.forecast_cache[product],
                )

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
                    order_book=order_book,
                    fair_value=forecast,
                )

                orders = strat.market_make(
                    ask_slip=product_params["ask_slip"],
                    bid_slip=product_params["bid_slip"],
                )

                result[product] = orders

                self.forecast_cache[product].append(forecast)
                self.mid_cache[product].append(mid_price)
                self.micro_cache[product].append(micro_price)

            del FV, strat, orders
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data
