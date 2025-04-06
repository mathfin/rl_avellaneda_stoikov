import pandas as pd


class MarketMakerManager:
    def __init__(self, avellaneda_model):
        self.model = avellaneda_model
        self.q = 0
        self.t = 0
        self.cash_surplus = 0
        self.last_timestamp = None
        self.order_quantity = None

    def step(self, actual_data: pd.DataFrame):
        market_mid_price = self.estimate_mid_price(actual_data.price, actual_data.qty)
        market_std = self.estimate_dispersion(actual_data.price, actual_data.qty)

        r_price = self.model.reservation_price(market_mid_price, market_std, self.q, self.t)
        spread = self.model.optimal_spread()

        bid = r_price - spread / 2
        ask = r_price + spread / 2

        return bid, ask

    def update_state(self, actual_data, bid, ask, time_elapsed):
        self.t += time_elapsed

        relevant_data = actual_data[(actual_data['timestamp'] > self.last_timestamp) &
                                    (actual_data['timestamp'] <= self.last_timestamp + time_elapsed)]

        if relevant_data.shape[0] == 0:
            return 0, None, None, 0, 0, None

        quantile_05_price = relevant_data.price.quantile(0.05)
        quantile_95_price = relevant_data.price.quantile(0.95)

        executed_qty_bid = relevant_data[relevant_data['price'] <= bid]['qty'].sum()
        executed_qty_ask = relevant_data[relevant_data['price'] >= ask]['qty'].sum()

        executed_bid = min(executed_qty_bid, self.order_quantity)
        executed_ask = min(executed_qty_ask, self.order_quantity)

        q_change = executed_bid - executed_ask
        cash_change = ((executed_ask * ask) - (executed_bid * bid))

        self.q += q_change
        self.cash_surplus += cash_change

        avg_price = self.estimate_mid_price(relevant_data.price, relevant_data.qty)
        current_inventory_price = avg_price * self.q
        current_profit = self.cash_surplus + current_inventory_price

        self.last_timestamp += time_elapsed
        return (relevant_data.shape[0], quantile_05_price, quantile_95_price,
                executed_bid, executed_ask, current_profit)

    @staticmethod
    def estimate_mid_price(market_price: pd.Series, market_qty: pd.Series):
        return (market_price * market_qty).sum() / market_qty.sum()

    @staticmethod
    def estimate_dispersion(market_price: pd.Series, market_qty: pd.Series):
        """
          Не нашел явного указания в оригинальной статье,
          если правильно понял, то дисперсия там возникает,
          чтобы дисперсия у нормального распределения приращений
          в Винеровском процессе была не T-t, а такая же как у актива, то есть std^2 * (T-t)
          поэтому тут дисперсия у .diff рассчитана, а не у цены

          TODO мне кажется гораздо эффективней будет сделать
          модель прогнозирующую дисперсию на основе внешних данных,
          например когда известно что будет заседание по ставке,
          по идее если ожидают большую дисперсию
          это не должно отражаться на цене базового актива,
          но должно отражаться на цене опционов,
          можно на этих данных построить модель
          и добавить сюда вместо этой функции,
          прогнозируя дисперсию на перед и снижая риск,
          но сейчас нет времени на эксперименты

        """
        price_diff = market_price.diff().dropna()
        qty_aligned = market_qty.iloc[1:]

        math_exp = (price_diff * qty_aligned).sum() / qty_aligned.sum()
        second_moment = ((price_diff ** 2) * qty_aligned).sum() / qty_aligned.sum()

        dispersion = second_moment - math_exp ** 2

        return dispersion
