import pandas as pd
import numpy as np

from trading_env.avellaneda_stoikov import AvellanedaStoikovModel
from trading_env.mm_agent import MarketMakerManager


class CustomEnvironment:
    def __init__(self, initial_timestamp, dataset,
                 T, TICKRATE, WINDOW):
        self.T = T
        self.time_step = T // TICKRATE
        self.simulation_counter = 1
        self.window = WINDOW
        self.order_quantity = 0.001
        self.last_timestamp = initial_timestamp
        self.dataset = dataset  # TODO заменить на обращение в бд
        self.full_history = []

    def step(self, action):

        gamma = action[0]
        k = action[1]

        avellaneda_model = AvellanedaStoikovModel(gamma=gamma, k=k, T=self.T)
        mm_agent = MarketMakerManager(avellaneda_model)

        for t in range(0, self.T, self.time_step):
            actual_data = self._get_dataset()[-self.window:]

            if t == 0:
                mm_agent.last_timestamp = self.last_timestamp
                mm_agent.order_quantity = self.order_quantity
                old_bid, old_ask = None, None
                datashape = 0
                quantile_05_price, quantile_95_price = None, None
                current_q = 0
                current_cash_surplus = 0
                executed_bid, executed_ask, current_profit = 0, 0, 0
            else:
                datashape, quantile_05_price, quantile_95_price, executed_bid, executed_ask, current_profit = mm_agent.update_state(
                    self._get_dataset(),
                    bid,
                    ask,
                    self.time_step)
                current_q = mm_agent.q
                current_cash_surplus = mm_agent.cash_surplus
                old_bid, old_ask = bid, ask

            bid, ask = mm_agent.step(actual_data)

            self.full_history.append({
                'mm_agent_num': self.simulation_counter,
                'gamma': gamma,
                'k': k,
                't': t,
                'timestamp': self.last_timestamp,
                'datashape': datashape,
                'mid_price': mm_agent.estimate_mid_price(actual_data.price, actual_data.qty),
                'quantile_05_price': quantile_05_price,
                'quantile_95_price': quantile_95_price,
                'old_bid': old_bid,
                'old_ask': old_ask,
                'executed_bid': executed_bid,
                'executed_ask': executed_ask,
                'current_q': current_q,
                'current_cash_surplus': current_cash_surplus,
                'current_profit': current_profit
            })

            self.last_timestamp += self.time_step

        state = self._get_features()
        reward = self._get_reward()
        done = (self.simulation_counter % 10 == 0)

        self.simulation_counter += 1

        return state, reward, done

    def get_initial_state(self):
        return self._get_features()

    def _get_dataset(self):

        return self.dataset[self.dataset.timestamp <= self.last_timestamp].sort_values('timestamp')

    def _get_reward(self):
        current_memory = pd.DataFrame(self.full_history)
        current_memory = current_memory[current_memory.mm_agent_num == self.simulation_counter]

        current_memory = current_memory.sort_values('timestamp')
        current_memory.current_profit = current_memory.current_profit.ffill().fillna(0)

        reward = self.get_sharp_ratio(current_memory.current_profit)

        return reward

    def _get_features(self):
        """
            Тут для простоты пока сделал фичи размерности 5 просто агрегаты собрал,
            чтобы не заниматься пока работой с временными рядами,
            а доделать общий пайплайн, если будет время поправлю
        """

        actual_data = self._get_dataset()[-self.window:]

        avg_price = MarketMakerManager.estimate_mid_price(actual_data.price, actual_data.qty)
        std = MarketMakerManager.estimate_dispersion(actual_data.price, actual_data.qty)
        liquidity_sum = actual_data.qty.sum()
        liquidity_std = actual_data.qty.std()
        price_diff = actual_data.price.max() - actual_data.price.min()

        features = np.array([avg_price, std, liquidity_sum, liquidity_std, price_diff], dtype=np.float32)

        return features

    @staticmethod
    def get_sharp_ratio(profit):
        return (profit.mean() / (0.1 + profit.std()))

