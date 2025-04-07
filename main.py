import pandas as pd
from sqlalchemy import func

from config import update_timestep, TICKRATE, T, SIMULATION_PERIODS, WINDOW, action_std_decay_rate, min_action_std
from data_base.engine import engine, session_maker
from data_base.models import Base, TradesData
from data_collecion.data_collector import fetch_and_save_trades
from pool import set_seed
from ppo_model.ppo_manager import PPO
from trading_env.custom_env import CustomEnvironment

set_seed()

def run_train(binance_data, update_timestep,
              T, TICKRATE, SIMULATION_PERIODS, WINDOW):
    last_timestamp = binance_data.timestamp.iloc[100]

    env = CustomEnvironment(initial_timestamp=last_timestamp,
                            dataset=binance_data,
                            T=T,
                            TICKRATE=TICKRATE,
                            WINDOW=WINDOW)
    state = env.get_initial_state()
    print_running_reward = 0

    for sim_num in range(1, SIMULATION_PERIODS + 1):

        action = ppo_agent.select_action(state)

        state, reward, done = env.step(action)

        print_running_reward += reward

        ppo_agent.buffer.rewards.append(reward)
        ppo_agent.buffer.is_terminals.append(done)

        if sim_num % update_timestep == 0:
            ppo_agent.update()

        if sim_num % 50 == 0:
            # print average reward till last episode
            print_avg_reward = print_running_reward / sim_num
            print_avg_reward = round(print_avg_reward, 2)

            print("Timestep : {} \t\t Average Reward : {}".format(sim_num, print_avg_reward))
            print(action)
        if sim_num % 100 == 0:
            ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)

    results_df = pd.DataFrame(env.full_history)
    return results_df



if __name__ == '__main__':
    Base.metadata.create_all(engine)

    session = session_maker()
    try:
        count = session.query(func.count(TradesData.price)).scalar()
        print(f"В таблице {count} записей.")
    finally:
        session.close()

    if count < 1_000_000:
        print("Записей недостаточно. Начинается заполнение таблицы.")
        fetch_and_save_trades(engine)
    else:
        print("Достаточное количество записей. Заполнение не требуется.")

    binance_data = pd.read_sql('select price, qty, timestamp from trades_data', con=engine)
    binance_data['time'] = pd.to_datetime(binance_data['timestamp'], unit='ms')
    binance_data = binance_data.sort_values('timestamp').reset_index(drop=True)

    ppo_agent = PPO(state_dim=5,
                    action_dim=2,
                    lr_actor=0.0003,
                    lr_critic=0.001,
                    gamma=0.99,
                    K_epochs=40,
                    eps_clip=0.1)

    results_df = run_train(
        binance_data,
        update_timestep,
        T=T,
        TICKRATE=TICKRATE,
        SIMULATION_PERIODS=SIMULATION_PERIODS,
        WINDOW=WINDOW
    )

    results_df.to_csv('./app/data/results.csv', index=False)
