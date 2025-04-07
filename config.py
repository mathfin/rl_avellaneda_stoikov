import torch
from os import getenv
from dotenv import load_dotenv

load_dotenv('.env')

PSQL_URL = getenv('psql_url')

BINANCE_API_KEY=getenv('binance_api_key')
BINANCE_API_SECRET=getenv('binance_api_secret')

SEED = 777

# Раз в 50 секунд обновляются параметры gamma и k моделью ppo,
# а моделью avellaneda stoikov раз в 2 секунды выставляют новые ордера и смотрят нет ли выполненных

T = 50000
TICKRATE = 25

SIMULATION_PERIODS = 623 # (max(timestamp) - min(timestamp)) / T
WINDOW = 100

update_timestep = 12
action_std_decay_rate = 0.1
min_action_std = 0.05

device = torch.device('cpu')

if (torch.cuda.is_available()):
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
