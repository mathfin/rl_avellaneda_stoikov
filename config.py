import torch
from os import getenv
from dotenv import load_dotenv

load_dotenv('.env')

PSQL_URL = getenv('psql_url')

BINANCE_API_KEY=getenv('binance_api_key')
BINANCE_API_SECRET=getenv('binance_api_secret')

SEED = 777

T = 50000
TICKRATE = 25
SIMULATION_PERIODS = 36
WINDOW = 100

update_timestep = 12


device = torch.device('cpu')

if (torch.cuda.is_available()):
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
