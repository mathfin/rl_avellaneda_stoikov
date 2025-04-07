from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from config import PSQL_URL

engine = create_engine(PSQL_URL)
session_maker = sessionmaker(bind=engine)
