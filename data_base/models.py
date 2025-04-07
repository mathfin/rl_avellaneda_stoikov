from sqlalchemy import DateTime, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import MetaData


class Base(DeclarativeBase):
    metadata = MetaData()

    created: Mapped[DateTime] = mapped_column(DateTime, server_default=func.now())
    updated: Mapped[DateTime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())

class TradesData(Base):
    __tablename__ = 'trades_data'

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    price: Mapped[float] = mapped_column(nullable=False)
    qty: Mapped[float] = mapped_column(nullable=False)
    timestamp: Mapped[float] = mapped_column(nullable=False)
