"""Programa para mapear as colunas dos dados."""

from sqlalchemy import VARCHAR
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Classe para declarar o tipo de base.

    Args:
        DeclarativeBase (_type_): Base declarativa do sql
    """


class ReferenceData(Base):
    """
    Mapea os dados de referencia da base de dados SQLite.

    Args:
        Base (_type_): Base declarativa do sql
    """

    __tablename__ = 'Reference_data'

    time: Mapped[float] = mapped_column(VARCHAR(), primary_key=True)
    E1: Mapped[float] = mapped_column(VARCHAR())
    T2: Mapped[float] = mapped_column(VARCHAR())
    T3: Mapped[float] = mapped_column(VARCHAR())
    T4: Mapped[float] = mapped_column(VARCHAR())
    T5: Mapped[float] = mapped_column(VARCHAR())
    T6: Mapped[float] = mapped_column(VARCHAR())
    T7: Mapped[float] = mapped_column(VARCHAR())
    T8: Mapped[float] = mapped_column(VARCHAR())
    T9: Mapped[float] = mapped_column(VARCHAR())
    Text: Mapped[float] = mapped_column(VARCHAR())
    DEMANDA: Mapped[float] = mapped_column(VARCHAR())
    m1: Mapped[float] = mapped_column(VARCHAR())
    m2: Mapped[float] = mapped_column(VARCHAR())
    m3: Mapped[float] = mapped_column(VARCHAR())


class MfBoi(Base):
    """
    Mapea os dados defeituosos da caldeira da base de dados SQLite.

    Args:
        Base (_type_): Base declarativa do sql
    """

    __tablename__ = 'Op_data_MF_boi'

    time: Mapped[float] = mapped_column(VARCHAR(), primary_key=True)
    E1: Mapped[float] = mapped_column(VARCHAR())
    T2: Mapped[float] = mapped_column(VARCHAR())
    T3: Mapped[float] = mapped_column(VARCHAR())
    T4: Mapped[float] = mapped_column(VARCHAR())
    T5: Mapped[float] = mapped_column(VARCHAR())
    T6: Mapped[float] = mapped_column(VARCHAR())
    T7: Mapped[float] = mapped_column(VARCHAR())
    T8: Mapped[float] = mapped_column(VARCHAR())
    T9: Mapped[float] = mapped_column(VARCHAR())
    Text: Mapped[float] = mapped_column(VARCHAR())
    DEMANDA: Mapped[float] = mapped_column(VARCHAR())
    m1: Mapped[float] = mapped_column(VARCHAR())
    m2: Mapped[float] = mapped_column(VARCHAR())
    m3: Mapped[float] = mapped_column(VARCHAR())


class MfHx(Base):
    """
    Mapea os dados defeituosos da caldeira da base de dados SQLite.

    Args:
        Base (_type_): Base declarativa do sql
    """

    __tablename__ = 'Op_data_MF_hx'

    time: Mapped[float] = mapped_column(VARCHAR(), primary_key=True)
    E1: Mapped[float] = mapped_column(VARCHAR())
    T2: Mapped[float] = mapped_column(VARCHAR())
    T3: Mapped[float] = mapped_column(VARCHAR())
    T4: Mapped[float] = mapped_column(VARCHAR())
    T5: Mapped[float] = mapped_column(VARCHAR())
    T6: Mapped[float] = mapped_column(VARCHAR())
    T7: Mapped[float] = mapped_column(VARCHAR())
    T8: Mapped[float] = mapped_column(VARCHAR())
    T9: Mapped[float] = mapped_column(VARCHAR())
    Text: Mapped[float] = mapped_column(VARCHAR())
    DEMANDA: Mapped[float] = mapped_column(VARCHAR())
    m1: Mapped[float] = mapped_column(VARCHAR())
    m2: Mapped[float] = mapped_column(VARCHAR())
    m3: Mapped[float] = mapped_column(VARCHAR())


class Mf1(Base):
    """Mapea os dados defeituosos da caldeira da base de dados SQLite.

    Args:
        Base (_type_): Base declarativa do sql
    """

    __tablename__ = 'Op_data_MF1_tank'

    time: Mapped[float] = mapped_column(VARCHAR(), primary_key=True)
    E1: Mapped[float] = mapped_column(VARCHAR())
    T2: Mapped[float] = mapped_column(VARCHAR())
    T3: Mapped[float] = mapped_column(VARCHAR())
    T4: Mapped[float] = mapped_column(VARCHAR())
    T5: Mapped[float] = mapped_column(VARCHAR())
    T6: Mapped[float] = mapped_column(VARCHAR())
    T7: Mapped[float] = mapped_column(VARCHAR())
    T8: Mapped[float] = mapped_column(VARCHAR())
    T9: Mapped[float] = mapped_column(VARCHAR())
    Text: Mapped[float] = mapped_column(VARCHAR())
    DEMANDA: Mapped[float] = mapped_column(VARCHAR())
    m1: Mapped[float] = mapped_column(VARCHAR())
    m2: Mapped[float] = mapped_column(VARCHAR())
    m3: Mapped[float] = mapped_column(VARCHAR())


class Mf2(Base):
    """Mapea os dados defeituosos da caldeira da base de dados SQLite.

    Args:
        Base (_type_): Base declarativa do sql
    """

    __tablename__ = 'Op_data_MF2_tank'

    time: Mapped[float] = mapped_column(VARCHAR(), primary_key=True)
    E1: Mapped[float] = mapped_column(VARCHAR())
    T2: Mapped[float] = mapped_column(VARCHAR())
    T3: Mapped[float] = mapped_column(VARCHAR())
    T4: Mapped[float] = mapped_column(VARCHAR())
    T5: Mapped[float] = mapped_column(VARCHAR())
    T6: Mapped[float] = mapped_column(VARCHAR())
    T7: Mapped[float] = mapped_column(VARCHAR())
    T8: Mapped[float] = mapped_column(VARCHAR())
    T9: Mapped[float] = mapped_column(VARCHAR())
    Text: Mapped[float] = mapped_column(VARCHAR())
    DEMANDA: Mapped[float] = mapped_column(VARCHAR())
    m1: Mapped[float] = mapped_column(VARCHAR())
    m2: Mapped[float] = mapped_column(VARCHAR())
    m3: Mapped[float] = mapped_column(VARCHAR())


class Mf3(Base):
    """Mapea os dados defeituosos da caldeira da base de dados SQLite.

    Args:
        Base (_type_): Base declarativa do sql
    """

    __tablename__ = 'Op_data_MF3_tank'

    time: Mapped[float] = mapped_column(VARCHAR(), primary_key=True)
    E1: Mapped[float] = mapped_column(VARCHAR())
    T2: Mapped[float] = mapped_column(VARCHAR())
    T3: Mapped[float] = mapped_column(VARCHAR())
    T4: Mapped[float] = mapped_column(VARCHAR())
    T5: Mapped[float] = mapped_column(VARCHAR())
    T6: Mapped[float] = mapped_column(VARCHAR())
    T7: Mapped[float] = mapped_column(VARCHAR())
    T8: Mapped[float] = mapped_column(VARCHAR())
    T9: Mapped[float] = mapped_column(VARCHAR())
    Text: Mapped[float] = mapped_column(VARCHAR())
    DEMANDA: Mapped[float] = mapped_column(VARCHAR())
    m1: Mapped[float] = mapped_column(VARCHAR())
    m2: Mapped[float] = mapped_column(VARCHAR())
    m3: Mapped[float] = mapped_column(VARCHAR())
