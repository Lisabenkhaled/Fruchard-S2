import datetime as dt

def datetime_to_years(date_value, ref_date):
    """
    Convertit une date (ou une liste de dates) en temps (float, années)
    """
    
    if date_value is None:
        return None

    # Si c'est une liste de dates
    if isinstance(date_value, list):
        return [datetime_to_years(d, ref_date) for d in date_value]

    # Si c'est une seule date
    if isinstance(date_value, dt.datetime) and isinstance(ref_date, dt.datetime):
        return (date_value - ref_date).days / 365.0
    
    if isinstance(date_value, dt.date) and isinstance(ref_date, dt.date):
        return (date_value - ref_date).days / 365.0

    # Si c'est déjà un float
    if isinstance(date_value, (int, float)):
        return float(date_value)

def years_to_datetime(t_value, ref_date):
    """
    Convertit un temps (en années) en datetime (ou liste de datetime)
    """
    if t_value is None:
        return None

    # Si liste de temps
    if isinstance(t_value, (list, tuple)):
        return [years_to_datetime(t, ref_date) for t in t_value]

    # Si c'est déjà une date 
    if isinstance(t_value, (dt.date, dt.datetime)):
        return t_value

    # Conversion float EN date
    if isinstance(t_value, (int, float)):
        days = float(t_value) * 365.0
        return ref_date + dt.timedelta(days=days)