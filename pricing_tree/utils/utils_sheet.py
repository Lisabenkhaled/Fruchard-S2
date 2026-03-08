import xlwings as xw

def ensure_sheet(wb: xw.Book, name: str) -> xw.Sheet:
    """Renvoie la feuille si elle existe, sinon la crée."""
    try:
        return wb.sheets[name]
    except Exception:
        return wb.sheets.add(name)