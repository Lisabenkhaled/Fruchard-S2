def ensure_sheet(wb, name):
        """Renvoie la feuille si elle existe, sinon la crée."""
        try:
            return wb.sheets[name]
        except Exception:
            return wb.sheets.add(name)