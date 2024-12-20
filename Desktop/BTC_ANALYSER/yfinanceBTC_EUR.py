import yfinance as yf

# Télécharger les données
ticker = "BTC-EUR"  # Symbole pour Bitcoin en EUR
data = yf.download(ticker, start="2021-01-01", end="2024-12-20")

# Afficher les premières lignes
print(data.head())

# Sauvegarder dans un fichier CSV
data.to_csv("btc_eur_data.csv")
