import logging
import requests
import psycopg2
import os
import time
from datetime import datetime

# Configure logging to print messages to the console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# PostgreSQL connection information
host = os.getenv('POSTGRES_HOST', 'ie7945.postgres.database.azure.com')
database = os.getenv('POSTGRES_DB', 'ie7945')
user = os.getenv('POSTGRES_USER', 'ie7945')
password = os.getenv('POSTGRES_PASSWORD', 'AgKpAmRePTUUZ9j')

# API URLs for different cryptocurrency exchanges
crypto_api_urls = {
    "kraken": "https://api.kraken.com/0/public/Ticker?pair=XBTUSD",
    "coingecko": "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd&include_24hr_vol=true",
    "coinpaprika": "https://api.coinpaprika.com/v1/tickers/btc-bitcoin"
}

# Function to fetch Kraken data
def fetch_kraken_data():
    try:
        logging.info("Fetching Kraken data...")
        response = requests.get(crypto_api_urls["kraken"])
        response.raise_for_status()
        data = response.json()
        logging.info(f"Fetched Kraken data: {data}")

        pair_key = list(data['result'].keys())[0]  # Dynamic pair key
        price = float(data['result'][pair_key]['c'][0])
        volume = float(data['result'][pair_key]['v'][1])

        return {"symbol": "BTC", "price": price, "volume": volume, "exchange": "Kraken", "timestamp": datetime.now()}
    except Exception as e:
        logging.error(f"Error fetching Kraken data: {e}")
        return None

# Function to fetch CoinGecko data
def fetch_coingecko_data():
    try:
        logging.info("Fetching CoinGecko data...")
        response = requests.get(crypto_api_urls["coingecko"])
        response.raise_for_status()
        data = response.json()
        logging.info(f"Fetched CoinGecko data: {data}")

        price = float(data['bitcoin']['usd'])
        volume = float(data['bitcoin']['usd_24h_vol'])  # 24-hour volume

        return {"symbol": "BTC", "price": price, "volume": volume, "exchange": "CoinGecko", "timestamp": datetime.now()}
    except Exception as e:
        logging.error(f"Error fetching CoinGecko data: {e}")
        return None

# Function to fetch Coinpaprika data
def fetch_coinpaprika_data():
    try:
        logging.info("Fetching Coinpaprika data...")
        response = requests.get(crypto_api_urls["coinpaprika"])
        response.raise_for_status()
        data = response.json()
        logging.info(f"Fetched Coinpaprika data: {data}")

        price = float(data['quotes']['USD']['price'])
        volume = float(data['quotes']['USD']['volume_24h'])

        return {"symbol": "BTC", "price": price, "volume": volume, "exchange": "Coinpaprika", "timestamp": datetime.now()}
    except Exception as e:
        logging.error(f"Error fetching Coinpaprika data: {e}")
        return None

# Function to insert data into the PostgreSQL database
def insert_into_db(record, conn, cursor):
    try:
        logging.info(f"Inserting record into database: {record}")
        insert_query = '''INSERT INTO crypto_table (symbol, price, volume, exchange, timestamp)
                         VALUES (%s, %s, %s, %s, %s)'''
        cursor.execute(insert_query, (record["symbol"], record["price"], record["volume"], record["exchange"], record["timestamp"]))
        conn.commit()
        logging.info(f"Inserted data for {record['symbol']} from {record['exchange']} at {record['timestamp']}")
    except Exception as e:
        logging.error(f"Error inserting data: {e}")

# Function to fetch and process cryptocurrency data
def fetch_crypto_data():
    logging.info('Fetching cryptocurrency data...')

    # Connect to PostgreSQL database
    try:
        logging.info("Connecting to PostgreSQL database...")
        conn = psycopg2.connect(host=host, database=database, user=user, password=password)
        cursor = conn.cursor()
        logging.info("Successfully connected to the database!")

        # Fetch and process data from each API (Kraken, CoinGecko, Coinpaprika)
        for api_function in [fetch_kraken_data, fetch_coingecko_data, fetch_coinpaprika_data]:
            data = api_function()
            if data:
                logging.info(f"Fetched data: {data}")
                insert_into_db(data, conn, cursor)

    except Exception as e:
        logging.error(f"Error connecting to database: {e}")
    finally:
        if 'cursor' in locals():
            cursor.close()
            logging.info("Cursor closed.")
        if 'conn' in locals():
            conn.close()
            logging.info("Connection closed.")

    logging.info('Crypto data fetching completed.')

# Main execution block with continuous loop
if __name__ == "__main__":
    try:
        logging.info("Starting cryptocurrency data fetching loop...")
        while True:
            fetch_crypto_data()
            logging.info("Waiting for 60 seconds before the next fetch...")
            time.sleep(60)  # Wait for 60 seconds before fetching again
    except KeyboardInterrupt:
        logging.info("Terminating the program on user interrupt (Ctrl+C).")
    except Exception as e:
        logging.error(f"Unexpected error occurred: {e}")
