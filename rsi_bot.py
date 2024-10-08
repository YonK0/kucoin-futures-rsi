#import kucoin
#from kucoin.client import Trade
import asyncio
import time
import ccxt.async_support as ccxt 
import time
import pandas as pd
from datetime import datetime, timedelta , timezone
import pandas as pd
import logging
import yaml
import sys


#=========================================CONFIG============================================================#

# Configure the logger
logging.basicConfig(
    filename='trading_bot.log',  # Log file path
    level=logging.INFO,  # Set the logging level
    format='%(asctime)s - %(levelname)s - %(message)s',  # Log format
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger()

# Function to load YAML config
def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def update_config(new_data):
    with open("out.yaml", 'w') as file:
        yaml.safe_dump(new_data, file)

# Load the configuration
config_out = load_config("out.yaml")

# Load the configuration
config = load_config("config.yaml")

# Access the values from the YAML file
kucoin_api_key = config['kucoin']['api_key']
kucoin_api_secret = config['kucoin']['api_secret']
kucoin_api_passphrase = config['kucoin']['passphrase']

exchange = ccxt.kucoinfutures({
        'apiKey': kucoin_api_key,
        'secret': kucoin_api_secret,
        'password': kucoin_api_passphrase,
        'enableRateLimit': True
    })
#=======================================CONFIG=============================================================#




# Virtual Wallet
virtual_wallet = {
    'balance': 200,    # Initial balance in USDT
    'toinvest': 0.1,   # 10%
    'assets': 200,       # just used for comparation with new balance
    'open_trades': {}  # List of open trades
}

#######################################################################
# Calculate RSI
def calculate_rsi(data, window=14, timeframe='1h'):
    # Convert the index to a DatetimeIndex for resampling
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
    data.set_index('timestamp', inplace=True)

    # Resample the data to 4-hour intervals if needed
    if timeframe:
        data = data.resample(timeframe).agg({'close': 'last'}).dropna()

    # Calculate price differences
    delta = data['close'].diff()
    
    # Calculate gains and losses
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    
    # Use exponentially weighted moving averages (EWMA) for gains and losses
    avg_gain = gain.ewm(com=window - 1, min_periods=window).mean()
    avg_loss = loss.ewm(com=window - 1, min_periods=window).mean()
    
    # Relative Strength (RS)
    rs = avg_gain / avg_loss
    
    # Relative Strength Index (RSI)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

# Check if a symbol already exists in open buy or sell trades
def symbol_in_open_trades(symbol):
    return symbol in virtual_wallet['open_trades']


# Fetch data asynchronously
async def fetch_data(exchange, symbol, timeframe):
    data = await exchange.fetch_ohlcv(symbol, timeframe)
    return pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

# Simulate a buy order
def virtual_buy(symbol, price):
    global virtual_wallet
    amount_to_invest = virtual_wallet['balance'] * virtual_wallet['toinvest'] # Invest 10% of the balance
    amount = amount_to_invest / price
    virtual_wallet['balance'] -= amount_to_invest

    target_price = price * 1.05  # 5% profit target
    stop_loss_price = price / 1.02 # 2% stop loss

    # Store the open trade
    virtual_wallet['open_trades'][symbol]= {
        'buy_price': price,
        'target_price': target_price,
        'stop_loss_price': stop_loss_price,
        'amount_to_inverst': amount_to_invest,
        'amount': amount,
        'trade_type': 'buy',  # Mark this as a buy trade
        #'timestamp': datetime.now()
    }
    config_out['kucoin']['buy'] += 1
    update_config(config_out)
    logger.info(f"[Buy] {amount:.4f} of {symbol} at {price} USDT. Target: {target_price} Stop loss: {stop_loss_price}")

# Simulate a sell order
def virtual_sell(symbol, price):
    global virtual_wallet
    amount_to_invest = virtual_wallet['balance'] * virtual_wallet['toinvest'] # Invest 10% of the balance
    amount = amount_to_invest / price
    virtual_wallet['balance'] -= amount_to_invest
    target_price = price /  1.05     # 5% TP
    stop_loss_price = price *  1.02  # 2% SL 

    # Store the open trade
    virtual_wallet['open_trades'][symbol]= {
        'sell_price': price,
        'target_price': target_price,
        'stop_loss_price': stop_loss_price,
        'amount_to_inverst': amount_to_invest,
        'amount': amount,
        'trade_type': 'sell',  # Mark this as a sell trade
        #'timestamp': datetime.now()
    }

    config_out['kucoin']['sell'] += 1
    update_config(config_out)
    logger.info(f"[Sell] {amount:.4f} of {symbol} at {price} USDT. Target: {target_price} Stop loss: {stop_loss_price}")

# Check for profit target or stop loss
async def check_open_trades():
    while True:
        global virtual_wallet
        for symbol_key, trade in virtual_wallet['open_trades'].copy().items():
                try : 
                    ticker = await exchange.fetch_ticker(symbol_key)
                    current_price = float(ticker['last'])
                except : 
                    logger.error(f"Error fetching ticker for {symbol_key}")
                    # virtual_wallet['balance'] += trade['amount_to_inverst']
                    # virtual_wallet['open_trades'].pop(symbol_key)
                    continue

                # For buy trades, target price is higher, stop loss is lower
                if trade['trade_type'] == 'buy':
                    
                    if current_price >= trade['target_price']:
                        logger.info(f"Buy [TP] for {symbol_key}. at {current_price} USDT.")

                        virtual_wallet['balance'] += trade['amount_to_inverst'] + (trade['amount_to_inverst'] * 0.05)
                        virtual_wallet['open_trades'].pop(symbol_key)
                        #New balance
                        new_balance = virtual_wallet['balance']- virtual_wallet['assets']
                        logger.info(f"W/L= {new_balance}")
                        #Logging
                        config_out['kucoin']['W/L'] = new_balance
                        config_out['kucoin']['w_buy'] += 1
                        update_config(config_out)

                    elif current_price <= trade['stop_loss_price']:
                        logger.info(f"Buy [SL] hit for {symbol_key}. at {current_price} USDT.")
                        
                        virtual_wallet['balance'] += trade['amount_to_inverst'] - (trade['amount_to_inverst'] * 0.02)
                        virtual_wallet['open_trades'].pop(symbol_key)
                        #New balance
                        new_balance = virtual_wallet['balance']- virtual_wallet['assets']
                        logger.info(f"W/L= {new_balance}")
                        #Logging
                        config_out['kucoin']['W/L'] = new_balance
                        config_out['kucoin']['SL'] += 1
                        update_config(config_out)


                # For sell trades, target price is lower, stop loss is higher
                elif trade['trade_type'] == 'sell':
                    if current_price <= trade['target_price']:
                        logger.info(f"Sell [TP] for {symbol_key}. at {current_price} USDT.")

                        virtual_wallet['balance'] += trade['amount_to_inverst'] + (trade['amount_to_inverst'] * 0.05)
                        virtual_wallet['open_trades'].pop(symbol_key)
                        #New balance
                        new_balance = virtual_wallet['balance']- virtual_wallet['assets']
                        logger.info(f"W/L= {new_balance}")
                        #Logging
                        config_out['kucoin']['W/L'] = new_balance
                        config_out['kucoin']['w_sell'] += 1
                        update_config(config_out)

                    elif current_price >= trade['stop_loss_price']:
                        logger.info(f"Sell [SL] hit for {symbol_key}. at {current_price} USDT.")
                        virtual_wallet['balance'] += trade['amount_to_inverst'] - (trade['amount_to_inverst'] * 0.02)
                        virtual_wallet['open_trades'].pop(symbol_key)
                        #New balance
                        new_balance = virtual_wallet['balance']- virtual_wallet['assets']
                        logger.info(f"W/L= {new_balance}")
                        #Logging
                        config_out['kucoin']['W/L'] = new_balance
                        config_out['kucoin']['SL'] += 1
                        update_config(config_out)

        #repeat every 300 ms
        await asyncio.sleep(0.3)

# Analyze data and trade based on RSI
async def analyze_and_trade():
    while True : 
        #print("im analysing \//")
        await exchange.load_markets()
        symbols = [market for market in exchange.markets if market.endswith('USDT')]
        for symbol in symbols :
            if not symbol_in_open_trades(symbol):
                
                try:
                    df_1h, df_4h = await asyncio.gather(
                        fetch_data(exchange, symbol, '1h'),
                        fetch_data(exchange, symbol, '4h')
                    )
                except Exception as e:
                    logger.error(f"Error fetching data for {symbol}: {e}")
                    continue
                rsi_1h = calculate_rsi(df_1h).iloc[-1]
                rsi_4h = calculate_rsi(df_4h).iloc[-1]
                current_price = df_1h['close'].iloc[-1]
                #print("hereee")
                # logger.info(f"{symbol} RSI__1H = {rsi_1h}")   
                # logger.info(f"{symbol} RSI__4H = {rsi_4h}")
                

                if rsi_1h > 70 and rsi_4h > 70 and virtual_wallet['balance'] > 0 :
                    #logger.info(f"{symbol}: Placing SELL order, RSI high on both 1h and 4h.")
                    virtual_sell(symbol, current_price)

                elif rsi_1h < 30 and rsi_4h < 30 and virtual_wallet['balance'] > 0 :
                    #logger.info(f"{symbol}: Placing BUY order, RSI low on both 1h and 4h.")
                    virtual_buy(symbol, current_price)
        
        #repeat every 300 ms
        await asyncio.sleep(0.5)

# Main function for running the bot
async def main():
    try:
        task1 = asyncio.create_task(check_open_trades())
        task2 = asyncio.create_task(analyze_and_trade())
        await asyncio.gather(task1, task2)
    finally:
        await exchange.close()  # Ensure that the exchange session is closed properly


if __name__ == "__main__":
    if sys.platform.startswith('windows'):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())  # This line is for Windows users
    logger.info(f"Hello new test started : {datetime.now()}")
    asyncio.run(main())

 