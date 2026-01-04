import os
import json
import re
import time
import threading
import smtplib
from datetime import datetime, timedelta
from functools import wraps
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import numpy as np
import pytz
import requests
import yfinance as yf
import pandas as pd

from flask import Flask, request, jsonify, session, send_from_directory
from flask_cors import CORS

# -----------------------
# Constants & Config
# -----------------------
APP_SECRET = "stock_agent_secret_key_2026"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Fix the path to point to the frontend directory
FRONTEND_DIR = r"C:\Users\admin\OneDrive\Documents\stock-agent\frontend"
USERS_FILE = os.path.join(BASE_DIR, "users.json")

# Create users.json if it doesn't exist
if not os.path.exists(USERS_FILE):
    with open(USERS_FILE, 'w') as f:
        json.dump({}, f)


# Enhanced Avatar URLs
MALE_AVATARS = [
    'https://images.unsplash.com/photo-1472099645785-5658abf4ff4e?w=150&h=150&fit=crop&crop=face',
    'https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=150&h=150&fit=crop&crop=face',
    'https://images.unsplash.com/photo-1535713875002-d1d0cf377fde?w=150&h=150&fit=crop&crop=face',
    'https://images.unsplash.com/photo-1480429370139-e0132c086e2a?w=150&h=150&fit=crop&crop=face',
    'https://images.unsplash.com/photo-1560250097-0b93528c311a?w=150&h=150&fit=crop&crop=face',
    'https://images.unsplash.com/photo-1500648767791-00dcc994a43e?w=150&h=150&fit=crop&crop=face',
    'https://images.unsplash.com/photo-1544725176-7c40e5a71c5e?w=150&h=150&fit=crop&crop=face',
    'https://images.unsplash.com/photo-1519345182560-3f2917c472ef?w=150&h=150&fit=crop&crop=face'
]

FEMALE_AVATARS = [
    'https://images.unsplash.com/photo-1494790108755-2616b612b786?w=150&h=150&fit=crop&crop=face',
    'https://images.unsplash.com/photo-1438761681033-6461ffad8d80?w=150&h=150&fit=crop&crop=face',
    'https://images.unsplash.com/photo-1544005313-94ddf0286df2?w=150&h=150&fit=crop&crop=face',
    'https://images.unsplash.com/photo-1534751516642-a1af1ef26a56?w=150&h=150&fit=crop&crop=face',
    'https://images.unsplash.com/photo-1487412720507-e7ab37603c6f?w=150&h=150&fit=crop&crop=face',
    'https://images.unsplash.com/photo-1544725176-7c40e5a71c5e?w=150&h=150&fit=crop&crop=face',
    'https://images.unsplash.com/photo-1517841905240-472988babdf9?w=150&h=150&fit=crop&crop=face',
    'https://images.unsplash.com/photo-1531123897727-8f129e1688ce?w=150&h=150&fit=crop&crop=face'
]

# -----------------------
# Flask App Initialization
# -----------------------
app = Flask(__name__)
app.secret_key = APP_SECRET
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SESSION_COOKIE_SECURE'] = False  # Set to True in production with HTTPS
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=1)

CORS(app, supports_credentials=True, origins=['http://10.13.120.2:5000', 'http://localhost:5000'])

# -----------------------
# Authentication Decorator
# -----------------------
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "email" not in session:
            return jsonify({"success": False, "message": "Not authenticated"}), 401
        return f(*args, **kwargs)
    return decorated_function

# -----------------------
# Real-time price cache
# -----------------------
price_cache = {}
cache_lock = threading.Lock()
CACHE_DURATION = 10

# -----------------------
# Utility Functions
# -----------------------
def load_users():
    try:
        if not os.path.exists(USERS_FILE):
            return {}
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading users: {str(e)}")
        return {}

def save_users(users):
    try:
        temp_path = USERS_FILE + ".tmp"
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(users, f, indent=2, ensure_ascii=False)
        os.replace(temp_path, USERS_FILE)
        return True
    except Exception as e:
        print(f"Error saving users: {str(e)}")
        return False

def is_valid_email(email: str) -> bool:
    return bool(re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email))

def assign_random_avatar(gender):
    """Assign a random avatar based on gender"""
    import random
    valid_genders = ['Male', 'Female']
    if not gender or gender not in valid_genders:
        avatars = MALE_AVATARS
    else:
        avatars = MALE_AVATARS if gender.lower() == 'male' else FEMALE_AVATARS
    return random.choice(avatars)

def send_welcome_email(recipient_email, recipient_name):
    """Fixed email sending function"""
    try:
        # Create message
        msg = MIMEMultipart()
        msg['From'] = MAIL_USERNAME
        msg['To'] = recipient_email
        msg['Subject'] = "Welcome to AI Optimised Stock Guider 2025"

        # Create HTML content
        html_content = f"""
        <html>
            <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
                <div style="max-width: 600px; margin: 0 auto; padding: 20px; border: 1px solid #ddd; border-radius: 10px;">
                    <h1 style="color: #2c3e50; text-align: center;">Welcome to AI Optimised Stock Guider! 🎉</h1>
                    <h2 style="color: #3498db;">Hello {recipient_name}!</h2>
                    <p>Thank you for joining our AI-powered stock analysis platform. We're excited to have you on board!</p>
                    
                    <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin: 20px 0;">
                        <h3 style="color: #27ae60;">What you can do:</h3>
                        <ul>
                            <li>📈 Get real-time stock analysis</li>
                            <li>💡 Receive AI-powered investment recommendations</li>
                            <li>📊 View interactive charts and historical data</li>
                            <li>🔔 Track market movements in real-time</li>
                        </ul>
                    </div>
                    
                    <p>We're committed to helping you make informed investment decisions with our advanced AI algorithms.</p>
                    
                    <div style="text-align: center; margin: 30px 0;">
                        <p style="font-size: 14px; color: #7f8c8d;">
                            Developed by <strong>Karthik BK</strong><br>
                            Final Year PES Student<br>
                            📧 karthikabk2020@gmail.com
                        </p>
                    </div>
                    
                    <p style="text-align: center; font-size: 12px; color: #95a5a6;">
                        Thank you for using our app. We heartily welcome you to our family! 😊🤗🙏
                    </p>
                </div>
            </body>
        </html>
        """

        # Attach HTML content
        msg.attach(MIMEText(html_content, 'html'))

        # Send email
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(MAIL_USERNAME, MAIL_PASSWORD)
            server.send_message(msg)
        
        print(f"Welcome email sent successfully to {recipient_email}")
        return True
        
    except Exception as e:
        print(f"Error sending email to {recipient_email}: {str(e)}")
        return False

def get_cached_price(symbol):
    with cache_lock:
        if symbol in price_cache:
            cached_data = price_cache[symbol]
            if time.time() - cached_data['timestamp'] < CACHE_DURATION:
                return cached_data['data']
    return None

def set_cached_price(symbol, data):
    with cache_lock:
        price_cache[symbol] = {'data': data, 'timestamp': time.time()}

def get_real_time_price(symbol):
    """Get real-time price data from Yahoo Finance"""
    try:
        # Check cache first
        cached = get_cached_price(symbol)
        if cached:
            return cached

        # Get real data from Yahoo Finance
        ticker = yf.Ticker(symbol)
        
        # Get current info
        info = ticker.info
        current_price = info.get('currentPrice') or info.get('regularMarketPrice')
        previous_close = info.get('previousClose')
        
        if current_price is None or previous_close is None:
            # Fallback to historical data
            hist = ticker.history(period="2d")
            if len(hist) >= 2:
                current_price = hist['Close'].iloc[-1]
                previous_close = hist['Close'].iloc[-2]
            else:
                return None

        # Calculate percentage change
        pct_change = ((current_price - previous_close) / previous_close) * 100
        
        result = {
            "price": round(float(current_price), 2),
            "prev_close": round(float(previous_close), 2),
            "pctChange": round(float(pct_change), 2)
        }
        
        # Cache the result
        set_cached_price(symbol, result)
        return result
        
    except Exception as e:
        print(f"Error getting real-time price for {symbol}: {str(e)}")
        return None

def get_real_tickers():
    """Get real stock data for major stocks"""
    symbols = [
        "AAPL", "TSLA", "MSFT", "GOOGL", "AMZN", 
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS",
        "BAJFINANCE.NS", "WIPRO.NS", "SBIN.NS"
    ]
    
    result = []
    
    for symbol in symbols:
        price_data = get_real_time_price(symbol)
        if price_data:
            result.append({
                "symbol": symbol,
                "price": price_data["price"],
                "pct": price_data["pctChange"]
            })
        else:
            # Fallback with some variation
            result.append({
                "symbol": symbol,
                "price": 100 + len(result) * 50,
                "pct": round((len(result) - 4) * 1.5, 2)
            })
    
    # Sort by performance
    gainers = sorted([s for s in result if s["pct"] >= 0], key=lambda x: x["pct"], reverse=True)
    fallers = sorted([s for s in result if s["pct"] < 0], key=lambda x: x["pct"])
    
    return {
        "all": result,
        "gainers": gainers[:5],
        "fallers": fallers[:5]
    }

def get_real_chart_data(symbol):
    """Get real historical data for charts"""
    try:
        ticker = yf.Ticker(symbol)
        
        # Get intraday data for today
        hist = ticker.history(period="1d", interval="5m")
        
        if hist.empty:
            # Fallback to daily data
            hist = ticker.history(period="5d", interval="1d")
        
        if hist.empty:
            return generate_fallback_chart_data(symbol)
        
        # Prepare chart data
        prices = {}
        volumes = {}
        
        for timestamp, row in hist.iterrows():
            time_str = timestamp.strftime("%H:%M")
            prices[time_str] = round(float(row['Close']), 2)
            volumes[time_str] = int(row['Volume']) if 'Volume' in row else 1000000
        
        time_points = list(prices.keys())
        
        return {
            "prices": prices,
            "volume": volumes,
            "time_points": time_points,
            "metrics": {
                "current": list(prices.values())[-1] if prices else 0,
                "high": max(prices.values()) if prices else 0,
                "low": min(prices.values()) if prices else 0,
                "open": list(prices.values())[0] if prices else 0,
                "prev_close": get_real_time_price(symbol)["prev_close"] if get_real_time_price(symbol) else 0,
                "change": (list(prices.values())[-1] - list(prices.values())[0]) if prices else 0,
                "change_pct": ((list(prices.values())[-1] - list(prices.values())[0]) / list(prices.values())[0] * 100) if prices and list(prices.values())[0] != 0 else 0
            }
        }
        
    except Exception as e:
        print(f"Error getting real chart data for {symbol}: {str(e)}")
        return generate_fallback_chart_data(symbol)

def generate_fallback_chart_data(symbol):
    """Generate fallback chart data when real data is unavailable"""
    import random
    base_price = get_real_time_price(symbol)
    base_price = base_price["price"] if base_price else 100
    
    time_points = ["09:30", "10:15", "11:00", "11:45", "12:30", "13:15", "14:00", "14:45", "15:30"]
    prices = {}
    volumes = {}
    
    current_price = base_price
    for time_str in time_points:
        # Simulate realistic price movement
        change_percent = random.uniform(-0.5, 0.5)
        current_price = current_price * (1 + change_percent / 100)
        prices[time_str] = round(current_price, 2)
        volumes[time_str] = random.randint(500000, 2000000)
    
    return {
        "prices": prices,
        "volume": volumes,
        "time_points": time_points,
        "metrics": {
            "current": current_price,
            "high": max(prices.values()),
            "low": min(prices.values()),
            "open": list(prices.values())[0],
            "prev_close": base_price * 0.99,
            "change": current_price - list(prices.values())[0],
            "change_pct": ((current_price - list(prices.values())[0]) / list(prices.values())[0]) * 100
        }
    }

def analyze_stock(symbol: str, planned_amount=1000):
    """Analyze stock and provide CLEAR investment recommendations with proper volatility and market status"""
    try:
        tk = yf.Ticker(symbol)
        
        # Get 60 days of historical data for proper comparison
        df = tk.history(period="60d", interval="1d")
        
        # Get current price data
        current_price_data = get_real_time_price(symbol)
        if not current_price_data:
            return {
                "symbol": symbol,
                "recommendation": "HOLD",
                "confidence": "Low",
                "market_sentiment": "Neutral",
                "advice": "Insufficient data for analysis",
                "score": 0,
                "planned_amount": planned_amount
            }
        
        latest = current_price_data["price"]
        daily_pct = current_price_data["pctChange"]
        
        if df.empty or len(df) < 30:
            return {
                "symbol": symbol,
                "recommendation": "HOLD",
                "confidence": "Low",
                "market_sentiment": "Neutral", 
                "advice": "Insufficient historical data",
                "score": 0,
                "planned_amount": planned_amount
            }
        
        # Calculate key technical indicators
        close_prices = df['Close']
        high_prices = df['High'] 
        low_prices = df['Low']
        volumes = df['Volume']
        
        # PROPER VOLATILITY CALCULATION
        daily_returns = close_prices.pct_change().dropna()
        if len(daily_returns) > 1:
            # Annualized volatility (standard deviation of daily returns * sqrt(252))
            volatility = np.std(daily_returns) * np.sqrt(252) * 100  # As percentage
        else:
            volatility = 25.0  # Default moderate volatility
        
        # Moving averages for trend analysis
        avg10 = float(close_prices.tail(10).mean())
        avg30 = float(close_prices.tail(30).mean())
        avg50 = float(close_prices.tail(50).mean()) if len(df) >= 50 else avg30
        
        # Recent performance (last 5, 10, 20 days)
        price_5d_ago = float(close_prices.iloc[-5]) if len(df) >= 5 else latest
        price_10d_ago = float(close_prices.iloc[-10]) if len(df) >= 10 else latest
        price_20d_ago = float(close_prices.iloc[-20]) if len(df) >= 20 else latest
        
        change_5d = ((latest - price_5d_ago) / price_5d_ago) * 100
        change_10d = ((latest - price_10d_ago) / price_10d_ago) * 100
        change_20d = ((latest - price_20d_ago) / price_20d_ago) * 100
        
        # Support and resistance levels
        recent_high_20d = float(high_prices.tail(20).max())
        recent_low_20d = float(low_prices.tail(20).min())
        
        # Volume analysis
        avg_volume_20d = float(volumes.tail(20).mean())
        current_volume = float(volumes.iloc[-1]) if len(volumes) > 0 else avg_volume_20d
        volume_ratio = current_volume / avg_volume_20d if avg_volume_20d > 0 else 1
        
        # MARKET STATUS CHECK
        market_status = "Open"
        market_message = ""
        
        # Check if market is likely closed (no recent price movement or unusual conditions)
        current_time = datetime.now()
        market_hours_message = ""
        
        # For US stocks (9:30 AM - 4:00 PM EST)
        est_time = current_time.astimezone(pytz.timezone('US/Eastern'))
        is_weekend = est_time.weekday() >= 5  # Saturday or Sunday
        is_market_hours = not is_weekend and (9 <= est_time.hour < 16)
        
        if not is_market_hours:
            market_status = "Closed"
            market_message = "🔒 Market is closed for today. Analysis based on latest available data."
        
        # DECISIVE ANALYSIS LOGIC - CLEAR BUY/SELL SIGNALS
        reasons = []
        buy_signals = 0
        sell_signals = 0
        total_signals = 0
        
        # SIGNAL 1: Today's Performance (Strong Weight)
        total_signals += 1
        if daily_pct > 2.5:
            buy_signals += 1
            reasons.append(f"🚀 STRONG UP: Stock up {daily_pct:.1f}% today (bullish momentum)")
        elif daily_pct > 1.0:
            buy_signals += 0.5
            reasons.append(f"📈 UP: Stock up {daily_pct:.1f}% today (positive)")
        elif daily_pct < -2.5:
            sell_signals += 1
            reasons.append(f"💥 STRONG DOWN: Stock down {abs(daily_pct):.1f}% today (bearish pressure)")
        elif daily_pct < -1.0:
            sell_signals += 0.5
            reasons.append(f"📉 DOWN: Stock down {abs(daily_pct):.1f}% today (negative)")
        else:
            reasons.append(f"➡️ FLAT: Small change of {daily_pct:.1f}% today")
        
        # SIGNAL 2: Short-term Trend (5-day performance)
        total_signals += 1
        if change_5d > 4:
            buy_signals += 1
            reasons.append(f"📊 STRONG 5-day trend: Up {change_5d:.1f}% this week")
        elif change_5d > 2:
            buy_signals += 0.5
            reasons.append(f"📊 Positive 5-day trend: Up {change_5d:.1f}%")
        elif change_5d < -4:
            sell_signals += 1
            reasons.append(f"📊 WEAK 5-day trend: Down {abs(change_5d):.1f}% this week")
        elif change_5d < -2:
            sell_signals += 0.5
            reasons.append(f"📊 Negative 5-day trend: Down {abs(change_5d):.1f}%")
        
        # SIGNAL 3: Medium-term Trend (10-20 day performance)
        total_signals += 1
        avg_medium_term = (change_10d + change_20d) / 2
        if avg_medium_term > 6:
            buy_signals += 1
            reasons.append(f"📈 STRONG medium-term: Up {avg_medium_term:.1f}% avg over 10-20 days")
        elif avg_medium_term > 3:
            buy_signals += 0.5
            reasons.append(f"📈 Good medium-term: Up {avg_medium_term:.1f}% avg")
        elif avg_medium_term < -6:
            sell_signals += 1
            reasons.append(f"📉 WEAK medium-term: Down {abs(avg_medium_term):.1f}% avg over 10-20 days")
        elif avg_medium_term < -3:
            sell_signals += 0.5
            reasons.append(f"📉 Poor medium-term: Down {abs(avg_medium_term):.1f}% avg")
        
        # SIGNAL 4: Moving Average Position
        total_signals += 1
        above_moving_averages = 0
        if latest > avg10: above_moving_averages += 1
        if latest > avg30: above_moving_averages += 1
        if latest > avg50: above_moving_averages += 1
        
        if above_moving_averages == 3:
            buy_signals += 1
            reasons.append(f"✅ Trading ABOVE all key moving averages (bullish setup)")
        elif above_moving_averages >= 2:
            buy_signals += 0.5
            reasons.append(f"✅ Above most moving averages ({above_moving_averages}/3)")
        elif above_moving_averages == 0:
            sell_signals += 1
            reasons.append(f"❌ Trading BELOW all key moving averages (bearish setup)")
        else:
            sell_signals += 0.5
            reasons.append(f"❌ Below most moving averages ({above_moving_averages}/3)")
        
        # SIGNAL 5: Support/Resistance Levels
        total_signals += 1
        distance_to_high = ((recent_high_20d - latest) / latest) * 100
        distance_to_low = ((latest - recent_low_20d) / latest) * 100
        
        if distance_to_low < 3:
            buy_signals += 1
            reasons.append(f"🛡️ Near STRONG SUPPORT: Only {distance_to_low:.1f}% above 20-day low")
        elif distance_to_low < 8:
            buy_signals += 0.5
            reasons.append(f"🛡️ Decent support: {distance_to_low:.1f}% above 20-day low")
        elif distance_to_high < 3:
            sell_signals += 1
            reasons.append(f"🛑 Near STRONG RESISTANCE: Only {distance_to_high:.1f}% below 20-day high")
        elif distance_to_high < 8:
            sell_signals += 0.5
            reasons.append(f"🛑 Near resistance: {distance_to_high:.1f}% below 20-day high")
        
        # SIGNAL 6: Volume Confirmation
        total_signals += 1
        if volume_ratio > 1.5 and daily_pct > 1:
            buy_signals += 1
            reasons.append(f"💰 HIGH volume confirmation: {volume_ratio:.1f}x average volume on up day")
        elif volume_ratio > 1.2 and daily_pct > 0:
            buy_signals += 0.5
            reasons.append(f"💰 Good volume: {volume_ratio:.1f}x average on positive day")
        elif volume_ratio > 1.5 and daily_pct < -1:
            sell_signals += 1
            reasons.append(f"💰 HIGH volume selling: {volume_ratio:.1f}x average volume on down day")
        elif volume_ratio > 1.2 and daily_pct < 0:
            sell_signals += 0.5
            reasons.append(f"💰 Selling volume: {volume_ratio:.1f}x average on negative day")
        
        # CALCULATE FINAL DECISION
        buy_ratio = buy_signals / total_signals if total_signals > 0 else 0
        sell_ratio = sell_signals / total_signals if total_signals > 0 else 0
        
        # CLEAR DECISION MAKING - NO WISHY-WASHY HOLD
        if buy_ratio >= 0.6 and buy_ratio > sell_ratio:
            if buy_ratio >= 0.8:
                recommendation = "STRONG BUY"
                confidence = "Very High"
                advice = "EXCELLENT opportunity - multiple strong bullish signals aligned"
                risk_level = "Low"
                sentiment = "Very Bullish"
            else:
                recommendation = "BUY"
                confidence = "High"
                advice = "Good buying opportunity - clear bullish signals present"
                risk_level = "Low-Medium"
                sentiment = "Bullish"
                
        elif sell_ratio >= 0.6 and sell_ratio > buy_ratio:
            if sell_ratio >= 0.8:
                recommendation = "STRONG SELL"
                confidence = "Very High"
                advice = "AVOID/REDUCE - multiple strong bearish signals aligned"
                risk_level = "High"
                sentiment = "Very Bearish"
            else:
                recommendation = "SELL"
                confidence = "High"
                advice = "Consider selling - clear bearish signals present"
                risk_level = "Medium-High"
                sentiment = "Bearish"
                
        else:
            # Only HOLD if signals are truly mixed (close to 50/50)
            if abs(buy_ratio - sell_ratio) < 0.2:
                recommendation = "HOLD"
                confidence = "Medium"
                advice = "Mixed signals - wait for clearer direction"
                risk_level = "Medium"
                sentiment = "Neutral"
            elif buy_ratio > sell_ratio:
                recommendation = "BUY"
                confidence = "Medium"
                advice = "Slight edge to buyers - consider small position"
                risk_level = "Medium"
                sentiment = "Mildly Bullish"
            else:
                recommendation = "SELL"
                confidence = "Medium"
                advice = "Slight edge to sellers - consider reducing exposure"
                risk_level = "Medium"
                sentiment = "Mildly Bearish"
        
        # Today's specific sentiment with clear categories
        if daily_pct > 5:
            today_sentiment = "Extremely Bullish"
            sentiment_emoji = "🚀🔥"
        elif daily_pct > 2.5:
            today_sentiment = "Very Bullish"
            sentiment_emoji = "🚀"
        elif daily_pct > 1:
            today_sentiment = "Bullish" 
            sentiment_emoji = "📈"
        elif daily_pct > -1:
            today_sentiment = "Neutral"
            sentiment_emoji = "➡️"
        elif daily_pct > -2.5:
            today_sentiment = "Bearish"
            sentiment_emoji = "📉"
        elif daily_pct > -5:
            today_sentiment = "Very Bearish"
            sentiment_emoji = "💥"
        else:
            today_sentiment = "Extremely Bearish"
            sentiment_emoji = "📉🔥"
        
        # Volatility classification
        if volatility > 40:
            volatility_level = "Very High"
            volatility_color = "danger"
        elif volatility > 25:
            volatility_level = "High"
            volatility_color = "warning"
        elif volatility > 15:
            volatility_level = "Moderate"
            volatility_color = "info"
        else:
            volatility_level = "Low"
            volatility_color = "success"
        
        # Investment allocation based on recommendation strength
        if recommendation == "STRONG BUY":
            recommended_amount = planned_amount * 0.9
            expected_return = 0.18
        elif recommendation == "BUY":
            recommended_amount = planned_amount * 0.7
            expected_return = 0.12
        elif recommendation == "HOLD":
            recommended_amount = planned_amount * 0.3
            expected_return = 0.05
        elif recommendation == "SELL":
            recommended_amount = 0
            expected_return = -0.10
        else:  # STRONG SELL
            recommended_amount = 0
            expected_return = -0.20

        # Calculate profits
        monthly_profit = recommended_amount * (expected_return / 12)
        quarterly_profit = recommended_amount * (expected_return / 4)
        annual_profit = recommended_amount * expected_return

        return {
            "symbol": symbol,
            "recommendation": recommendation,
            "confidence": confidence,
            "market_sentiment": sentiment,
            "today_sentiment": today_sentiment,
            "today_sentiment_emoji": sentiment_emoji,
            "market_status": market_status,
            "market_message": market_message,
            "advice": advice,
            "score": round((buy_ratio - sell_ratio) * 100, 1),  # Net score from -100 to +100
            "buy_signals": f"{buy_signals}/{total_signals}",
            "sell_signals": f"{sell_signals}/{total_signals}",
            "details": {
                "current_price": round(float(latest), 2),
                "risk_level": risk_level,
                "volatility": round(float(volatility), 1),
                "volatility_level": volatility_level,
                "volatility_color": volatility_color,
                "daily_change": round(float(daily_pct), 2),
                "5_day_change": round(float(change_5d), 2),
                "10_day_change": round(float(change_10d), 2),
                "20_day_change": round(float(change_20d), 2),
                "volume_ratio": round(volume_ratio, 2),
                "above_moving_averages": f"{above_moving_averages}/3",
                "distance_to_support": f"{distance_to_low:.1f}%",
                "distance_to_resistance": f"{distance_to_high:.1f}%",
                "reasons": reasons
            },
            "planned_amount": float(planned_amount),
            "investment_recommendation": {
                "recommended_amount": round(float(recommended_amount), 2),
                "risk_text": f"{risk_level} risk",
                "allocation_percentage": round((recommended_amount / planned_amount * 100) if planned_amount > 0 else 0, 1)
            },
            "expected_profit": {
                "monthly": round(float(monthly_profit), 2),
                "quarterly": round(float(quarterly_profit), 2),
                "annual": round(float(annual_profit), 2),
                "monthly_return_pct": round((expected_return / 12 * 100), 2),
                "quarterly_return_pct": round((expected_return / 4 * 100), 2),
                "annual_return_pct": round((expected_return * 100), 2),
                "calculation_basis": "Based on current vs historical price analysis"
            }
        }
        
    except Exception as e:
        print(f"Analysis error for {symbol}: {str(e)}")
        # Simple fallback based only on today's performance
        last_price_data = get_real_time_price(symbol)
        if last_price_data:
            daily_pct = last_price_data["pctChange"]
            
            if daily_pct > 2:
                recommendation = "BUY"
                sentiment = "Bullish"
                today_sentiment = "Bullish"
            elif daily_pct < -2:
                recommendation = "SELL" 
                sentiment = "Bearish"
                today_sentiment = "Bearish"
            else:
                recommendation = "HOLD"
                sentiment = "Neutral"
                today_sentiment = "Neutral"
        
            return {
                "symbol": symbol,
                "recommendation": recommendation,
                "confidence": "Medium",
                "market_sentiment": sentiment,
                "today_sentiment": today_sentiment,
                "today_sentiment_emoji": "📊",
                "market_status": "Unknown",
                "market_message": "Limited data available",
                "score": round(daily_pct, 2),
                "planned_amount": planned_amount,
                "advice": f"Based on today's {daily_pct:.1f}% movement",
                "details": {
                    "current_price": last_price_data["price"],
                    "daily_change": round(daily_pct, 2),
                    "volatility": 25.0,
                    "volatility_level": "Moderate",
                    "reasons": [f"Today's performance: {daily_pct:.1f}%"]
                }
            }
    
        return {
            "symbol": symbol,
            "recommendation": "HOLD",
            "confidence": "Low",
            "market_sentiment": "Neutral",
            "today_sentiment": "Neutral",
            "today_sentiment_emoji": "📊",
            "market_status": "Unknown",
            "market_message": "No market data available",
            "score": 0,
            "planned_amount": planned_amount,
            "advice": "Insufficient data for analysis"
        }


def validate_stock_symbol(symbol):
    """Validate stock symbol format"""
    if not symbol:
        return False
    
    # Basic validation - allow letters, numbers, and .NS for Indian stocks
    if not re.match(r'^[A-Z0-9.-]+$', symbol.upper()):
        return False
    
    # Common validations
    if len(symbol) > 20:  # Reasonable length limit
        return False
    
    return True

# -----------------------
# Routes
# -----------------------

@app.route("/")
def serve_frontend():
    """Serve the main frontend page"""
    try:
        # Check if frontend directory exists
        if not os.path.exists(FRONTEND_DIR):
            return jsonify({
                "error": "Frontend directory not found",
                "expected_path": FRONTEND_DIR,
                "current_directory": BASE_DIR
            }), 404
        
        index_path = os.path.join(FRONTEND_DIR, 'index.html')
        if not os.path.exists(index_path):
            return jsonify({
                "error": "index.html not found",
                "expected_path": index_path
            }), 404
            
        return send_from_directory(FRONTEND_DIR, 'index.html')
    except Exception as e:
        return jsonify({
            "error": f"Server error: {str(e)}",
            "frontend_dir": FRONTEND_DIR,
            "base_dir": BASE_DIR
        }), 500

@app.route("/<path:path>")
def serve_static(path):
    """Serve static files from frontend directory"""
    try:
        return send_from_directory(FRONTEND_DIR, path)
    except Exception as e:
        return jsonify({"error": f"File not found: {path}"}), 404

@app.route("/api/status")
def api_status():
    return jsonify({"message": "Stock Agent API is running", "status": "active"})

@app.route("/api/signup", methods=["POST"])
def signup():
    try:
        data = request.get_json()
        email = data.get("email", "").strip().lower()  # Normalize email to lowercase
        password = data.get("password")
        name = data.get("name", "")
        gender = data.get("gender", "")
        country = data.get("country", "")
        amount = data.get("amount", 0)
        
        print(f"Signup attempt for email: {email}")
        
        if not email or not password:
            return jsonify({"success": False, "message": "Email and password are required"})
            
        if not is_valid_email(email):
            return jsonify({"success": False, "message": "Invalid email format"})
            
        users = load_users()
        print(f"Current users in database: {list(users.keys())}")
        
        # Check if email already exists (case-insensitive)
        existing_emails = [e.lower() for e in users.keys()]
        if email.lower() in existing_emails:
            print(f"Email {email} already exists in database")
            return jsonify({"success": False, "message": "Email already exists"})
        
        # Assign random avatar based on gender
        avatar = assign_random_avatar(gender)
            
        # Create new user
        users[email] = {
            "password": password, 
            "name": name,
            "gender": gender,
            "country": country,
            "amount": float(amount) if amount else 0,
            "avatar": avatar,
            "created_at": datetime.now().isoformat()  # Add timestamp for tracking
        }
        
        if save_users(users):
            # Verify the user was added
            users_after_signup = load_users()
            if email in users_after_signup:
                print(f"Successfully created account for: {email}")
                send_welcome_email(email, name)
                
                # Auto-login after signup
                session.permanent = True
                session["email"] = email
                session["name"] = name
                session["country"] = country
                session["gender"] = gender
                session["amount"] = float(amount) if amount else 0
                session["avatar"] = avatar
                
                return jsonify({
                    "success": True, 
                    "message": "Signup successful",
                    "user": {
                        "name": name,
                        "email": email,
                        "gender": gender,
                        "country": country,
                        "amount": float(amount) if amount else 0,
                        "avatar": avatar
                    }
                })
            else:
                print(f"ERROR: User {email} not found after signup")
                return jsonify({"success": False, "message": "Failed to create account"})
        else:
            print(f"Failed to save user data for: {email}")
            return jsonify({"success": False, "message": "Failed to save user data"})
            
    except Exception as e:
        print(f"Signup error for {data.get('email', 'unknown')}: {str(e)}")
        return jsonify({"success": False, "message": "Server error during signup"})

@app.route("/api/debug-users", methods=["GET"])
def debug_users():
    """Debug endpoint to check current users (remove in production)"""
    try:
        users = load_users()
        return jsonify({
            "success": True,
            "user_count": len(users),
            "users": list(users.keys()),
            "file_path": USERS_FILE,
            "file_exists": os.path.exists(USERS_FILE)
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})
@app.route("/api/login", methods=["POST"])
def login():
    try:
        data = request.get_json()
        email = data.get("email")
        password = data.get("password")
        
        if not email or not password:
            return jsonify({"success": False, "message": "Email and password are required"})
            
        users = load_users()
        if email not in users or users[email]["password"] != password:
            return jsonify({"success": False, "message": "Invalid email or password"})
            
        user_data = users[email]
        
        # Ensure avatar exists, assign one if missing
        if "avatar" not in user_data or not user_data["avatar"]:
            user_data["avatar"] = assign_random_avatar(user_data.get("gender", ""))
            users[email] = user_data
            save_users(users)
            
        # Set session data
        session.permanent = True
        session["email"] = email
        session["name"] = user_data["name"]
        session["country"] = user_data.get("country", "")
        session["gender"] = user_data.get("gender", "")
        session["amount"] = user_data.get("amount", 0)
        session["avatar"] = user_data.get("avatar", "")
        
        print(f"User logged in: {email}, session: {dict(session)}")
        
        return jsonify({
            "success": True, 
            "message": "Login successful",
            "name": user_data["name"],
            "email": email,
            "country": user_data.get("country", ""),
            "gender": user_data.get("gender", ""),
            "amount": user_data.get("amount", 0),
            "avatar": user_data.get("avatar", "")
        })
        
    except Exception as e:
        print(f"Login error: {str(e)}")
        return jsonify({"success": False, "message": "Server error during login"})


@app.route("/api/logout", methods=["POST"])
def logout():
    email = session.get("email", "Unknown")
    session.clear()
    print(f"User logged out: {email}")
    return jsonify({"success": True, "message": "Logout successful"})

@app.route("/api/delete-account", methods=["POST"])
@login_required
def delete_account():
    """Delete user account - COMPLETELY remove user data"""
    try:
        email = session["email"]
        print(f"Attempting to delete account for: {email}")
        
        users = load_users()
        print(f"Current users before deletion: {list(users.keys())}")
        
        if email not in users:
            print(f"User {email} not found in users file")
            return jsonify({"success": False, "message": "User not found"})
        
        # COMPLETELY delete user from the database
        del users[email]
        print(f"User {email} removed from memory")
        
        # Save the updated users data (without the deleted user)
        if save_users(users):
            # Verify the user was actually removed
            users_after_deletion = load_users()
            print(f"Users after deletion: {list(users_after_deletion.keys())}")
            
            if email in users_after_deletion:
                print(f"ERROR: User {email} still exists after deletion!")
                return jsonify({"success": False, "message": "Failed to completely delete account"})
            
            # Clear session
            session.clear()
            print(f"Account successfully deleted for: {email}")
            
            return jsonify({
                "success": True, 
                "message": "Account deleted successfully"
            })
        else:
            print(f"Failed to save users file after deleting {email}")
            return jsonify({"success": False, "message": "Failed to delete account"})
            
    except Exception as e:
        print(f"Delete account error for {session.get('email', 'unknown')}: {str(e)}")
        return jsonify({"success": False, "message": f"Server error: {str(e)}"})

@app.route("/api/update-investment", methods=["POST"])
@login_required
def update_investment():
    """Update user investment amount"""
    try:
        data = request.get_json()
        new_amount = data.get("amount")
        
        if not new_amount or float(new_amount) <= 0:
            return jsonify({"success": False, "message": "Invalid amount"})
            
        email = session["email"]
        users = load_users()
        
        if email not in users:
            return jsonify({"success": False, "message": "User not found"})
            
        # Convert to float and update
        new_amount_float = float(new_amount)
        users[email]["amount"] = new_amount_float
        session["amount"] = new_amount_float
        
        if save_users(users):
            return jsonify({
                "success": True, 
                "message": "Investment plan updated successfully",
                "new_amount": new_amount_float
            })
        else:
            return jsonify({"success": False, "message": "Failed to save user data"})
            
    except Exception as e:
        print(f"Update investment error: {str(e)}")
        return jsonify({"success": False, "message": f"Server error during update: {str(e)}"})

@app.route("/api/about", methods=["GET"])
def about_info():
    """Get detailed about information"""
    about_data = {
        "developer": {
            "name": "Karthik BK",
            "education": "Final Year PES Student",
            "email": "karthikabk2020@gmail.com",
            "role": "Full Stack Developer & AI Enthusiast"
        },
        "app": {
            "name": "AI Optimised Stock Guider 2025",
            "version": "2.0.0",
            "description": "Advanced AI-powered stock analysis and investment recommendation platform",
            "features": [
                "Real-time stock data analysis",
                "AI-powered investment recommendations",
                "Interactive charts and historical data",
                "Personalized portfolio management",
                "Market trend predictions"
            ],
            "technology": [
                "Python Flask Backend",
                "Machine Learning Algorithms",
                "Real-time Data Processing",
                "Interactive Charting",
                "Responsive Web Design"
            ]
        },
        "mission": "To democratize stock market analysis through AI and help users make informed investment decisions.",
        "contact": {
            "email": "karthikabk2020@gmail.com",
            "support": "For technical support and feedback"
        }
    }
    return jsonify({"success": True, "about": about_data})

# Update the existing routes to use real data:

@app.route("/api/tickers", methods=["GET"])
def get_tickers():
    try:
        tickers_data = get_real_tickers()  # Use real data instead of sample
        return jsonify(tickers_data)
    except Exception as e:
        return jsonify({"success": False, "message": "Failed to fetch tickers"})

@app.route("/api/today", methods=["GET"])
def today_data():
    symbol = request.args.get("symbol")
    if not symbol:
        return jsonify({"success": False, "message": "Symbol parameter is required"})
    
    # Validate symbol
    if not validate_stock_symbol(symbol):
        return jsonify({"success": False, "message": "Invalid stock symbol"})
        
    try:
        chart_data = get_real_chart_data(symbol)  # Use real chart data
        return jsonify({
            "success": True,
            "today": chart_data["prices"],
            "volume": chart_data["volume"],
            "time_points": chart_data["time_points"],
            "metrics": chart_data["metrics"]
        })
        
    except Exception as e:
        print(f"Today data error: {e}")
        return jsonify({"success": False, "message": "Failed to fetch chart data"})

@app.route("/api/analyze", methods=["POST"])
@login_required
def analyze():
    try:
        data = request.get_json()
        symbol = data.get("symbol", "").strip().upper()
        
        if not symbol:
            return jsonify({"success": False, "message": "Symbol is required"})
        
        # Validate stock symbol
        if not validate_stock_symbol(symbol):
            return jsonify({"success": False, "message": "Invalid stock symbol format"})
            
        planned_amount = session.get("amount", 1000)
        analysis_result = analyze_stock(symbol, planned_amount)
        
        if not analysis_result:
            return jsonify({"success": False, "message": "Analysis failed"})
        
        # Get real chart data
        chart_data = get_real_chart_data(symbol)
            
        return jsonify({
            "success": True,
            "symbol": symbol,
            "recommendation": analysis_result["recommendation"],
            "advice": analysis_result["advice"],
            "details": analysis_result["details"],
            "planned_amount": analysis_result["planned_amount"],
            "investment_recommendation": analysis_result["investment_recommendation"],
            "expected_profit": analysis_result["expected_profit"],
            "chart_data": chart_data
        })
        
    except Exception as e:
        print(f"Analysis error: {e}")
        return jsonify({"success": False, "message": f"Analysis failed: {str(e)}"})

@app.route("/api/history", methods=["GET"])
def history_data():
    symbol = request.args.get("symbol")
    if not symbol:
        return jsonify({"success": False, "message": "Symbol parameter is required"})
        
    try:
        tk = yf.Ticker(symbol)
        df = tk.history(period="1mo", interval="1d")
        if df.empty:
            return jsonify({"success": False, "message": "No historical data available"})
            
        history_data = {}
        for timestamp, row in df.iterrows():
            date_key = timestamp.strftime("%Y-%m-%d")
            history_data[date_key] = float(row["Close"])
            
        prices = list(history_data.values())
        summary = {
            "high": max(prices),
            "low": min(prices),
            "avg": sum(prices) / len(prices)
        }
            
        return jsonify({
            "success": True, 
            "history": history_data,
            "summary": summary
        })
        
    except Exception as e:
        return jsonify({"success": False, "message": "Failed to fetch historical data"})

@app.route("/api/stream", methods=["GET"])
def stream_updates():
    symbol = request.args.get("symbol")
    if not symbol:
        return jsonify({"success": False, "message": "Symbol parameter is required"})
    
    def generate():
        while True:
            price_data = get_real_time_price(symbol)
            if price_data:
                yield f"data: {json.dumps(price_data)}\n\n"
            time.sleep(2)
    
    return app.response_class(generate(), mimetype="text/plain")

# -----------------------
# Main
# -----------------------
if __name__ == "__main__":
    # Create users.json if it doesn't exist
    if not os.path.exists(USERS_FILE):
        with open(USERS_FILE, "w") as f:
            json.dump({}, f)
    
    print("Starting Stock Agent Server with Real-time Data...")
    print(f"Frontend directory: {FRONTEND_DIR}")
    print(f"Backend directory: {BASE_DIR}")
    print("Available endpoints:")
    print("- / (Main frontend)")
    print("- /api/tickers (Real-time stock data)")
    print("- /api/analyze (AI stock analysis)") 
    print("- /api/today (Real chart data)")
    print("- /api/about (App information)")
    print("- /api/delete-account (Account management)")
    
    # Check if frontend directory exists
    if not os.path.exists(FRONTEND_DIR):
        print(f"⚠️  WARNING: Frontend directory not found at: {FRONTEND_DIR}")
        print("Please make sure your directory structure is:")
        print("stock-agent/")
        print("├── backend/")
        print("│   ├── main.py")
        print("│   └── users.json")
        print("└── frontend/")
        print("    └── index.html")
    

    app.run(host="192.168.0.113", port=5000, debug=True)
