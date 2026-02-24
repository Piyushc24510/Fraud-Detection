from flask import Flask, render_template, request, session, redirect, url_for, flash, jsonify, send_file
from functools import wraps
import pickle
import os
from datetime import datetime, timedelta
import hashlib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.sentiment import SentimentIntensityAnalyzer
from textstat import textstat
import json
from collections import Counter
import io
import csv
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import sqlite3

# ==========================================
# FLASK APP SETUP
# ==========================================
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "your-secret-key-here-change-it-in-production")

# ==========================================
# DATABASE SETUP
# ==========================================
DATABASE = 'fraud_detection.db'

def get_db():
    """Get database connection"""
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initialize database with tables"""
    conn = get_db()
    cursor = conn.cursor()
    
    # Users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            name TEXT NOT NULL,
            email TEXT,
            role TEXT DEFAULT 'user',
            joined TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # History table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            review TEXT NOT NULL,
            review_full TEXT,
            is_fake INTEGER NOT NULL,
            confidence REAL NOT NULL,
            checked_by TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create default admin if not exists
    cursor.execute("SELECT * FROM users WHERE username = 'admin'")
    if not cursor.fetchone():
        admin_pass = hashlib.sha256("admin123".encode()).hexdigest()
        cursor.execute('''
            INSERT INTO users (username, password, name, role, joined)
            VALUES (?, ?, ?, ?, ?)
        ''', ('admin', admin_pass, 'Administrator', 'admin', datetime.now().strftime("%Y-%m-%d")))
    
    conn.commit()
    conn.close()

# Initialize database on startup
init_db()

# ==========================================
# EMAIL NOTIFICATION CONFIG
# ==========================================
NOTIFY_EMAIL_SENDER   = "your_gmail@gmail.com"
NOTIFY_EMAIL_PASSWORD = "pvan ukge mzzq grbf"  # Use Gmail App Password, not your regular password
NOTIFY_EMAIL_RECEIVER = "chawlapiyush780@gmail.com"
NOTIFY_EMAIL_ENABLED  = False  # Set True after configuring Gmail App Password

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user" not in session:
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

def get_user_history(username=None):
    """Get history from database"""
    conn = get_db()
    cursor = conn.cursor()
    
    if username:
        cursor.execute("SELECT * FROM history WHERE checked_by = ? ORDER BY id DESC", (username,))
    else:
        cursor.execute("SELECT * FROM history ORDER BY id DESC")
    
    rows = cursor.fetchall()
    conn.close()
    
    return [dict(row) for row in rows]

def add_to_history(review, review_full, is_fake, confidence, checked_by):
    """Add review to database history"""
    conn = get_db()
    cursor = conn.cursor()
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute('''
        INSERT INTO history (review, review_full, is_fake, confidence, checked_by, timestamp)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (review, review_full, is_fake, confidence, checked_by, timestamp))
    
    conn.commit()
    conn.close()

# ==========================================
# EMAIL NOTIFICATION FUNCTION
# ==========================================
def send_registration_email(username, name, role):
    """Send email notification when a new user registers"""
    if not NOTIFY_EMAIL_ENABLED:
        return

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"🔔 New User Registration — {name} ({username})"
        msg["From"]    = NOTIFY_EMAIL_SENDER
        msg["To"]      = NOTIFY_EMAIL_RECEIVER

        registered_at = datetime.now().strftime("%d %b %Y, %I:%M %p")

        html_body = f"""
<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"></head>
<body style="margin:0;padding:0;background:#0a0a0a;font-family:'Segoe UI',Arial,sans-serif;">
  <table width="100%" cellpadding="0" cellspacing="0" style="background:#0a0a0a;padding:40px 20px;">
    <tr><td align="center">
      <table width="600" cellpadding="0" cellspacing="0"
             style="background:linear-gradient(135deg,#1e1e1e,#2a2a2a);
                    border-radius:16px;border:1px solid #333;
                    box-shadow:0 0 30px rgba(0,255,153,0.15);max-width:600px;width:100%;">
        <tr>
          <td style="background:linear-gradient(90deg,#00ff99,#00cc7a);
                     padding:28px 36px;text-align:center;">
            <h1 style="margin:0;color:#000;font-size:22px;font-weight:700;">
              🛡️ Review Fraud Detection System
            </h1>
            <p style="margin:6px 0 0;color:#003322;font-size:13px;">New User Registration Alert</p>
          </td>
        </tr>
        <tr>
          <td style="padding:36px;">
            <p style="margin:0 0 24px;color:#aaa;font-size:14px;">A new account has been created:</p>
            <table width="100%" style="background:#111;border-radius:12px;border:1px solid #333;margin-bottom:28px;">
              <tr style="border-bottom:1px solid #222;">
                <td style="padding:14px 20px;color:#666;font-size:13px;width:38%;">👤 Full Name</td>
                <td style="padding:14px 20px;color:#fff;font-size:14px;font-weight:600;">{name}</td>
              </tr>
              <tr style="border-bottom:1px solid #222;">
                <td style="padding:14px 20px;color:#666;font-size:13px;">🔑 Username</td>
                <td style="padding:14px 20px;color:#00ff99;font-size:14px;font-weight:600;
                           font-family:monospace;">{username}</td>
              </tr>
              <tr style="border-bottom:1px solid #222;">
                <td style="padding:14px 20px;color:#666;font-size:13px;">🎭 Role</td>
                <td style="padding:14px 20px;">
                  <span style="background:#1a3a2a;color:#00ff99;padding:4px 12px;
                               border-radius:20px;font-size:12px;font-weight:700;
                               text-transform:uppercase;">{role}</span>
                </td>
              </tr>
              <tr>
                <td style="padding:14px 20px;color:#666;font-size:13px;">🕐 Registered At</td>
                <td style="padding:14px 20px;color:#fff;font-size:13px;">{registered_at}</td>
              </tr>
            </table>
          </td>
        </tr>
        <tr>
          <td style="background:#111;padding:20px 36px;border-top:1px solid #222;text-align:center;">
            <p style="margin:0;color:#444;font-size:12px;">Automated notification from Review Fraud Detection System</p>
          </td>
        </tr>
      </table>
    </td></tr>
  </table>
</body>
</html>
"""
        msg.attach(MIMEText(html_body, "html"))

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(NOTIFY_EMAIL_SENDER, NOTIFY_EMAIL_PASSWORD)
            server.sendmail(NOTIFY_EMAIL_SENDER, NOTIFY_EMAIL_RECEIVER, msg.as_string())

        print(f"✅ Registration email sent for user: {username}")

    except Exception as e:
        print(f"❌ Email error (non-critical): {e}")

# ==========================================
# LOAD ML MODEL
# ==========================================
try:
    model = pickle.load(open("model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    print("✅ ML Model loaded successfully")
except:
    print("⚠️ Model not found - run train_model.py first")
    model = None
    vectorizer = None

# NLTK Setup
try:
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    sia = SentimentIntensityAnalyzer()
except:
    print("⚠️ NLTK data not found")

# ==========================================
# ROUTES
# ==========================================

@app.route("/")
def home():
    if "user" in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route("/login", methods=["GET", "POST"])
def login():
    if "user" in session:
        return redirect(url_for('dashboard'))
    
    error = None
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
        user = cursor.fetchone()
        conn.close()
        
        if user:
            hashed_password = hash_password(password)
            if user['password'] == hashed_password:
                session["user"] = username
                session["user_name"] = user['name']
                session["user_role"] = user['role']
                
                next_page = request.args.get('next')
                if next_page:
                    return redirect(next_page)
                return redirect(url_for('dashboard'))
            else:
                error = "Invalid password"
        else:
            error = "Username not found"
    
    return render_template("login.html", error=error)

@app.route("/register", methods=["GET", "POST"])
def register():
    if "user" in session:
        return redirect(url_for('dashboard'))
    
    error = None
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        confirm_password = request.form.get("confirm_password")
        name = request.form.get("name")
        
        if not username or not password or not confirm_password or not name:
            error = "All fields are required"
        elif password != confirm_password:
            error = "Passwords do not match"
        elif len(password) < 6:
            error = "Password must be at least 6 characters"
        else:
            conn = get_db()
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
            
            if cursor.fetchone():
                error = "Username already exists"
            else:
                hashed_pass = hash_password(password)
                joined = datetime.now().strftime("%Y-%m-%d")
                
                cursor.execute('''
                    INSERT INTO users (username, password, name, role, joined)
                    VALUES (?, ?, ?, ?, ?)
                ''', (username, hashed_pass, name, 'user', joined))
                
                conn.commit()
                conn.close()
                
                # Send email notification
                send_registration_email(username, name, "user")
                
                success = "Registration successful! Please login."
                return render_template("login.html", success=success)
            
            conn.close()
    
    return render_template("register.html", error=error)

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route("/dashboard")
@login_required
def dashboard():
    history = get_user_history()
    
    total = len(history)
    fake = sum(1 for h in history if h.get('is_fake', False))
    genuine = total - fake
    
    stats = {
        'total': total,
        'fake': fake,
        'genuine': genuine,
        'fake_percent': round((fake / total) * 100, 1) if total > 0 else 0,
        'genuine_percent': round((genuine / total) * 100, 1) if total > 0 else 0
    }
    
    return render_template("dashboard_advanced.html",
        user_name=session.get("user_name"),
        user_role=session.get("user_role"),
        **stats,
        history=history[:10]
    )

@app.route("/check", methods=["GET", "POST"])
@login_required
def check():
    result = None
    if request.method == "POST" and model:
        review_text = request.form.get("review", "").strip()
        
        if review_text:
            # Preprocess
            text_lower = review_text.lower()
            text_clean = re.sub('[^a-zA-Z]', ' ', text_lower)
            words = [stemmer.stem(w) for w in text_clean.split() if w and w not in stop_words]
            processed = " ".join(words)
            
            # Predict
            vec = vectorizer.transform([processed])
            prediction = model.predict(vec)[0]
            probabilities = model.predict_proba(vec)[0]

            # Extract features for explanation
            word_count   = len(review_text.split())
            unique_words = round(len(set(review_text.lower().split())) / max(word_count, 1) * 100, 1)
            exclamations = review_text.count('!')
            caps_ratio   = round(sum(c.isupper() for c in review_text) / max(len(review_text), 1) * 100, 1)
            extreme_list = ['worst','best','terrible','amazing','horrible','perfect',
                            'awful','fantastic','garbage','excellent','pathetic','outstanding']
            extreme_words = sum(1 for w in review_text.lower().split() if w.strip('.,!?') in extreme_list)

            fake_prob    = round(probabilities[1] * 100, 2)
            genuine_prob = round(probabilities[0] * 100, 2)
            confidence   = round(max(probabilities) * 100, 2)

            # Build red/green flags
            red_flags   = []
            green_flags = []

            if exclamations > 2:
                red_flags.append(f"Excessive exclamation marks ({exclamations})")
            if caps_ratio > 15:
                red_flags.append(f"High use of capital letters ({caps_ratio}%)")
            if extreme_words >= 2:
                red_flags.append(f"Multiple extreme/emotional words detected ({extreme_words})")
            if word_count <= 5:
                red_flags.append("Very short review — lacks detail")
            if fake_prob > 70:
                red_flags.append(f"High fake probability score ({fake_prob}%)")

            if word_count > 20:
                green_flags.append(f"Detailed review with {word_count} words")
            if unique_words > 70:
                green_flags.append(f"High word diversity ({unique_words}%) — natural writing")
            if exclamations == 0:
                green_flags.append("No excessive punctuation")
            if extreme_words == 0:
                green_flags.append("No extreme emotional language")
            if genuine_prob > 70:
                green_flags.append(f"High genuine probability ({genuine_prob}%)")

            result = {
                'text':               review_text,
                'review':             review_text[:100] if len(review_text) > 100 else review_text,
                'review_full':        review_text,
                'is_fake':            bool(prediction),
                'confidence':         confidence,
                'fake_probability':   fake_prob,
                'genuine_probability':genuine_prob,
                'explanation': {
                    'summary': (
                        f"This review has been classified as {'FAKE' if prediction else 'GENUINE'} "
                        f"with {confidence}% confidence. "
                        f"{'Several suspicious patterns were detected.' if red_flags else 'No major suspicious patterns found.'}"
                    ),
                    'red_flags':   red_flags,
                    'green_flags': green_flags,
                    'technical_details': {
                        'word_count':     word_count,
                        'unique_words':   unique_words,
                        'sentiment_score': round(fake_prob - genuine_prob, 1),
                        'exclamations':   exclamations,
                        'caps_ratio':     caps_ratio,
                        'extreme_words':  extreme_words
                    }
                }
            }
            
            # Add to history
            add_to_history(
                result['review'],
                result['review_full'],
                result['is_fake'],
                result['confidence'],
                session.get("user")
            )
    
    return render_template("check.html",
        user_name=session.get("user_name"),
        user_role=session.get("user_role"),
        result=result
    )

@app.route("/history")
@login_required
def history_view():
    user_role = session.get("user_role")
    
    if user_role == 'admin':
        history = get_user_history()
    else:
        history = get_user_history(session.get("user"))
    
    return render_template("history.html",
        user_name=session.get("user_name"),
        user_role=user_role,
        history=history
    )

@app.route("/analytics")
@login_required
def analytics():
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    
    if not start_date or not end_date:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
    else:
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
        end_date = datetime.strptime(end_date, "%Y-%m-%d")
    
    history = get_user_history()
    
    # Filter by date
    filtered = []
    for item in history:
        try:
            item_date = datetime.strptime(item['timestamp'], "%Y-%m-%d %H:%M:%S")
            if start_date <= item_date <= end_date:
                filtered.append(item)
        except:
            continue
    
    # Stats
    total = len(filtered)
    fake = sum(1 for h in filtered if h.get('is_fake'))
    genuine = total - fake
    
    # User stats
    user_stats = {}
    for item in filtered:
        user = item.get('checked_by', 'Unknown')
        if user not in user_stats:
            user_stats[user] = {'total': 0, 'fake': 0, 'genuine': 0}
        user_stats[user]['total'] += 1
        if item.get('is_fake'):
            user_stats[user]['fake'] += 1
        else:
            user_stats[user]['genuine'] += 1
    
    return render_template("analytics.html",
        user_name=session.get("user_name"),
        user_role=session.get("user_role"),
        total=total,
        fake=fake,
        genuine=genuine,
        history=filtered,
        user_stats=user_stats,
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d")
    )

@app.route("/users")
@login_required
def user_management():
    # Only admin can access
    if session.get("user_role") != "admin":
        return redirect(url_for('dashboard'))

    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT id, username, name, role, joined FROM users ORDER BY id DESC")
    users = [dict(row) for row in cursor.fetchall()]

    # Per-user review counts
    cursor.execute("SELECT checked_by, COUNT(*) as total FROM history GROUP BY checked_by")
    review_counts = {row['checked_by']: row['total'] for row in cursor.fetchall()}
    conn.close()

    for u in users:
        u['review_count'] = review_counts.get(u['username'], 0)

    return render_template("user_management.html",
        user_name=session.get("user_name"),
        user_role=session.get("user_role"),
        users=users,
        current_user=session.get("user")
    )

@app.route("/users/delete/<int:user_id>", methods=["POST"])
@login_required
def delete_user(user_id):
    if session.get("user_role") != "admin":
        return redirect(url_for('dashboard'))

    conn = get_db()
    cursor = conn.cursor()
    # Prevent deleting own account
    cursor.execute("SELECT username FROM users WHERE id = ?", (user_id,))
    row = cursor.fetchone()
    if row and row['username'] == session.get("user"):
        conn.close()
        return redirect(url_for('user_management'))

    cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
    conn.commit()
    conn.close()
    return redirect(url_for('user_management'))

@app.route("/users/role/<int:user_id>", methods=["POST"])
@login_required
def change_role(user_id):
    if session.get("user_role") != "admin":
        return redirect(url_for('dashboard'))

    new_role = request.form.get("role")
    if new_role not in ("admin", "user"):
        return redirect(url_for('user_management'))

    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("UPDATE users SET role = ? WHERE id = ?", (new_role, user_id))
    conn.commit()
    conn.close()
    return redirect(url_for('user_management'))

@app.route("/clear-history")
@login_required
def clear_history():
    if session.get("user_role") == "admin":
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM history")
        conn.commit()
        conn.close()
    return redirect(url_for('history_view'))

@app.errorhandler(404)
def not_found(e):
    return render_template("404.html"), 404

# ==========================================
# RUN APP
# ==========================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_ENV") != "production"
    app.run(debug=debug, host="127.0.0.1", port=port)