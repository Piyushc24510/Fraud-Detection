import os
import subprocess
import sys

def main():
    print("=" * 50)
    print("REVIEW FRAUD DETECTION SYSTEM")
    print("=" * 50)
    
    # Check if model files exist
    if not os.path.exists("model.pkl") or not os.path.exists("vectorizer.pkl"):
        print("\n📊 Training model for the first time...")
        subprocess.run([sys.executable, "train_model.py"])
    
    print("\n🚀 Starting Flask application...")
    print("📱 Access the application at: http://localhost:5000")
    print("👤 Admin login: http://localhost:5000/admin")
    print("🔑 Username: admin | Password: admin123")
    print("=" * 50)
    
    # Run the Flask app
    subprocess.run([sys.executable, "app.py"])

if __name__ == "__main__":
    main()