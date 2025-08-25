# Refurbished_Phone_App
"A Streamlit-based Refurbished Phone App with SQLite database to manage phones, pricing, and listings. Built using Python, it includes admin login, phone catalog, and dynamic search features. Designed to simplify refurbished phone sales and provide an easy-to-use management dashboard."


# 🚀 Features

Admin authentication (secure login)

Add, view, and manage phone details

Pricing and inventory management

Export inventory as CSV

Built with a clean UI using Streamlit


# 🛠️ Tools & Libraries Used

Python 3

Streamlit (for web interface)

SQLite3 (database)

Pandas (CSV export/import)


# ⚡ How to Run Locally

1. Clone this repo:

git clone https://github.com/yourusername/refurbished-phone-app.git
cd refurbished-phone-app


2. Create and activate virtual environment:

python -m venv .venv  
.venv\Scripts\activate   # Windows  
source .venv/bin/activate # Mac/Linux


3. Install dependencies:

pip install -r requirements.txt


4. Run the app:

streamlit run app.py

# requirements.txt file for the refurbished phone app.
You can create a new file named requirements.txt in your project folder and copy this content inside 👇

streamlit==1.37.0
pandas==2.2.2
altair==5.3.0
numpy==1.26.4

✅ Explanation:
streamlit → for the web interface

pandas → for CSV export/import

altair → for charts/visualization (Streamlit dependency)

numpy → often required by pandas & Streamlit


⚡ Now when someone clones my repo, they just need to run:

pip install -r requirements.txt


# 🔑 Default Login

Username: admin

Password: admin123
