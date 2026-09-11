import streamlit as st
import bcrypt
import os
import sqlite3
import base64

# Define Base Directory for dynamic paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -----------------------------------------DATABASE SETUP--------------------------------------
@st.cache_resource
def init_db():
    db_path = os.path.join(BASE_DIR, "..", "accounts.db")
    conn = sqlite3.connect(db_path, check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password BLOB NOT NULL
        )
    ''')
    conn.commit()
    return conn

db = init_db()

# -----------------------------------------LOGING AND VERIFICATION--------------------------------------
def validate_login(email, password):
    cursor = db.cursor()
    query = "SELECT password FROM users WHERE email = ?"
    cursor.execute(query, (email,))
    row = cursor.fetchone()

    if row and bcrypt.checkpw(password.encode('utf-8'), row[0]):
        return True
    return False

def create_account(email, password):
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    cursor = db.cursor()
    try:
        query = "INSERT INTO users (email, password) VALUES (?, ?)"
        cursor.execute(query, (email, hashed_password))
        db.commit()
        return True
    except sqlite3.IntegrityError:
        return False # Email already exists

# -----------------------------------------SET BACKGROUND IMAGE--------------------------------------
def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    if not os.path.exists(png_file):
        return # Gracefully fail if image is missing
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

# -----------------------------------------LOGIN PAGE--------------------------------------
st.title("Medicine Fraud Detection App")

bg_path = os.path.join(BASE_DIR, "..", "widgets", "medicine-capsules.png")
set_background(bg_path)

st.write("Welcome! Please log in or sign up to continue:")

option = st.radio("Select Option", ["Login", "Sign Up"], key="login_radio")

if option == "Sign Up":
    new_email = st.text_input("New Email")
    new_password = st.text_input("New Password", type="password")

    if st.button("Sign Up"):
        if new_email and new_password:
            success = create_account(new_email, new_password)
            if success:
                st.success("Account created successfully! Now you can log in!")
            else:
                st.error("An account with this email already exists.")
        else:
            st.warning("Please provide both email and password.")

elif option == "Login":
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if st.button("Log In"):
        if validate_login(email, password):
            st.success("Login Successful!")
            st.switch_page("pages/main.py")
        else:
            st.warning('Verify your information! Invalid email or password.')