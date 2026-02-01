# --- Standard library imports ---
import os
import re
import uuid
import sqlite3
import datetime
from io import BytesIO
import glob
import json

# --- Flask & Werkzeug ---
from flask import Flask, render_template, request, redirect, session, url_for, flash, send_file
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

# --- Data processing & ML ---
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib

# --- Visualization ---
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FuncFormatter
import seaborn as sns

# --- PDF generation ---
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

# === Paths and Constants ===
UPLOAD_FOLDER = 'uploads'
PLOT_FOLDER = 'static/plots'
PREDICTION_FOLDER = 'static/predictions'
MODEL_PATH = os.path.join('model', 'best_model.pkl')
FEATURES_PATH = os.path.join('model', 'model_features.pkl')
ENCODER_PATH = os.path.join('model', 'encoders.pkl')
SCALER_PATH = os.path.join('model', 'scaler.pkl')
USER_FILE = 'users.json'  


os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PLOT_FOLDER, exist_ok=True)
os.makedirs(PREDICTION_FOLDER, exist_ok=True)

# === Flask App Init ===
app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# === Load Model Once Globally ===
model = joblib.load(MODEL_PATH)
label_encoders = joblib.load(ENCODER_PATH)
Features_data = joblib.load(FEATURES_PATH)
scaler = joblib.load(SCALER_PATH)

# === Feature Columns ===
feature_names = ['Tenure', 'PreferredLoginDevice', 'CityTier', 'WarehouseToHome',
                 'PreferredPaymentMode', 'Gender', 'HourSpendOnApp', 'NumberOfDeviceRegistered',
                 'PreferedOrderCat', 'SatisfactionScore', 'MaritalStatus', 'NumberOfAddress',
                 'Complain', 'OrderAmountHikeFromlastYear', 'CouponUsed', 'OrderCount',
                 'DaySinceLastOrder', 'CashbackAmount']

def generate_charts(df):
    chart_paths = []
    output_folder = 'static/plots'
    os.makedirs(output_folder, exist_ok=True)

    # Styles
    title_font = {'fontsize': 16, 'fontweight': 'bold'}
    label_font = {'fontsize': 14, 'fontweight': 'bold'}
    tick_fontsize = 12
    legend_fontsize = 14

    def save_fig(title):
        file_path = os.path.join(output_folder, f"{title}.png")
        plt.tight_layout()
        plt.savefig(file_path, dpi=300)
        chart_paths.append('/' + file_path.replace("\\", "/"))
        plt.close()

    # 1. Overall churn donut chart
    if 'Churn' in df.columns:
        churn_counts = df['Churn'].value_counts().reindex([0,1], fill_value=0)
        plt.figure(figsize=(6,6))
        wedges, texts, autotexts = plt.pie(churn_counts, labels=['Not Churn', 'Churn'],
            autopct='%1.1f%%', startangle=90,
            colors=['#7b2cbf','#f72585'], textprops={'fontsize':12})
        
        centre_circle = plt.Circle((0,0),0.70,fc='white')
        plt.gcf().gca().add_artist(centre_circle)
        plt.title("Overall Churn Rate", **title_font)
        save_fig("overall_churn_rate")

    # 2. Numerical features → Histogram with Hue + Percentages
    num_features = [
        'Tenure','CityTier','WarehouseToHome','HourSpendOnApp',
        'NumberOfDeviceRegistered','SatisfactionScore','NumberOfAddress',
        'CashbackAmount','OrderAmountHikeFromlastYear',
        'CouponUsed','OrderCount','DaySinceLastOrder'
    ]

    for feature in num_features:
        if feature in df.columns and 'Churn' in df.columns:
            df_feature = df[[feature, 'Churn']].dropna()
            if df_feature.empty:
                continue

            # Map 0 → Not Churn, 1 → Churn
            df_feature['Churn'] = df_feature['Churn'].map({1: "Churn", 0: "Not Churn"})

            plt.figure(figsize=(8,5))

            ax = sns.histplot(
                data=df_feature,
                x=feature,
                hue='Churn',
                hue_order=["Not Churn", "Churn"],
                bins=10,
                multiple='dodge',
                palette={"Not Churn": "#7b2cbf", "Churn": "#f72585"},
                stat='count',
                legend=True  )

            plt.xlabel(feature, **label_font)
            plt.ylabel("Number of Customers", **label_font)
            plt.title(f"Churn Distribution by {feature}", **title_font)
            plt.xticks(rotation=45, fontsize=tick_fontsize)
            plt.yticks(fontsize=tick_fontsize)

            # ✅ Manually rebuild legend if seaborn still fails
            handles, labels = ax.get_legend_handles_labels()
            if handles:  # only if seaborn returned something
                plt.legend(handles, ["Not Churn", "Churn"], fontsize=legend_fontsize,
                title="Churn")

            save_fig(f"{feature} vs Churn")

    # 3. Categorical features → Grouped bar plots
    cat_features = ['PreferredLoginDevice','PreferredPaymentMode','Gender',
                    'PreferedOrderCat','MaritalStatus','Complain']

    for feature in cat_features:
        if feature in df.columns and 'Churn' in df.columns:
            plt.figure(figsize=(8,5))
            
            sns.countplot(
                data=df,
                x=feature,
                hue='Churn',
                palette=['#7b2cbf','#f72585'],
                dodge=True)
            
            plt.xlabel(feature, **label_font)
            plt.ylabel("Count", **label_font)
            plt.title(f"Churn Distribution by {feature}", **title_font)
            plt.xticks(rotation=30, fontsize=tick_fontsize)
            plt.yticks(fontsize=tick_fontsize)
            plt.legend(['Not Churn','Churn'], fontsize=legend_fontsize, title='Churn')
            
            save_fig(f"{feature} vs Churn")

    return chart_paths

def load_users():
    try:
        with open(USER_FILE, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save_users(users):
    """Save users dictionary to JSON file."""
    with open(USER_FILE, 'w') as f:
        json.dump(users, f, indent=4)
        
# === LOGIN ===
@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Load users from file
        users = load_users()
        user = users.get(username)  # user is a dict with 'password' and optional 'id'

        # Check password
        if user and check_password_hash(user["password"], password):
            # Save user info in session
            session['user'] = username
            session['user_id'] = user.get("id", username)  # fallback to username if no id

            # Clear old recommendations and visualizations for this session
            session.pop('recommendations', None)
            session.pop('visualization_data', None)

            # Clear old prediction file for this user
            predicted_file_path = f"static/predictions/results_{session['user_id']}.csv"
            if os.path.exists(predicted_file_path):
                try:
                    os.remove(predicted_file_path)
                except Exception as e:
                    print(f"Error deleting {predicted_file_path}: {e}")

            # Clear old plot images so visualization starts fresh
            plot_folder = os.path.join('static', 'plots')
            if os.path.exists(plot_folder):
                files = glob.glob(os.path.join(plot_folder, '*.png'))
                for f in files:
                    try:
                        os.remove(f)
                    except Exception as e:
                        print(f"Error deleting {f}: {e}")

            flash("Logged in successfully!")
            return redirect(url_for('bulk_upload'))
        else:
            flash("Invalid username or password")            

    return render_template('login.html')

# === SIGNUP ===
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password']

        if not username or not password:
            flash("Username and password cannot be empty")
            return render_template('signup.html')

        users = load_users()

        if username in users:
            flash('Username already exists')
            return render_template('signup.html')

        # Hash the password and generate a simple ID
        hashed_password = generate_password_hash(password)
        user_id = max([u.get("id", 0) for u in users.values()] + [0]) + 1

        users[username] = {
            "id": user_id,
            "password": hashed_password
        }

        save_users(users)
        flash('Signup successful! Please login.')
        return redirect(url_for('login'))

    return render_template('signup.html')

# === BULK UPLOAD  ===
@app.route('/bulk_upload', methods=['GET', 'POST'])
def bulk_upload():
    if 'user' not in session:
        return redirect(url_for('login'))

    predictions = None
    categorical_cols = ['PreferredLoginDevice', 'PreferredPaymentMode', 'Gender', 
                        'PreferedOrderCat', 'MaritalStatus']

    if request.method == 'POST':
        file = request.files.get('file')
        if file and file.filename.endswith(('.csv', '.xlsx', '.xls')):
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

            try:                
                # Read uploaded file
                df = pd.read_csv(filepath) if filepath.endswith('.csv') else pd.read_excel(filepath)
                original_df = df.copy()

                # Drop CustomerID if exists
                if 'CustomerID' in df.columns:
                    df = df.drop(columns=['CustomerID'])

                # === Handle missing values ===
                # Fill missing numerical values with mean
                num_cols = df.select_dtypes(include=['float64', 'int64']).columns
                for col in num_cols:
                    df[col] = df[col].fillna(df[col].mean())

                # Fill missing categorical values with mode
                cat_cols = df.select_dtypes(include='object').columns
                for col in cat_cols:
                    if df[col].isnull().sum() > 0:
                        df[col] = df[col].fillna(df[col].mode()[0])
                    # Clean category values (remove spaces / unify case)
                    df[col] = df[col].astype(str).str.strip().str.title()

                # === Encode categorical features using saved encoders ===
                def normalize_word(x, known_classes):
                    x_clean = re.sub(r"\s+", "", x.lower())  # remove spaces, lowercase
                    for cls in known_classes:
                        cls_clean = re.sub(r"\s+", "", cls.lower())
                        if x_clean == cls_clean:
                            return cls  # return matched class in original form
                    return "Unknown"  # fallback if no match

                for col in categorical_cols:
                    if col in df.columns:
                        # Clean and standardize input
                        df[col] = df[col].astype(str).str.strip().str.title()
                        le = label_encoders[col]

                        # Try to normalize values against known classes
                        df[col] = df[col].apply(lambda x: normalize_word(x, le.classes_))

                        # Ensure "Unknown" exists in the encoder
                        if "Unknown" not in le.classes_:
                            le.classes_ = np.append(le.classes_, "Unknown")

                        # Transform into numeric labels
                        df[col] = le.transform(df[col])

                # === Scale numerical features using saved scaler ===
                numerical_cols = ['Tenure', 'WarehouseToHome', 'HourSpendOnApp',
                                'NumberOfDeviceRegistered', 'SatisfactionScore',
                                'NumberOfAddress', 'OrderAmountHikeFromlastYear',
                                'CouponUsed', 'OrderCount', 'DaySinceLastOrder',
                                'CashbackAmount']

                df[numerical_cols] = scaler.transform(df[numerical_cols])

                # === Align features with training ===
                df = df.reindex(columns=Features_data, fill_value=0)

                # === Predict Churn ===
                predictions_array = model.predict(df)

                # Store predictions directly as 0/1
                original_df['Churn'] = predictions_array  

                # Convert to dict for rendering in template
                predictions = original_df.to_dict(orient='records')
                
                # Save predictions CSV
                predictions_folder = os.path.join('static', 'predictions')
                os.makedirs(predictions_folder, exist_ok=True)
                result_path = os.path.join(predictions_folder, f'results_{session["user_id"]}.csv')
                original_df.to_csv(result_path, index=False)

                flash("File uploaded and predictions generated successfully!")

            except Exception as e:
                flash(f"Prediction error: {e}")
        else:
            flash("Please upload a valid file (.csv, .xlsx, .xls).")       

    return render_template('bulk_upload.html', predictions=predictions)

# === SINGLE PREDICTION ===
@app.route('/single_prediction', methods=['GET', 'POST'])
def single_prediction():
    if 'user' not in session:
        return redirect(url_for('login'))

    result = None
    categorical_cols = ['PreferredLoginDevice', 'PreferredPaymentMode', 'Gender', 
                        'PreferedOrderCat', 'MaritalStatus']

    if request.method == 'POST':
        try:
            # Get customer ID from form
            customer_id = request.form.get('customer_id')
            
            # Collect form data
            data = [request.form.get(f) for f in feature_names]
            df = pd.DataFrame([data], columns=feature_names)

            # Convert numeric columns to float if possible
            for col in df.columns:
                try:
                    df[col] = df[col].astype(float)
                except:
                    pass

            # Encode categorical features using saved label_encoders
            for col in categorical_cols:
                if col in df.columns:
                    df[col] = df[col].astype(str)
                    df[col] = label_encoders[col].transform(df[col])

            df = df[Features_data]

            # Make prediction
            prediction = model.predict(df)[0]

            result = "Churn" if prediction == 1 else "Not Churn"
            
            return redirect(url_for("single_prediction", result=result) + "#prediction-result")

        except Exception as e:
            flash(f"Prediction failed: {e}")

    # fetch result from query params if redirected
    result = request.args.get("result")

    return render_template("single_prediction.html", result=result, feature_names=feature_names)

@app.route('/visualizations')
def visualizations():

    if 'user' not in session:
        return redirect(url_for('login'))
    
    # Load the predicted file
    user_id = session['user_id']
    predicted_file_path = f'static/predictions/results_{user_id}.csv'

    if not os.path.exists(predicted_file_path):
        empty_kpis = {
        'total_customers': 0,
        'churned_customers': 0,
        'retained_customers': 0,
        'avg_cashback': 0,
        'avg_screen_time': 0
        }
        return render_template('visualization.html', data=None, kpis=empty_kpis)

    # Read the predicted CSV
    df = pd.read_csv(predicted_file_path)

    # Ensure 'Churn' column exists in the predicted file
    if 'Churn' not in df.columns:
        flash("'Churn' column missing in predicted file. Please run prediction first.", 'danger')
        return redirect(url_for('bulk_upload'))

    # Compute KPIs
    total_customers = len(df)
    churned_customers = df[df['Churn'] == 1].shape[0]
    retained_customers = df[df['Churn'] == 0].shape[0]

    avg_cashback = round(df['CashbackAmount'].mean(), 2) if 'CashbackAmount' in df.columns else 0
    avg_screen_time = round(df['HourSpendOnApp'].mean(), 2) if 'HourSpendOnApp' in df.columns else 0

    kpis = {
        'total_customers': total_customers,
        'churned_customers': churned_customers,
        'retained_customers': retained_customers,
        'avg_cashback': avg_cashback,
        'avg_screen_time': avg_screen_time
    }

    # Generate charts from the predicted data
    charts = generate_charts(df)

    return render_template('visualization.html', charts=charts, kpis=kpis)

@app.route('/recommendations', methods=['GET', 'POST'])
def recommendations():

    user_id = session['user_id']
    predicted_csv_path = f'static/predictions/results_{user_id}.csv'

    # Check if predictions exist
    if not os.path.exists(predicted_csv_path):
        return render_template('recommendations.html', recommendations=[])

    # Read predictions CSV
    df = pd.read_csv(predicted_csv_path)

    features = [
        'Tenure', 'PreferredLoginDevice', 'CityTier', 'WarehouseToHome',
        'PreferredPaymentMode', 'Gender', 'HourSpendOnApp', 'NumberOfDeviceRegistered',
        'PreferedOrderCat', 'SatisfactionScore', 'MaritalStatus', 'NumberOfAddress',
        'Complain', 'OrderAmountHikeFromlastYear', 'CouponUsed', 'OrderCount',
        'DaySinceLastOrder', 'CashbackAmount'
    ]

    recommendations = []
    to_insert = []

    # Generate professional recommendations
    for feature in features:
        if feature in df.columns and 'Churn' in df.columns:
            churn_rate = df.groupby(feature)['Churn'].mean()
            if churn_rate.max() - churn_rate.min() > 0.1:
                strategy = ""
                if feature == 'Tenure':
                    strategy = ("Short-tenure customers are at higher risk of churn. "
                                "Welcome new customers with personalized guidance, reward programs, and early engagement activities during their first 3 months." )               
                
                elif feature == 'HourSpendOnApp':
                    strategy = ("Low app engagement correlates with churn. "
                                "Encourage customers to use the app more by adding fun challenges, helpful notifications, and personalized content to increase active usage.")
                
                elif feature == 'SatisfactionScore':
                    strategy = ("Low satisfaction leads to churn. "
                                "Conduct customer feedback surveys, enhance product or service quality.")
                
                elif feature == 'OrderCount':
                    strategy = ("Low order frequency is driving churn. "
                                "Offer subscription models, personalized discounts, and reactivation campaigns for inactive users.")
                
                elif feature == 'Complain':
                    strategy = ("High churn among customers who complain. "
                                "Customers who have complaints are more likely to leave. Quickly resolve their issues, check how happy they are with your service, and follow up with personalized solutions to keep them satisfied.")                
                
                elif feature == 'CouponUsed':
                    strategy = ("Non-coupon users churn more. "
                                "Offer customers special deals, personalized coupons, and loyalty rewards to keep them coming back.")
                
                elif feature == 'CashbackAmount':
                    strategy = ("Low cashback users are churning. "
                                "Introduce cashback tiers for loyal customers and double-cashback weekends to incentivize repeat purchases.")
                
                elif feature == 'PreferredLoginDevice':
                    strategy = ("Device preference impacts churn. "
                                "Optimize mobile and desktop user experience, ensure consistent app performance, and tailor offers by device usage.")
                
                elif feature == 'PreferredPaymentMode':
                    strategy = ("Churn varies by payment mode. "
                                "Make it easy for customers to pay by offering smooth checkout, digital wallets, and rewards for using their favorite payment methods.")
                
                elif feature == 'CityTier':
                    strategy = ("City tier differences affect churn. "
                                "Customize campaigns for smaller cities, improve delivery to remote areas, and create content that feels local and relevant to each region.")
                
                elif feature == 'WarehouseToHome':
                    strategy = ("Delivery distance influences churn. "
                                "Improve last-mile delivery efficiency, introduce express delivery options, and partner with local logistics providers.")
                
                else:
                    strategy = (f"{feature} strongly influences churn. "
                                "Implement customer segmentation and targeted retention strategies to reduce churn.")

                recommendations.append(strategy)

    # Handle PDF download
    if request.method == 'POST':
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter, leftMargin=50, rightMargin=50)
        styles = getSampleStyleSheet()
        story = []

        story.append(Paragraph("Customer Churn Recommendations", styles['Title']))
        story.append(Spacer(1, 20))
        for rec in recommendations:
            story.append(Paragraph(f"- {rec}", styles['Normal']))
            story.append(Spacer(1, 10))

        doc.build(story)
        buffer.seek(0)
        return send_file(
            buffer,
            as_attachment=True,
            download_name="churn_recommendations.pdf",
            mimetype='application/pdf'
        )

    return render_template('recommendations.html', recommendations=recommendations)

@app.route('/download')
def download():
    # File is saved as: results_<user_id>.csv
    filename = f'results_{session["user_id"]}.csv'
    result_path = os.path.join('static', 'predictions', filename)

    if os.path.exists(result_path):
        return send_file(result_path, as_attachment=True)
    else:
        flash("No results found to download.", "warning")
        return redirect(url_for('bulk_upload'))


# === LOGOUT ===
@app.route('/logout')
def logout():
    user_id = session.get('user_id')

    # Clear session
    session.clear()

    # Remove user's old prediction file
    if user_id:
        predicted_file_path = f'static/predictions/results_{user_id}.csv'
        if os.path.exists(predicted_file_path):
            try:
                os.remove(predicted_file_path)
            except Exception as e:
                print(f"Error deleting {predicted_file_path}: {e}")

    return redirect(url_for('login'))

# === MAIN ===
if __name__ == '__main__':
    app.run(debug=True)