from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_user, login_required, logout_user, UserMixin, current_user
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.date import DateTrigger
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
from dotenv import load_dotenv
import smtplib
from email.message import EmailMessage
import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import pickle
import matplotlib.pyplot as plt

# ================== CONFIG ==================
load_dotenv()
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:krish123@localhost/medical_remainder'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# ================== MODELS ==================
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)

class Appointment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(150), nullable=False)
    patient_name = db.Column(db.String(150), nullable=False)
    datetime = db.Column(db.DateTime, nullable=False)
    reminder_sent = db.Column(db.Boolean, default=False)

# ================== LOGIN HANDLER ==================
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# ================== EMAIL REMINDER ==================
def send_email_reminder(to_email, message):
    try:
        msg = EmailMessage()
        msg['Subject'] = 'Health Alert'
        msg['From'] = os.environ.get("EMAIL_USER")
        msg['To'] = to_email
        msg.set_content(message)

        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(os.environ.get("EMAIL_USER"), os.environ.get("EMAIL_PASS"))
            smtp.send_message(msg)
        print(f"✅ Reminder sent to {to_email}")
        return True
    except Exception as e:
        print(f"❌ Could not send email: {e}")
        return False

# ================== BREAST CANCER MODEL ==================
MODEL_PATH = "breast_cancer_model.h5"
if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
else:
    model = None

# ================== SCHEDULER ==================
scheduler = BackgroundScheduler()
scheduler.start()

def schedule_appointment_reminder(appt):
    if appt.reminder_sent:
        return

    def send_reminder():
        message = f"Reminder: You have an appointment for {appt.patient_name} at {appt.datetime.strftime('%Y-%m-%d %H:%M')}"
        if send_email_reminder(appt.email, message):
            with app.app_context():
                appt.reminder_sent = True
                db.session.commit()

    scheduler.add_job(
        func=send_reminder,
        trigger=DateTrigger(run_date=appt.datetime),
        id=f"appt_{appt.id}",
        replace_existing=True
    )
    print(f"📅 Scheduled reminder for {appt.patient_name} at {appt.datetime}")

# Schedule reminders at startup
with app.app_context():
    future_appointments = Appointment.query.filter(
        Appointment.datetime >= datetime.now(),
        Appointment.reminder_sent == False
    ).all()
    for appt in future_appointments:
        schedule_appointment_reminder(appt)

# ================== PLOTS ==================
def generate_plots():
    os.makedirs("static/plots", exist_ok=True)
    if not os.path.exists("training_history.pkl"):
        print("⚠️ training_history.pkl not found.")
        return
    with open("training_history.pkl", "rb") as f:
        history = pickle.load(f)

    # Accuracy
    plt.figure()
    plt.plot(history["accuracy"], label="train_accuracy")
    plt.plot(history["val_accuracy"], label="val_accuracy")
    plt.title("Training & Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("static/plots/accuracy.png")
    plt.close()

    # Loss
    plt.figure()
    plt.plot(history["loss"], label="train_loss")
    plt.plot(history["val_loss"], label="val_loss")
    plt.title("Training & Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("static/plots/loss.png")
    plt.close()

@app.route('/accuracy_plot')
@login_required
def accuracy_plot():
    generate_plots()
    return redirect(url_for('static', filename='plots/accuracy.png'))

@app.route('/loss_plot')
@login_required
def loss_plot():
    generate_plots()
    return redirect(url_for('static', filename='plots/loss.png'))

# ================== ROUTES ==================
@app.route('/')
def home():
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        if User.query.filter_by(email=email).first():
            flash('Email already registered')
            return redirect(url_for('register'))
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        new_user = User(email=email, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        flash('Registration successful! Please log in.')
        return redirect(url_for('login'))
    return render_template('remainder/register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('modules'))
        else:
            flash('Invalid email or password')
            return redirect(url_for('login'))
    return render_template('remainder/login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/modules')
@login_required
def modules():
    return render_template("modules.html")

@app.route('/dashboard')
@login_required
def dashboard():
    appointments = Appointment.query.filter_by(email=current_user.email).order_by(Appointment.datetime).all()
    return render_template('remainder/dashboard.html', appointments=appointments)

@app.route('/add_appointment', methods=['GET', 'POST'])
@login_required
def add_appointment():
    if request.method == 'POST':
        patient_name = request.form['patient_name']
        appointment_time = datetime.strptime(request.form['datetime'], '%Y-%m-%dT%H:%M')
        new_appointment = Appointment(
            email=current_user.email,
            patient_name=patient_name,
            datetime=appointment_time
        )
        db.session.add(new_appointment)
        db.session.commit()
        schedule_appointment_reminder(new_appointment)
        flash("Appointment added successfully and reminder will be sent.")
        return redirect(url_for('dashboard'))
    return render_template('remainder/add_appointment.html')

# ================== BREAST CANCER PREDICTION ==================
@app.route('/cancer', methods=['GET', 'POST'])
@login_required
def cancer_index():
    if request.method == 'POST':
        if 'file' not in request.files or request.files['file'].filename == '':
            flash("Please upload a cancer test image!")
            return redirect(url_for('cancer_index'))
        file = request.files['file']
        upload_folder = os.path.join("static", "uploads")
        os.makedirs(upload_folder, exist_ok=True)
        file_path = os.path.join(upload_folder, file.filename)
        file.save(file_path)

        prediction_text = ""
        suggestion = ""
        email_msg = ""

        if model:
            img = image.load_img(file_path, target_size=(150, 150))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            prediction = model.predict(img_array)[0][0]
            email_msg = f"Dear {current_user.email},\n\n"
            if prediction > 0.5:
                prediction_text = f"Cancer Detected ⚠️ (Probability: {prediction:.2f})"
                suggestion = "Consult a doctor as soon as possible."
                email_msg += "⚠️ Cancer predicted. Please consult a doctor."
            else:
                prediction_text = f"No Cancer Detected ✅ (Probability: {prediction:.2f})"
                suggestion = "Maintain regular checkups and healthy lifestyle."
                email_msg += "✅ No cancer detected. Stay healthy."
        else:
            prediction_text = "Cancer prediction model not available."
            suggestion = ""
            email_msg = "Cancer prediction model not available."

        # Send immediate email
        send_email_reminder(current_user.email, email_msg)

        return render_template("cancer/result.html",
                               prediction_text=prediction_text,
                               uploaded_image=file_path,
                               suggestion=suggestion)
    return render_template("cancer/index.html")

# ================== MAIN ==================
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    print("✅ Flask app started")
    app.run(debug=True)
