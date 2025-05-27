from flask import Flask, render_template, redirect, request, session, url_for, send_file, flash, jsonify
import face_recognition
import cv2
import os
import pickle
from datetime import datetime, timedelta, date, time
import numpy as np
import csv
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import io
import base64
from PIL import Image
from pytz import timezone
from dotenv import load_dotenv
import json
from google.oauth2.service_account import Credentials
import shutil

load_dotenv()

credential_json = os.getenv("GOOGLE_CREDENTIALS_JSON")
app = Flask(__name__)
app.secret_key = 'secret'  # Required for sessions

ENCODINGS_FILE = 'face_encodings.pkl'
IMAGES_DIR = 'images'
TEMP_CHECKIN_IMAGES_DIR = 'temp_checkin_images'  # New directory for check-in images

# Create temporary check-in images directory if it doesn't exist
if not os.path.exists(TEMP_CHECKIN_IMAGES_DIR):
    os.makedirs(TEMP_CHECKIN_IMAGES_DIR)

# Google Sheets setup
SCOPE = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
credential_dict = json.loads(credential_json)
CREDS = Credentials.from_service_account_info(credential_dict, scopes=SCOPE)
CLIENT = gspread.authorize(CREDS)
ALL_SHEET = CLIENT.open("Tars_Attendancee").sheet1

def load_encodings():
    if os.path.exists(ENCODINGS_FILE):
        with open(ENCODINGS_FILE, 'rb') as f:
            encodings = pickle.load(f)
        print(f"Loaded encodings for {len(encodings)} users: {list(encodings.keys())}")
        return encodings
    print("No encodings file found, starting fresh.")
    return {}

def save_encodings(data):
    with open(ENCODINGS_FILE, 'wb') as f:
        pickle.dump(data, f)
    print(f"Saved encodings for {len(data)} users: {list(data.keys())}")

def read_attendance_from_sheet():
    try:
        sheet_data = ALL_SHEET.get_all_values()
        if not sheet_data or sheet_data[0] == ['Name']:
            return []
        headers = sheet_data[0]
        attendance = []
        for row in sheet_data[1:]:
            name = row[0]
            for i in range(1, len(headers), 4):
                if i + 3 >= len(headers):
                    break
                date = headers[i].replace(' Attendance', '')
                time_range = row[i] if i < len(row) else ''
                expected_checkout = row[i + 1] if i + 1 < len(row) else ''
                hours = row[i + 2] if i + 2 < len(row) else ''
                day_status = row[i + 3] if i + 3 < len(row) else ''
                if time_range:
                    in_time, out_time = parse_time_range(time_range)
                    status = 'Present' if in_time else 'Absent'
                    attendance.append([name, date, in_time, out_time, status, expected_checkout, hours, day_status])
        return attendance
    except Exception as e:
        print(f"Error in read_attendance_from_sheet: {e}")
        return []

def parse_time_range(time_range):
    if '-' in time_range:
        in_time, out_time = time_range.split(' - ')
        return in_time, out_time
    return time_range, ''

def calculate_hours(in_time, out_time):
    if not in_time or not out_time:
        return ''
    try:
        in_time_dt = datetime.strptime(in_time, '%H:%M:%S')
        out_time_dt = datetime.strptime(out_time, '%H:%M:%S')
        time_diff = out_time_dt - in_time_dt
        if time_diff.total_seconds() < 0:
            time_diff += timedelta(days=1)
        hours = time_diff.total_seconds() / 3600
        return f"{hours:.2f}"
    except ValueError:
        return ''

def update_sheet(attendance):
    try:
        today = datetime.now().strftime('%d/%m/%Y')
        headers = ALL_SHEET.row_values(1)
        if not headers or headers == ['Name']:
            headers = ['Name']
        else:
            existing_dates = set()
            for header in headers[1:]:
                if header.endswith(' Attendance'):
                    existing_dates.add(header.replace(' Attendance', ''))
            headers = ['Name']
            for date in sorted(existing_dates):
                headers.append(f"{date} Attendance")
                headers.append(f"{date} Check-out")
                headers.append(f"{date} Hours")
                headers.append(f"{date} Day Status")

        existing_dates = {header.replace(' Attendance', '') for header in headers if header.endswith(' Attendance')}
        for date in set(r[1] for r in attendance):
            if date not in existing_dates:
                headers.append(f"{date} Attendance")
                headers.append(f"{date} Check-out")
                headers.append(f"{date} Hours")
                headers.append(f"{date} Day Status")

        if f"{today} Attendance" not in headers:
            headers.append(f"{today} Attendance")
            headers.append(f"{today} Check-out")
            headers.append(f"{today} Hours")
            headers.append(f"{today} Day Status")

        all_names = list(load_encodings().keys())
        updated_data = []
        for name in all_names:
            row = [name]
            for i in range(1, len(headers), 4):
                date = headers[i].replace(' Attendance', '')
                time_range = ''
                check_out = ''
                hours = ''
                day_status = ''
                for record in attendance:
                    if record[0] == name and record[1] == date:
                        in_time = record[2] if record[2] else ''
                        out_time = record[3] if record[3] else ''
                        day_status = record[7] if len(record) > 7 else ''
                        if in_time and out_time:
                            time_range = f"{in_time} - {out_time}"
                            hours = calculate_hours(in_time, out_time)
                        elif in_time:
                            time_range = in_time
                        check_out = out_time
                row.append(time_range)
                row.append(check_out)
                row.append(hours)
                row.append(day_status)
            updated_data.append(row)

        ALL_SHEET.update('A1', [headers], value_input_option='RAW')
        if updated_data:
            ALL_SHEET.update('A2', updated_data, value_input_option='RAW')
    except Exception as e:
        print(f"Error in update_sheet: {e}")
        raise

def log_attendance(name, action):
    now = datetime.now()
    date_str = now.strftime('%d/%m/%Y')
    time_str = now.strftime('%H:%M:%S')

    attendance = read_attendance_from_sheet()
    today_records = [r for r in attendance if r[0] == name and r[1] == date_str]

    if not today_records:
        if action == 'checkin':
            checkin_time = datetime.strptime(time_str, '%H:%M:%S').time()
            checkin_dt = datetime.combine(datetime.today(), checkin_time)
            time_10_00 = datetime.combine(datetime.today(), time(10, 0))
            time_10_30 = datetime.combine(datetime.today(), time(10, 30))
            time_11_00 = datetime.combine(datetime.today(), time(11, 0))
            day_status = 'Full Day'
            expected_checkout = '18:30:00'  # Default 6:30 PM

            print(f'Helo world')
            if time_10_00 <= checkin_dt < time_10_30:
                # Check-in between 10:00 and 10:30 → 6:30 PM checkout, Full Day
                expected_checkout = '18:30:00'
            elif time_10_30 <= checkin_dt <= time_11_00:
                # Check-in between 10:31 and 11:00 → 8 hours after check-in, Full Day
                checkout_dt = checkin_dt + timedelta(hours=8)
                expected_checkout = checkout_dt.strftime('%H:%M:%S')
            elif checkin_dt > time_11_00:
                # Check-in after 11:00 → 6:30 PM checkout, Half Day
                expected_checkout = '18:30:00'
                day_status = 'Half Day'

            attendance.append([name, date_str, time_str, '', 'Present', expected_checkout, '', day_status])
            update_sheet(attendance)
            return True, expected_checkout, day_status
    else:
        last_record = today_records[-1]
        checkin_time = datetime.strptime(last_record[2], '%H:%M:%S') if last_record[2] else None
        checkout_time = datetime.strptime(last_record[3], '%H:%M:%S') if last_record[3] else None
        day_status = last_record[7] if len(last_record) > 7 else 'Full Day'

        if action == 'checkout' and checkin_time and not checkout_time:
            time_since_checkin = now - datetime.combine(date.today(), checkin_time.time())
            expected_checkout = datetime.strptime(last_record[5], '%H:%M:%S') if last_record[5] else datetime.strptime('18:30:00', '%H:%M:%S')
            if time_since_checkin >= timedelta(hours=7):
                last_record[3] = time_str
                last_record[4] = 'Present'
                last_record[6] = calculate_hours(last_record[2], time_str)
                update_sheet(attendance)
                return True, last_record[6], day_status
            return False, last_record[5], day_status
        elif action == 'checkin' and not checkout_time:
            if not checkin_time:
                last_record[2] = time_str
                last_record[4] = 'Present'
                checkin_dt = datetime.strptime(time_str, '%H:%M:%S')
                time_10_00 = datetime.combine(datetime.today(), time(10, 0))
                time_10_30 = datetime.combine(datetime.today(), time(10, 30))
                time_11_00 = datetime.combine(datetime.today(), time(11, 0))
                day_status = 'Full Day'
                expected_checkout = '18:30:00'

                if time_10_00 <= checkin_dt < time_10_30:
                    expected_checkout = '18:30:00'
                elif time_10_30 <= checkin_dt <= time_11_00:
                    expected_checkout = (checkin_dt + timedelta(hours=8)).strftime('%H:%M:%S')
                elif checkin_dt > time_11_00:
                    expected_checkout = '18:30:00'
                    day_status = 'Half Day'

                last_record[5] = expected_checkout
                last_record[7] = day_status
                update_sheet(attendance)
                return True, expected_checkout, day_status
    return False, '', day_status

def find_best_match(face_encoding, known_faces, tolerance=0.5, strict_threshold=0.45):
    matches = []
    for current_name, known_encodings in known_faces.items():
        distances = face_recognition.face_distance(known_encodings, face_encoding)
        for i, distance in enumerate(distances):
            if distance < tolerance:
                matches.append((current_name, distance))
    
    if matches:
        best_match = min(matches, key=lambda x: x[1])
        name, distance = best_match
        if distance < strict_threshold:
            print(f"Found match: {name} with distance {distance}")
            return name, distance
        else:
            print(f"Best match {name} rejected: distance {distance} >= {strict_threshold}")
    
    print("No valid match found for face.")
    return None

def get_checkin_image_base64(name, date_str):
    """Retrieve the check-in image for a user on a specific date as base64."""
    image_path = os.path.join(TEMP_CHECKIN_IMAGES_DIR, f"{name}_{date_str.replace('/', '-')}.jpg")
    if os.path.exists(image_path):
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            return f"data:image/jpeg;base64,{encoded_string}"
    return None

@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username == 'admin' and password == 'admin123':
            session['user'] = 'admin'
            return redirect('/admin_panel')
        elif username == 'user' and password == 'user123':
            session['user'] = 'user'
            return redirect('/user_panel')
        else:
            flash("Invalid username or password.")
            return redirect('/')

    return render_template('login.html')

@app.route('/admin_panel', methods=['GET', 'POST'])
def admin_panel():
    if session.get('user') != 'admin':
        return redirect('/')
    
    encodings = load_encodings()
    names = list(encodings.keys())
    attendance = read_attendance_from_sheet()
    today = datetime.now().strftime('%d/%m/%Y')
    initial_data = {name: {
        'checkin': '',
        'checkout': '',
        'status': 'Absent',
        'allow_checkout': False,
        'hours': '',
        'checkin_image': None,
        'day_status': 'N/A'
    } for name in names}
    
    todays_attendance = [r for r in attendance if r[1] == today]
    for name, date, checkin, checkout, status, expected_checkout, hours, day_status in todays_attendance:
        checkin_formatted = checkin.split(':')[0] + ':' + checkin.split(':')[1] if checkin else ''
        checkout_formatted = checkout.split(':')[0] + ':' + checkout.split(':')[1] if checkout else ''
        initial_data[name] = {
            'checkin': checkin_formatted,
            'checkout': checkout_formatted,
            'status': status,
            'allow_checkout': bool(checkin and not checkout),
            'hours': hours,
            'checkin_image': get_checkin_image_base64(name, today),
            'day_status': day_status if day_status else 'N/A'
        }

    if request.method == 'POST' and 'force_checkout' in request.form:
        name = request.form['force_checkout']
        now = datetime.now().strftime('%H:%M:%S')
        for record in todays_attendance:
            if record[0] == name and not record[3]:
                record[3] = now
                record[4] = 'Present'
                record[6] = calculate_hours(record[2], now)
        try:
            update_sheet(attendance)
            image_path = os.path.join(TEMP_CHECKIN_IMAGES_DIR, f"{name}_{today.replace('/', '-')}.jpg")
            if os.path.exists(image_path):
                os.remove(image_path)
                print(f"Deleted check-in image: {image_path}")
            flash("Checkout forced successfully.", "success")
        except Exception as e:
            print(f"Error forcing checkout: {e}")
            flash("Network error: Could not update attendance. Please try again.", "error")
        return redirect('/admin_panel')
        
    return render_template('admin_panel.html', 
                          names=names, 
                          attendance=todays_attendance,  
                          initial_data=initial_data,
                          today=today)

@app.route('/add_user', methods=['GET', 'POST'])
def add_user():
    if session.get('user') != 'admin':
        return redirect('/')

    if request.method == 'POST':
        try:
            if not request.is_json:
                return jsonify({'error': 'Invalid request. JSON data required.'}), 400

            data = request.get_json()
            name = data.get('name')
            images = data.get('images', [])

            if not name or not name.strip():
                return jsonify({'error': 'Name is required.'}), 400
            if not images:
                return jsonify({'error': 'No images provided.'}), 400

            encodings = load_encodings()
            if name in encodings:
                return jsonify({'error': 'User already registered. Try a different name.'}), 400

            person_dir = os.path.join(IMAGES_DIR, name)
            if not os.path.exists(person_dir):
                try:
                    os.makedirs(person_dir)
                    print(f"Created directory: {person_dir}")
                except Exception as e:
                    print(f"Error creating directory {person_dir}: {e}")
                    return jsonify({'error': f'Error creating directory for {name}: {str(e)}'}), 500

            known_face_encodings = []
            image_count = 0

            for image_data in images:
                try:
                    if ',' in image_data:
                        image_data = image_data.split(',')[1]
                    image_bytes = base64.b64decode(image_data)
                    image = Image.open(io.BytesIO(image_bytes))
                    frame = np.array(image)

                    if frame.shape[-1] == 4:
                        frame = frame[:, :, :3]
                    rgb_frame = frame

                    face_locations = face_recognition.face_locations(rgb_frame)
                    print(f"Image {image_count + 1}: Detected {len(face_locations)} faces")

                    if len(face_locations) == 0:
                        print(f"Image {image_count + 1}: No faces detected")
                        continue
                    if len(face_locations) > 1:
                        print(f"Image {image_count + 1}: Multiple faces detected, skipping")
                        continue

                    face_encoding = face_recognition.face_encodings(rgb_frame, face_locations)[0]
                    if face_encoding.shape != (128,):
                        print(f"Image {image_count + 1}: Invalid encoding shape: {face_encoding.shape}")
                        continue

                    known_face_encodings.append(face_encoding)
                    image_count += 1
                    image_path = os.path.join(person_dir, f'{name}_{image_count}.jpg')
                    try:
                        Image.fromarray(frame).save(image_path)
                        print(f"Saved image {image_count}: {image_path}")
                    except Exception as e:
                        print(f"Error saving image {image_path}: {e}")
                        continue

                except Exception as e:
                    print(f"Error processing image {image_count + 1}: {e}")
                    continue

            print(f"Capture complete. Total images: {image_count}, Total encodings: {len(known_face_encodings)}")

            if known_face_encodings:
                if len(known_face_encodings) < 5:
                    return jsonify({'error': 'Insufficient face captures (less than 5). Try again with better lighting or more angles.'}), 400
                encodings[name] = known_face_encodings
                try:
                    save_encodings(encodings)
                    headers = ALL_SHEET.row_values(1) or ['Name']
                    all_data = ALL_SHEET.get_all_values()[1:] or []
                    all_data.append([name] + [''] * (len(headers) - 1))
                    ALL_SHEET.clear()
                    ALL_SHEET.update('A1', [headers], value_input_option='RAW')
                    if all_data:
                        ALL_SHEET.update('A2', all_data, value_input_option='RAW')
                    return jsonify({'success': f'User {name} registered successfully'}), 200
                except Exception as e:
                    print(f"Error saving encodings or updating sheet: {e}")
                    return jsonify({'error': f'Network error: Could not save user data. Please try again.'}), 500
            else:
                return jsonify({'error': 'No valid faces detected. Try again with better lighting, closer to the camera, or different angles.'}), 400

        except Exception as e:
            print(f"Error in add_user: {e}")
            return jsonify({'error': f'Error processing request: {str(e)}'}), 500

    return render_template('add_user.html', error=None)

@app.route('/user_panel', methods=['GET', 'POST'])
def user_panel():
    if session.get('user') == 'admin' or session.get('user') is None:
        return redirect('/')

    known_faces = load_encodings()
    action = "Welcome, please start recognition"
    name = "Unknown"

    if request.method == 'POST':
        try:
            if not request.is_json:
                return jsonify({'action': 'Invalid request. Image data required.', 'name': name}), 400

            data = request.get_json()
            if 'image' not in data:
                return jsonify({'action': 'No image provided.', 'name': name}), 400

            image_data = data['image']
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            frame = np.array(image)

            if frame.shape[-1] == 4:
                frame = frame[:, :, :3]
            rgb_frame = frame

            face_locations = face_recognition.face_locations(rgb_frame)
            if face_locations:
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                for face_encoding in face_encodings:
                    match_result = find_best_match(face_encoding, known_faces, tolerance=0.5, strict_threshold=0.45)
                    if match_result:
                        name, best_distance = match_result
                        print(f"Recognized: {name} with distance {best_distance}")
                        attendance = read_attendance_from_sheet()
                        today = datetime.now().strftime('%d/%m/%Y')
                        today_records = [r for r in attendance if r[0] == name and r[1] == today]
                        now = datetime.now()

                        if not today_records:
                            noon = datetime.strptime(f"{today} 12:00:00", '%d/%m/%Y %H:%M:%S')
                            if now > noon:
                                action = "Check-in not allowed after 12:00 PM. Please contact the admin."
                            else:
                                success, expected_checkout, day_status = log_attendance(name, 'checkin')
                                if success:
                                    action = f"Checked in successfully. Expected check-out: {expected_checkout}. Day Status: {day_status}"
                                    image_path = os.path.join(TEMP_CHECKIN_IMAGES_DIR, f"{name}_{today.replace('/', '-')}.jpg")
                                    try:
                                        Image.fromarray(frame).save(image_path)
                                        print(f"Saved check-in image: {image_path}")
                                    except Exception as e:
                                        print(f"Error saving check-in image: {e}")
                        else:
                            last_record = today_records[-1]
                            checkin_time = datetime.strptime(last_record[2], '%H:%M:%S') if last_record[2] else None
                            checkout_time = datetime.strptime(last_record[3], '%H:%M:%S') if last_record[3] else None

                            if checkout_time:
                                action = 'Attendance complete for today'
                            elif checkin_time and not checkout_time:
                                time_since_checkin = now - datetime.combine(date.today(), checkin_time.time())
                                expected_checkout = datetime.strptime(last_record[5], '%H:%M:%S') if last_record[5] else datetime.strptime('18:30:00', '%H:%M:%S')
                                if time_since_checkin >= timedelta(hours=7):
                                    success, hours, day_status = log_attendance(name, 'checkout')
                                    if success:
                                        action = f"Checked out successfully. Hours: {hours}. Day Status: {day_status}"
                                        image_path = os.path.join(TEMP_CHECKIN_IMAGES_DIR, f"{name}_{today.replace('/', '-')}.jpg")
                                        if os.path.exists(image_path):
                                            os.remove(image_path)
                                            print(f"Deleted check-in image: {image_path}")
                                    else:
                                        action = "Error processing check-out. Please try again."
                                else:
                                    time_to_checkout = datetime.combine(date.today(), expected_checkout.time()) - now
                                    if time_to_checkout.total_seconds() > 0:
                                        hours, remainder = divmod(time_to_checkout.total_seconds(), 3600)
                                        minutes, seconds = divmod(remainder, 60)
                                        action = f"Cannot check out yet, wait until {last_record[5]} ({int(hours)}h {int(minutes)}m {int(seconds)}s)"
                                    else:
                                        action = f"Cannot check out yet, minimum 7 hours required. {int(7 - time_since_checkin.total_seconds() / 3600)} hours remaining."
                            else:
                                action = 'Invalid attendance state'
                    else:
                        name = "Unknown"
                        action = "Unknown user. Please register or contact the admin."
            else:
                action = "No face detected. Ensure your face is visible and try again."

            return jsonify({'action': action, 'name': name})

        except Exception as e:
            print(f"Error in user_panel: {e}")
            return jsonify({'action': f"Error processing image: {str(e)}", 'name': name}), 500

    return render_template('user_panel.html', name=name, action=action, known_faces=known_faces)

@app.route('/delete_user', methods=['GET'])
def delete_user():
    if session.get('user') != 'admin':
        return redirect('/')

    name = request.args.get('name', '')
    if not name:
        flash("No user specified for deletion.", "error")
        return redirect('/admin_panel')

    encodings = load_encodings()
    if name in encodings:
        try:
            del encodings[name]
            save_encodings(encodings)
            all_data = ALL_SHEET.get_all_values()[1:] or []
            updated_data = [row for row in all_data if row[0] != name]
            headers = ALL_SHEET.row_values(1) or ['Name']
            ALL_SHEET.clear()
            ALL_SHEET.update('A1', [headers], value_input_option='RAW')
            if updated_data:
                ALL_SHEET.update('A2', updated_data, value_input_option='RAW')
            person_dir = os.path.join(IMAGES_DIR, name)
            if os.path.exists(person_dir):
                shutil.rmtree(person_dir)
            for image_file in os.listdir(TEMP_CHECKIN_IMAGES_DIR):
                if image_file.startswith(f"{name}_"):
                    os.remove(os.path.join(TEMP_CHECKIN_IMAGES_DIR, image_file))
            print(f"Successfully deleted user: {name}")
            flash(f"User {name} deleted successfully", "success")
            return redirect('/admin_panel')
        except Exception as e:
            print(f"Error deleting user {name}: {e}")
            flash(f"Error deleting user {name}: {str(e)}", "error")
            return redirect('/admin_panel')
    print(f"User not found: {name}")
    flash(f"User {name} not found.", "error")
    return redirect('/admin_panel')

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect('/')

@app.route('/mark_attendance', methods=['POST'])
def mark_attendance():
    name = request.form['mark_user']
    status = request.form[f'status_{name}']
    india = timezone('Asia/Kolkata')
    date_str = datetime.now(india).strftime('%d/%m/%Y')
    checkin = request.form.get(f'checkin_{name}', '').strip()
    checkout = request.form.get(f'checkout_{name}', '').strip()

    attendance = read_attendance_from_sheet()
    today_records = [r for r in attendance if r[0] == name and r[1] == date_str]
    updated = False

    if today_records:
        last_record = today_records[-1]
        if checkin and not last_record[2]:
            try:
                checkin_dt = datetime.strptime(checkin + ':00', '%H:%M:%S')
                time_10_00 = datetime.combine(datetime.today(), time(10, 0))
                time_10_30 = datetime.combine(datetime.today(), time(10, 30))
                time_11_00 = datetime.combine(datetime.today(), time(11, 0))
                day_status = 'Full Day'
                expected_checkout = '18:30:00'
                if time_10_00 <= checkin_dt < time_10_30:
                    expected_checkout = '18:30:00'
                elif time_10_30 <= checkin_dt <= time_11_00:
                    expected_checkout = (checkin_dt + timedelta(hours=8)).strftime('%H:%M:%S')
                elif checkin_dt > time_11_00:
                    expected_checkout = '18:30:00'
                    day_status = 'Half Day'
                last_record[2] = checkin + ':00'
                last_record[4] = 'Present'
                last_record[5] = expected_checkout
                last_record[7] = day_status
                updated = True
            except ValueError:
                flash("Invalid check-in time format. Use HH:MM.", "error")
                return redirect('/admin_panel')
        elif checkout and last_record[2] and not last_record[3]:
            try:
                checkin_time = datetime.strptime(last_record[2], '%H:%M:%S')
                checkout_dt = datetime.strptime(checkout + ':00', '%H:%M:%S')
                time_since_checkin = datetime.now(india) - datetime.combine(date.today(), checkin_time.time())
                expected_checkout = datetime.strptime(last_record[5], '%H:%M:%S') if last_record[5] else datetime.strptime('18:30:00', '%H:%M:%S')
                if time_since_checkin >= timedelta(hours=7) or session.get('user') == 'admin':
                    last_record[3] = checkout + ':00'
                    last_record[4] = 'Present'
                    last_record[6] = calculate_hours(last_record[2], checkout + ':00')
                    updated = True
                else:
                    flash("Checkout not allowed yet. Wait until 7 hours have passed or contact admin.", "error")
                    return redirect('/admin_panel')
            except ValueError:
                flash("Invalid check-out time format. Use HH:MM.", "error")
                return redirect('/admin_panel')
    else:
        try:
            day_status = 'Full Day'
            expected_checkout = '18:30:00'
            checkin_time = checkin + ':00' if checkin else ''
            checkout_time = checkout + ':00' if checkout else ''
            print(f'   Hello world')
            if checkin:
                checkin_dt = datetime.strptime(f'{datetime.today().strftime("%Y-%m-%d")} {checkin}:00', '%Y-%m-%d %H:%M:%S')
                time_10_00 = datetime.combine(datetime.today(), time(10, 0))
                time_10_30 = datetime.combine(datetime.today(), time(10, 30))
                time_11_00 = datetime.combine(datetime.today(), time(11, 0))
                if time_10_00 <= checkin_dt < time_10_30:
                    expected_checkout = '18:30:00'
                elif time_10_30 <= checkin_dt <= time_11_00:
                    expected_checkout = (checkin_dt + timedelta(hours=8)).strftime('%H:%M:%S')
                elif checkin_dt > time_11_00:
                    expected_checkout = '18:30:00'
                    day_status = 'Half Day'
            hours = calculate_hours(checkin_time, checkout_time) if checkin_time and checkout_time else ''
            print(f'{day_status}')
            attendance.append([name, date_str, checkin_time, checkout_time, 'Present' if checkin_time else 'Absent', expected_checkout, hours, day_status])
            updated = True
        except ValueError:
            flash("Invalid time format. Use HH:MM for check-in/check-out.", "error")
            return redirect('/admin_panel')

    if updated:
        try:
            update_sheet(attendance)
            flash("Attendance updated successfully.", "success")
        except Exception as e:
            print(f"Error in mark_attendance: {e}")
            flash("Network error: Could not update attendance. Please try again.", "error")
            return redirect('/admin_panel')

    return redirect('/admin_panel')

# @app.route('/download_attendance')
# def download_attendance():
#     if session.get('user') != 'admin':
#         return redirect('/')
    
#     attendance = read_attendance_from_sheet()
#     output = io.StringIO()
#     writer = csv.writer(output)
#     writer.writerow(['Name', 'Date', 'Time Range', 'Status', 'Hours', 'Day Status'])
#     for record in attendance:
#         name, date, in_time, out_time, status, _, hours, day_status = record
#         time_range = f"{in_time} - {out_time}" if in_time and out_time else (in_time or out_time)
#         writer.writerow([name, date, time_range, status, hours, day_status])
#     output.seek(0)
#     return send_file(
#         io.BytesIO(output.getvalue().encode()),
#         as_attachment=True,
#         download_name='attendance_records.csv',
#         mimetype='text/csv'
#     )

from flask import send_file, redirect, session
import io
import pandas as pd

@app.route('/download_attendance')
def download_attendance():
    if session.get('user') != 'admin':
        return redirect('/')
    
    attendance = read_attendance_from_sheet()
    rows = []
    for record in attendance:
        name, date, in_time, out_time, status, _, hours, day_status = record
        time_range = f"{in_time} - {out_time}" if in_time and out_time else (in_time or out_time)
        rows.append([name, date, time_range, status, hours, day_status])
    
    df = pd.DataFrame(rows, columns=['Name', 'Date', 'Time Range', 'Status', 'Hours', 'Day Status'])

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Attendance')
    output.seek(0)

    return send_file(
        output,
        as_attachment=True,
        download_name='attendance_records.xlsx',
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )



@app.route('/history', methods=['GET', 'POST'])
def history():
    if session.get('user') != 'admin':
        return redirect('/')
    
    selected_date = None
    selected_user = None
    attendance_records = []
    user_history = []
    registered_users = list(load_encodings().keys())  # Fetch list of registered users

    # Debug: Log the number of registered users
    if not registered_users:
        flash("No registered users found. Please add users via the 'Add User' page.", "warning")
    else:
        flash(f"Found {len(registered_users)} registered users.", "info")

    if request.method == 'POST':
        selected_date = request.form.get('date')
        selected_user = request.form.get('name')

        if selected_user:  # If a user is selected, fetch their full history
            headers = ALL_SHEET.row_values(1)
            all_data = ALL_SHEET.get_all_values()[1:] or []
            for row in all_data:
                if row[0] != selected_user:
                    continue
                name = row[0]
                for i in range(1, len(headers), 4):  # Step by 4 for each date block
                    if i + 3 >= len(headers):
                        break
                    date = headers[i].replace(' Attendance', '')
                    time_range = row[i] if i < len(row) else ''
                    hours = row[i + 2] if i + 2 < len(row) else ''
                    day_status = row[i + 3] if i + 3 < len(row) else ''
                    if time_range:
                        in_time, out_time = parse_time_range(time_range)
                        status = 'Present' if in_time else 'Absent'
                        time_display = f"{in_time} - {out_time}" if in_time and out_time else (in_time or out_time or 'N/A')
                        user_history.append({
                            'date': date,
                            'status': status,
                            'time': time_display,
                            'hours': hours,
                            'day_status': day_status
                        })
        elif selected_date:  # If only a date is selected, fetch records for that date
            formatted_date = datetime.strptime(selected_date, '%Y-%m-%d').strftime('%d/%m/%Y')
            headers = ALL_SHEET.row_values(1)
            if f"{formatted_date} Attendance" in headers:
                col_idx = headers.index(f"{formatted_date} Attendance") + 1
                hours_col_idx = col_idx + 2
                day_status_col_idx = col_idx + 3
                all_data = ALL_SHEET.get_all_values()[1:] or []
                for row in all_data:
                    name = row[0]
                    time_range = row[col_idx - 1] if col_idx <= len(row) else ''
                    hours = row[hours_col_idx - 1] if hours_col_idx <= len(row) else ''
                    day_status = row[day_status_col_idx - 1] if day_status_col_idx <= len(row) else ''
                    if time_range:
                        in_time, out_time = parse_time_range(time_range)
                        status = 'Present' if in_time else 'Absent'
                        time_display = f"{in_time} - {out_time}" if in_time and out_time else (in_time or out_time or 'N/A')
                        attendance_records.append({
                            'name': name,
                            'status': status,
                            'time': time_display,
                            'hours': hours,
                            'day_status': day_status
                        })

    return render_template('history.html', 
                          selected_date=selected_date, 
                          selected_user=selected_user,
                          attendance_records=attendance_records,
                          user_history=user_history,
                          registered_users=registered_users)

if __name__ == '__main__':
    app.run(debug=True)