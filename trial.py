from flask import Flask, render_template, redirect, request, session, url_for, send_file, flash, jsonify
import face_recognition
import cv2
import os
import pickle
from datetime import datetime, timedelta, date
import numpy as np
import csv
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import io
import time as time_module
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
from google.oauth2.service_account import Credentials
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'secret'  # Required for sessions

# Google Sheets setup
SHEET_SCOPE = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
SHEET_CREDS = ServiceAccountCredentials.from_json_keyfile_name('attendance-sheets-credentials.json', SHEET_SCOPE)
SHEET_CLIENT = gspread.authorize(SHEET_CREDS)
ALL_SHEET = SHEET_CLIENT.open("Attendance_All").sheet1

# Google Drive setup
DRIVE_SCOPE = ['https://www.googleapis.com/auth/drive']
DRIVE_CREDS = Credentials.from_service_account_file('dataset.json', scopes=DRIVE_SCOPE)
DRIVE_SERVICE = build('drive', 'v3', credentials=DRIVE_CREDS)
FOLDER_ID = '1_fBIUmV9UHbBbREDzOlmlF_zM7u85oGh'  # Main folder ID
IMAGES_FOLDER_ID = '1vL8q8c7YkM5nD3kXzPqW2tR9sJ0uH4mN'  # Dedicated images folder ID

# Verify folder accessibility
def verify_folder_access(folder_id):
    try:
        DRIVE_SERVICE.files().get(fileId=folder_id).execute()
        logger.debug(f"Successfully accessed folder with ID: {folder_id}")
        return True
    except Exception as e:
        logger.error(f"Failed to access folder with ID {folder_id}: {e}")
        return False

# Check folder access on startup
if not verify_folder_access(FOLDER_ID):
    logger.error(f"Cannot access main folder {FOLDER_ID}. Check folder ID and permissions.")
if not verify_folder_access(IMAGES_FOLDER_ID):
    logger.error(f"Cannot access images folder {IMAGES_FOLDER_ID}. Check folder ID and permissions.")

ENCODINGS_FILE_NAME = 'face_encodings.pkl'

def upload_to_drive(file_path, file_name, folder_id):
    try:
        logger.debug(f"Uploading file {file_path} as {file_name} to folder {folder_id}")
        file_metadata = {
            'name': file_name,
            'parents': [folder_id]
        }
        media = MediaFileUpload(file_path, mimetype='image/jpeg')
        file = DRIVE_SERVICE.files().create(
            body=file_metadata,
            media_body=media,
            fields='id'
        ).execute()
        logger.info(f"Uploaded {file_name} to Google Drive with ID: {file.get('id')}")
        return file.get('id')
    except Exception as e:
        logger.error(f"Error uploading {file_name} to Google Drive folder {folder_id}: {e}")
        return None

def upload_user_face_data(name, face_encodings, images, folder_id):
    """
    Uploads user's face encodings and images to Google Drive.
    Returns True if successful, False otherwise.
    """
    try:
        # Create temporary directory for user data
        temp_dir = os.path.join('temp_user_data', name)
        os.makedirs(temp_dir, exist_ok=True)
        logger.debug(f"Created temporary directory for user data: {temp_dir}")

        # Save face encodings to temporary file
        encodings_path = os.path.join(temp_dir, f'{name}_encodings.pkl')
        with open(encodings_path, 'wb') as f:
            pickle.dump(face_encodings, f)
        
        # Upload encodings file
        encodings_file_id = upload_to_drive(encodings_path, f'{name}_encodings.pkl', folder_id)
        if not encodings_file_id:
            logger.error(f"Failed to upload face encodings for {name}")
            return False

        # Upload images
        for idx, image in enumerate(images):
            image_path = os.path.join(temp_dir, f'{name}_{idx+1}.jpg')
            cv2.imwrite(image_path, image)
            image_file_id = upload_to_drive(image_path, f'{name}_{idx+1}.jpg', folder_id)
            if not image_file_id:
                logger.error(f"Failed to upload image {idx+1} for {name}")
                return False

        # Clean up temporary directory
        import shutil
        shutil.rmtree(temp_dir)
        logger.debug(f"Removed temporary directory: {temp_dir}")
        
        logger.info(f"Successfully uploaded face data for {name} to Google Drive")
        return True
    except Exception as e:
        logger.error(f"Error uploading user face data for {name}: {e}")
        return False

def download_fromDrive(file_name, folder_id, destination_path):
    try:
        query = f"'{folder_id}' in parents and name = '{file_name}' and trashed=false"
        results = DRIVE_SERVICE.files().list(q=query, fields="files(id, name)").execute()
        files = results.get('files', [])
        if not files:
            logger.warning(f"No file named {file_name} found in Google Drive folder {folder_id}.")
            return False
        file_id = files[0]['id']
        request = DRIVE_SERVICE.files().get_media(fileId=file_id)
        with open(destination_path, 'wb') as f:
            downloader = MediaIoBaseDownload(f, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()
                logger.debug(f"Download progress for {file_name}: {int(status.progress() * 100)}%")
        logger.info(f"Downloaded {file_name} from Google Drive to {destination_path}")
        return True
    except Exception as e:
        logger.error(f"Error downloading {file_name} from Google Drive: {e}")
        return False

def list_drive_images(folder_id):
    try:
        query = f"'{folder_id}' in parents and mimeType contains 'image/' and trashed=false"
        results = DRIVE_SERVICE.files().list(q=query, fields="files(id, name)").execute()
        images = results.get('files', [])
        logger.debug(f"Found {len(images)} images in folder {folder_id}: {[img['name'] for img in images]}")
        return images
    except Exception as e:
        logger.error(f"Error listing images from Google Drive folder {folder_id}: {e}")
        return []

def load_encodings():
    temp_encodings_path = 'temp_face_encodings.pkl'
    if download_from_drive(ENCODINGS_FILE_NAME, FOLDER_ID, temp_encodings_path):
        try:
            with open(temp_encodings_path, 'rb') as f:
                encodings = pickle.load(f)
            logger.info(f"Loaded encodings for {len(encodings)} users: {list(encodings.keys())}")
            os.remove(temp_encodings_path)  # Clean up temporary file
            return encodings
        except Exception as e:
            logger.error(f"Error loading encodings from {temp_encodings_path}: {e}")
    logger.warning("No encodings file found in Google Drive or failed to load, starting fresh.")
    return {}

def save_encodings(data):
    temp_encodings_path = 'temp_face_encodings.pkl'
    try:
        with open(temp_encodings_path, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"Saved encodings locally to {temp_encodings_path}")
        file_id = upload_to_drive(temp_encodings_path, ENCODINGS_FILE_NAME, FOLDER_ID)
        if file_id:
            query = f"'{FOLDER_ID}' in parents and name = '{ENCODINGS_FILE_NAME}' and trashed=false"
            results = DRIVE_SERVICE.files().list(q=query, fields="files(id, name)").execute()
            files = results.get('files', [])
            for file in files:
                if file['id'] != file_id:
                    DRIVE_SERVICE.files().delete(fileId=file['id']).execute()
                    logger.info(f"Deleted old {ENCODINGS_FILE_NAME} with ID: {file['id']}")
            os.remove(temp_encodings_path)  # Clean up temporary file
            logger.info(f"Encodings saved to Google Drive for {len(data)} users: {list(data.keys())}")
        else:
            logger.error("Failed to upload encodings to Google Drive.")
    except Exception as e:
        logger.error(f"Error saving encodings: {e}")

def read_attendance_from_sheet():
    try:
        sheet_data = ALL_SHEET.get_all_values()
        if not sheet_data or sheet_data[0] == ['Name']:
            return []
        headers = sheet_data[0]
        attendance = []
        for row in sheet_data[1:]:
            name = row[0]
            for i in range(1, len(headers), 2):
                if i + 1 >= len(headers):
                    break
                date = headers[i].replace(' Attendance', '')
                time_range = row[i] if i < len(row) else ''
                hours = row[i + 1] if i + 1 < len(row) else ''
                if time_range:
                    in_time, out_time = parse_time_range(time_range)
                    status = 'Present' if in_time else 'Absent'
                    attendance.append([name, date, in_time, out_time, status, hours])
        return attendance
    except ConnectionResetError as e:
        logger.error(f"ConnectionResetError in read_attendance_from_sheet: {e}")
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
                headers.append(f"{date} Hours")

        existing_dates = {header.replace(' Attendance', '') for header in headers if header.endswith(' Attendance')}
        for date in set(r[1] for r in attendance):
            if date not in existing_dates:
                headers.append(f"{date} Attendance")
                headers.append(f"{date} Hours")

        if f"{today} Attendance" not in headers:
            headers.append(f"{today} Attendance")
            headers.append(f"{today} Hours")

        all_names = list(load_encodings().keys())
        updated_data = []
        for name in all_names:
            row = [name]
            for i in range(1, len(headers), 2):
                date = headers[i].replace(' Attendance', '')
                time_range = ''
                hours = ''
                for record in attendance:
                    if record[0] == name and record[1] == date:
                        in_time = record[2] if record[2] else ''
                        out_time = record[3] if record[3] else ''
                        if in_time and out_time:
                            time_range = f"{in_time} - {out_time}"
                            hours = calculate_hours(in_time, out_time)
                        elif in_time:
                            time_range = in_time
                row.append(time_range)
                row.append(hours)
            updated_data.append(row)

        ALL_SHEET.update('A1', [headers], value_input_option='RAW')
        if updated_data:
            ALL_SHEET.update('A2', updated_data, value_input_option='RAW')
    except ConnectionResetError as e:
        logger.error(f"ConnectionResetError in update_sheet: {e}")
        raise

def log_attendance(name, action):
    now = datetime.now()
    date_str = now.strftime('%d/%m/%Y')
    time_str = now.strftime('%H:%M:%S')

    attendance = read_attendance_from_sheet()
    today_records = [r for r in attendance if r[0] == name and r[1] == date_str]

    if not today_records:
        if action == 'checkin':
            attendance.append([name, date_str, time_str, '', 'Present', ''])
            update_sheet(attendance)
            return True
    else:
        last_record = today_records[-1]
        checkin_time = datetime.strptime(last_record[2], '%H:%M:%S') if last_record[2] else None
        checkout_time = datetime.strptime(last_record[3], '%H:%M:%S') if last_record[3] else None

        if action == 'checkout' and checkin_time and not checkout_time:
            time_since_checkin = now - datetime.combine(date.today(), checkin_time.time())
            if time_since_checkin >= timedelta(hours=7):
                last_record[3] = time_str
                last_record[4] = 'Present'
                update_sheet(attendance)
                return True
            return False
        elif action == 'checkin' and not checkout_time:
            if not checkin_time:
                last_record[2] = time_str
                last_record[4] = 'Present'
                update_sheet(attendance)
                return True
    return False

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
            logger.info(f"Found match: {name} with distance {distance}")
            return name, distance
        else:
            logger.warning(f"Best match {name} rejected: distance {distance} >= {strict_threshold}")
    
    logger.info("No valid match found for face.")
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
            return 'Invalid username or password. Try again.'

    return render_template('login.html')

@app.route('/admin_panel', methods=['GET', 'POST'])
def admin_panel():
    if session.get('user') != 'admin':
        return redirect('/')
    
    encodings = load_encodings()
    names = list(encodings.keys())
    attendance = read_attendance_from_sheet()
    today = datetime.now().strftime('%d/%m/%Y')
    initial_data = {name: {'checkin': '', 'checkout': '', 'status': 'Absent', 'allow_checkout': False, 'hours': ''} for name in names}
    
    todays_attendance = [r for r in attendance if r[1] == today]
    for name, date, checkin, checkout, status, hours in todays_attendance:
        checkin_formatted = checkin.split(':')[0] + ':' + checkin.split(':')[1] if checkin else ''
        checkout_formatted = checkout.split(':')[0] + ':' + checkout.split(':')[1] if checkout else ''
        initial_data[name] = {
            'checkin': checkin_formatted,
            'checkout': checkout_formatted,
            'status': status,
            'allow_checkout': bool(checkin and not checkout),
            'hours': hours
        }

    if request.method == 'POST' and 'force_checkout' in request.form:
        name = request.form['force_checkout']
        now = datetime.now().strftime('%H:%M:%S')
        for record in todays_attendance:
            if record[0] == name and not record[3]:
                record[3] = now
                record[4] = 'Present'
        try:
            update_sheet(attendance)
        except ConnectionResetError:
            flash("Network error: Could not update attendance. Please try again.", "error")
            return render_template('admin_panel.html', names=names, attendance=todays_attendance, initial_data=initial_data, today=today)
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

    existing_images = list_drive_images(IMAGES_FOLDER_ID)
    if request.method == 'POST':
        name = request.form['name']
        encodings = load_encodings()
        if name in encodings:
            return render_template('add_user.html', error="User already registered. Try a different name.", existing_images=existing_images)

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.error("Error: Could not open webcam.")
            return render_template('add_user.html', error="Error: Could not open webcam. Check camera connection and try again.", existing_images=existing_images)

        known_face_encodings = []
        captured_images = []
        image_count = 0
        logger.info(f"Starting image capture for {name}. Press 'q' to stop early.")

        while image_count < 20:
            ret, frame = cap.read()
            if not ret:
                logger.warning(f"Frame {image_count + 1}: Failed to capture frame.")
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)

            logger.debug(f"Frame {image_count + 1}: Detected {len(face_locations)} faces")
            if len(face_locations) > 0:
                face_encoding = face_recognition.face_encodings(rgb_frame, face_locations)[0]
                if face_encoding.shape != (128,):
                    logger.warning(f"Frame {image_count + 1}: Invalid encoding shape: {face_encoding.shape}")
                    continue
                known_face_encodings.append(face_encoding)
                captured_images.append(frame)
                image_count += 1

                for (top, right, bottom, left) in face_locations:
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            cv2.imshow("Capturing Images", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("Capture stopped by user (pressed 'q').")
                break

        cap.release()
        cv2.destroyAllWindows()
        logger.info(f"Capture complete. Total images: {image_count}, Total encodings: {len(known_face_encodings)}")

        if known_face_encodings:
            if len(known_face_encodings) < 5:
                return render_template('add_user.html', error="Insufficient face captures (less than 5). Try again with better lighting or more angles.", existing_images=existing_images)
            
            # Upload user face data to Google Drive
            if not upload_user_face_data(name, known_face_encodings, captured_images, IMAGES_FOLDER_ID):
                return render_template('add_user.html', error="Failed to upload user face data to Google Drive.", existing_images=existing_images)

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
                flash(f"User {name} registered successfully", "success")
                return redirect('/admin_panel')
            except ConnectionResetError as e:
                logger.error(f"ConnectionResetError in add_user: {e}")
                return render_template('add_user.html', error="Network error: Could not save user data. Please try again.", existing_images=existing_images)
            except Exception as e:
                logger.error(f"Error saving encodings: {e}")
                return render_template('add_user.html', error=f"Error saving encodings for {name}: {str(e)}", existing_images=existing_images)
        else:
            return render_template('add_user.html', error="No faces detected. Try again with better lighting, closer to the camera, or different angles.", existing_images=existing_images)

    return render_template('add_user.html', error=None, existing_images=existing_images)

@app.route('/user_panel', methods=['GET', 'POST'])
def user_panel():
    if session.get('user') == 'admin' or session.get('user') is None:
        return redirect('/')

    known_faces = load_encodings()
    existing_images = list_drive_images(IMAGES_FOLDER_ID)
    action = "Welcome, please start recognition"
    name = "Unknown"

    if request.method == 'POST':
        if 'start_recognition' in request.form:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                action = "Error: Could not open webcam."
                return render_template('user_panel.html', name=name, action=action, known_faces=known_faces, existing_images=existing_images)
            # Existing webcam capture logic...
        elif request.is_json:
            image_data = request.json.get('image')
            if image_data:
                import base64
                import io
                header, encoded = image_data.split(',', 1)
                img_data = base64.b64decode(encoded)
                img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
                rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_frame)
                if face_locations:
                    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                    for face_encoding in face_encodings:
                        match_result = find_best_match(face_encoding, known_faces)
                        if match_result:
                            name, _ = match_result
                            # Existing attendance logic...
                        else:
                            action = "Unknown user detected. Please register or contact the admin."
                else:
                    action = "No face detected."
                return jsonify({'name': name, 'action': action})

    return render_template('user_panel.html', name=name, action=action, known_faces=known_faces, existing_images=existing_images)

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

            query = f"'{IMAGES_FOLDER_ID}' in parents and name contains '{name}_' and trashed=false"
            results = DRIVE_SERVICE.files().list(q=query, fields="files(id, name)").execute()
            files = results.get('files', [])
            for file in files:
                DRIVE_SERVICE.files().delete(fileId=file['id']).execute()
                logger.info(f"Deleted {file['name']} from Google Drive")

            logger.info(f"Successfully deleted user: {name}")
            flash(f"User {name} deleted successfully", "success")
            return redirect('/admin_panel')
        except ConnectionResetError as e:
            logger.error(f"ConnectionResetError in delete_user: {e}")
            flash("Network error: Could not delete user from sheet. Please try again.", "error")
            return redirect('/admin_panel')
        except Exception as e:
            logger.error(f"Error deleting user {name}: {e}")
            flash(f"Error deleting user {name}: {str(e)}", "error")
            return redirect('/admin_panel')
    logger.warning(f"User not found: {name}")
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
    date_str = datetime.now().strftime('%d/%m/%Y')
    checkin = request.form.get(f'checkin_{name}', '').strip()
    checkout = request.form.get(f'checkout_{name}', '').strip()

    attendance = read_attendance_from_sheet()
    today_records = [r for r in attendance if r[0] == name and r[1] == date_str]
    updated = False

    if today_records:
        last_record = today_records[-1]
        if checkin and not last_record[2]:
            last_record[2] = checkin + ':00' if checkin else ''
            last_record[4] = 'Present'
            updated = True
        elif checkout and last_record[2] and not last_record[3]:
            checkin_time = datetime.strptime(last_record[2], '%H:%M:%S')
            time_since_checkin = datetime.now() - datetime.combine(date.today(), checkin_time.time())
            if time_since_checkin >= timedelta(hours=7) or session.get('user') == 'admin':
                last_record[3] = checkout + ':00' if checkout else ''
                last_record[4] = 'Present'
                updated = True
            else:
                flash("Checkout not allowed yet. Wait 7 hours or contact admin.", "error")
                return redirect('/admin_panel')
    else:
        attendance.append([name, date_str, checkin + ':00' if checkin else '', checkout + ':00' if checkout else '', 'Present' if checkin else 'Absent', ''])

    if updated or not today_records:
        try:
            update_sheet(attendance)
        except ConnectionResetError as e:
            logger.error(f"ConnectionResetError intokenmark_attendance: {e}")
            flash("Network error: Could not update attendance. Please try again.", "error")
            return redirect('/admin_panel')

    return redirect('/admin_panel')

@app.route('/download_attendance')
def download_attendance():
    if session.get('user') != 'admin':
        return redirect('/')
    
    attendance = read_attendance_from_sheet()
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['Name', 'Date', 'Time Range', 'Status', 'Hours'])
    for record in attendance:
        name, date, in_time, out_time, status, hours = record
        time_range = f"{in_time} - {out_time}" if in_time and out_time else (in_time or out_time)
        writer.writerow([name, date, time_range, status6952status, hours])
    output.seek(0)
    return send_file(
        io.BytesIO(output.getvalue().encode()),
        as_attachment=True,
        download_name='attendance_records.csv',
        mimetype='text/csv'
    )

@app.route('/history', methods=['GET', 'POST'])
def history():
    if session.get('user') != 'admin':
        return redirect('/')
    
    selected_date = None
    attendance_records = []

    if request.method == 'POST':
        selected_date = request.form.get('date')
        if selected_date:
            formatted_date = datetime.strptime(selected_date, '%Y-%m-%d').strftime('%d/%m/%Y')
            headers = ALL_SHEET.row_values(1)
            if f"{formatted_date} Attendance" in headers:
                col_idx = headers.index(f"{formatted_date} Attendance") + 1
                hours_col_idx = col_idx + 1
                all_data = ALL_SHEET.get_all_values()[1:] or []
                for row in all_data:
                    name = row[0]
                    time_range = row[col_idx - 1] if col_idx <= len(row) else ''
                    hours = row[hours_col_idx - 1] if hours_col_idx <= len(row) else ''
                    if time_range:
                        in_time, out_time = parse_time_range(time_range)
                        status = 'Present' if in_time else 'Absent'
                        time_display = f"{in_time} - {out_time}" if in_time and out_time else (in_time or out_time or 'N/A')
                        attendance_records.append({
                            'name': name,
                            'status': status,
                            'time': time_display
                        })

    return render_template('history.html', selected_date=selected_date, attendance_records=attendance_records)

def get_checkin_image_base64(name,date_str):
    image_path= os.path.join(TEMP_)

if __name__ == '__main__':
    app.run(debug=True)