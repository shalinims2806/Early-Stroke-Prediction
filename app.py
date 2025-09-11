from flask import Flask, render_template, request, jsonify, redirect, url_for, session
import pickle
import numpy as np
import pandas as pd
import os
import json
from flask_cors import CORS
from functools import wraps

static_folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
app = Flask(__name__, 
    static_folder=static_folder_path,
    template_folder='templates')
CORS(app)  # Enable CORS for all routes

# Set a secret key for session management
app.secret_key = os.urandom(24)

# Check if model exists
model_path = 'model/stroke_model.pkl'
model = None

try:
    if os.path.exists(model_path):
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
    else:
        print(f"Warning: Model file not found at {model_path}")
except Exception as e:
    print(f"Error loading model: {e}")

# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Sample user credentials (in a real app, this would be in a database)
USERS = {
    'admin': 'password123',
    'user': 'user123'
}

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return render_template('login.html')
    
    if request.method == 'POST':
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        
        if username in USERS and USERS[username] == password:
            session['user'] = username
            return jsonify({'success': True})
        else:
            return jsonify({'success': False, 'message': 'Invalid username or password'})

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

@app.route('/')
@login_required
def home():
    return render_template('index.html')

@app.route('/test')
@login_required
def test():
    response = app.response_class(
        response=json.dumps({"status": "working"}),
        status=200,
        mimetype='application/json'
    )
    # Explicitly set the content type header
    response.headers.set('Content-Type', 'application/json')
    return response

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    print("Predict endpoint called")
    try:
        # Get data from request
        data = request.get_json()
        print(f"Received data: {data}")
        
        if not data:
            print("No data received in request")
            return jsonify({
                'error': 'No data received',
                'stroke_probability': 0,
                'risk_level': 'Unknown'
            }), 400
        
        # For now, return a dummy prediction
        # We'll use age as a simple risk factor (just for demonstration)
        age = float(data['age'])
        bmi = float(data['bmi'])
        glucose = float(data['avg_glucose_level'])
        
        print(f"Processing values: age={age}, bmi={bmi}, glucose={glucose}")
        
        # Very simple risk calculation (not a real model)
        risk_factors = 0
        
        if age > 65:
            risk_factors += 3
        elif age > 45:
            risk_factors += 1
            
        if bmi > 30:
            risk_factors += 1
            
        if glucose > 140:
            risk_factors += 2
            
        if data['hypertension'] == '1':
            risk_factors += 2
            
        if data['heart_disease'] == '1':
            risk_factors += 2
            
        if data['smoking_status'] == 'smokes':
            risk_factors += 1
        
        # Calculate probability from risk factors (simplified)
        probability = min(0.9, risk_factors / 10.0)
        
        result = {
            'stroke_probability': round(float(probability), 3),
            'risk_level': 'High' if probability > 0.5 else 'Low'
        }
        print(f"Returning result: {result}")
        
        response = app.response_class(
            response=json.dumps(result),
            status=200,
            mimetype='application/json'
        )
        return response
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({
            'error': str(e),
            'stroke_probability': 0,
            'risk_level': 'Unknown'
        }), 500

@app.route('/test-page-with-external-js')
@login_required
def test_page_with_external_js():
    return render_template('test_external.html')

@app.route('/simple-predict', methods=['GET'])
@login_required
def simple_predict():
    print("Simple predict endpoint called")
    try:
        data = {
            'stroke_probability': 0.25,
            'risk_level': 'Low'
        }
        # Make sure the response is properly formatted as JSON
        response = app.response_class(
            response=json.dumps(data),
            status=200,
            mimetype='application/json'
        )
        return response
    except Exception as e:
        print(f"Error in simple_predict: {e}")
        return app.response_class(
            response=json.dumps({'error': str(e)}),
            status=500,
            mimetype='application/json'
        )

@app.route('/simple-test')
@login_required
def simple_test():
    return render_template('simple_test.html')

@app.route('/api-test', methods=['GET'])
@login_required
def api_test():
    return "API is working"

@app.route('/fetch-test')
@login_required
def fetch_test():
    return render_template('fetch_test.html')

@app.route('/json-test', methods=['GET'])
@login_required
def json_test():
    response = app.response_class(
        response=json.dumps({"test": "success"}),
        status=200,
        mimetype='application/json'
    )
    # Explicitly set header to ensure it's not overridden
    response.headers.set('Content-Type', 'application/json')
    return response

@app.route('/xhr-test')
@login_required
def xhr_test():
    return render_template('xhr_test.html')

@app.route('/fetch-basic-test')
@login_required
def fetch_basic_test():
    return render_template('fetch_basic_test.html')

@app.route('/plain-text')
@login_required
def plain_text():
    return "This is plain text"

@app.route('/json-test-simple')
@login_required
def json_test_simple():
    return {'simple': 'json'}  # Flask can auto-convert dictionaries to JSON responses

@app.route('/test-predict', methods=['GET'])
@login_required
def test_predict():
    # Sample data
    sample_data = {
        'gender': 'Male',
        'age': '65',
        'hypertension': '1',
        'heart_disease': '1',
        'ever_married': 'Yes',
        'work_type': 'Private',
        'residence_type': 'Urban',
        'avg_glucose_level': '150',
        'bmi': '30',
        'smoking_status': 'smokes'
    }
    
    # Use the same logic as in the predict function
    age = float(sample_data['age'])
    bmi = float(sample_data['bmi'])
    glucose = float(sample_data['avg_glucose_level'])
    
    risk_factors = 0
    
    if age > 65:
        risk_factors += 3
    elif age > 45:
        risk_factors += 1
        
    if bmi > 30:
        risk_factors += 1
        
    if glucose > 140:
        risk_factors += 2
        
    if sample_data['hypertension'] == '1':
        risk_factors += 2
        
    if sample_data['heart_disease'] == '1':
        risk_factors += 2
        
    if sample_data['smoking_status'] == 'smokes':
        risk_factors += 1
    
    probability = min(0.9, risk_factors / 10.0)
    
    return jsonify({
        'stroke_probability': round(float(probability), 3),
        'risk_level': 'High' if probability > 0.5 else 'Low'
    })

@app.route('/endpoint-test')
@login_required
def endpoint_test():
    return render_template('endpoint_test.html')

@app.route('/simple-form')
@login_required
def simple_form():
    return render_template('simple_form.html')

@app.route('/simple-form-predict')
@login_required
def simple_form_predict():
    try:
        # Get all parameters with defaults
        age = float(request.args.get('age', 0))
        bmi = float(request.args.get('bmi', 0))
        glucose = float(request.args.get('glucose', 0))
        hypertension = int(request.args.get('hypertension', 0))
        heart_disease = int(request.args.get('heart_disease', 0))
        smoking = request.args.get('smoking', 'never smoked')
        
        # Calculate risk score
        risk_factors = 0
        
        if age > 65:
            risk_factors += 3
        elif age > 45:
            risk_factors += 1
            
        if bmi > 30:
            risk_factors += 1
            
        if glucose > 140:
            risk_factors += 2
            
        if hypertension == 1:
            risk_factors += 2
            
        if heart_disease == 1:
            risk_factors += 2
            
        if smoking == 'smokes':
            risk_factors += 1
        
        # Calculate probability (max 0.9 or 90%)
        probability = min(0.9, risk_factors / 10.0)
        
        result = {
            'stroke_probability': round(probability, 3),
            'risk_level': 'High' if probability > 0.5 else 'Low'
        }
        
        return app.response_class(
            response=json.dumps(result),
            status=200,
            mimetype='application/json',
            headers={'Content-Type': 'application/json'}
        )
    except Exception as e:
        print(f"Error in simple form predict: {e}")
        return app.response_class(
            response=json.dumps({'error': str(e)}),
            status=500,
            mimetype='application/json',
            headers={'Content-Type': 'application/json'}
        )

@app.route('/endpoint-direct-test')
@login_required
def endpoint_direct_test():
    return render_template('endpoint_direct_test.html')

@app.route('/xhr-form')
@login_required
def xhr_form():
    return render_template('xhr_form.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) 