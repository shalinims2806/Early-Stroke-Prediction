from flask import Flask, jsonify
import json

app = Flask(__name__)

@app.route('/json')
def json_endpoint():
    return jsonify({"test": "success"})

@app.route('/text')
def text_endpoint():
    return "This is text"

@app.route('/custom')
def custom_endpoint():
    response = app.response_class(
        response=json.dumps({"custom": "response"}),
        status=200,
        mimetype='application/json'
    )
    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True) 