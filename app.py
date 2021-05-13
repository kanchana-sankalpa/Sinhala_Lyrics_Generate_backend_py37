from flask import Flask, request, jsonify
from flask_cors import CORS
#from server import get_model_api
from server import get_tpu_model_api


# define the app
app = Flask(__name__)
app.config['DEBUG'] = False
CORS(app)  # needed for cross-domain requests, allow everything by default


# load the model
#model_api = get_model_api()
model_tpu_api = get_tpu_model_api()


# API route

@app.route('/')
def index():
    return "Index API"

# HTTP Errors handlers
@app.errorhandler(404)
def url_error(e):
    return """
    Wrong URL!
    <pre>{}</pre>""".format(e), 404


@app.errorhandler(500)
def server_error(e):
    return """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e), 500


""" @app.route('/apiUS', methods=['POST'])
def api():
    input_data = request.form['input']
    # print("Input data : ", input_data)
    output_data = model_api(input_data)
    print(output_data)
    response = jsonify(output_data)
    return response """

@app.route('/apiTPU', methods=['POST'])
def apitpu():
    input_data = request.form['input']
    #print("Input data : ", input_data)
    output_data = model_tpu_api(input_data)
    print(output_data)
    response = jsonify(output_data)
    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0')