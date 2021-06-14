from flask import Flask, request, jsonify, render_template
import util
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('app.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route("/get_location_names")
def get_location_names():
    response = jsonify({
            "locations": util.get_location_names()
    })
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response
    
@app.route("/predict_home_price", methods = ["POST"])
def predict_home_price():
    total_sqft = float(request.form['total_sqft'])
    location = request.form['location']
    bhk = int(request.form['bhk'])
    bath = int(request.form['bath'])
    
    response = jsonify({
            'estimated_price': util.get_estimated_price(location, total_sqft, bhk, bath)
    })
    
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response
    
@app.errorhandler(404)
def error404(request):
    return render_template('404.html')

if __name__ == "__main__":
    print("Starting the Flask Server...")
    util.load_saved_artifacts()
    app.run(debug=True)