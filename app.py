import base64
from io import BytesIO
from flask import Flask, redirect, render_template, request, url_for
from torchvision import datasets

app = Flask(__name__)
dst = datasets.CIFAR100("~/.torch", download=True)

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/handle_data_single', methods=['POST'])
def handle_data_single():
    cifar_index = request.form['cifar_index']
    activation_function = request.form['activation_function']
    
    print(f"CIFAR Index: {cifar_index}")
    print(f"Activation Function: {activation_function}")
    return redirect(url_for('index'))

@app.route('/handle_data_multiple', methods=['POST'])
def handle_data_multiple():
    start_cifar_index = request.form['start_cifar_index']
    end_cifar_index = request.form['end_cifar_index']
    activation_function = request.form['activation_function']
    
    print(f"Start CIFAR Index: {start_cifar_index}")
    print(f"End CIFAR Index: {end_cifar_index}")
    print(f"Activation Function: {activation_function}")
    return redirect(url_for('index'))
 
if __name__ == '__main__':
	app.run(debug=True)