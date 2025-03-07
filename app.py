import base64
from io import BytesIO
from random import Random
import time
from flask import Flask, flash, json, redirect, render_template, request, url_for
from flask_socketio import SocketIO, emit
import secrets
from PIL import Image
from src.dlg import LeNet, ImageProcessing
from src.utils import Helper, LocalSession
from views.single_cifar import SingleCifarViewRegister
from views.multiple_cifar import MultipleCifarViewRegister
from torchvision import transforms

app = Flask(__name__)
app.config.update(
    SECRET_KEY=secrets.token_hex(32),
)
socketio = SocketIO(app)


local_session = LocalSession()
image_processing = ImageProcessing()

@app.route("/")
def index():
    return render_template("index.html")

SingleCifarViewRegister().register_routes(app=app, helper=Helper(), socketio=socketio, image_processing=image_processing)
MultipleCifarViewRegister().register_routes(app=app, socketio=socketio, image_processing=image_processing, helper=Helper(), local_session=local_session)
# File upload 
def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/handle_file_upload', methods=['POST'])
def handle_file_upload():
    try:
        if 'image_file' not in request.files:
            flash('No file part')
            return redirect(url_for('index'))
        
        file = request.files['image_file']
        
        if file.filename == '' or not allowed_file(file.filename):
            flash('No selected file or invalid file type')
            return redirect(url_for('index'))
        
        # Read and process the image
        image = Image.open(file.stream)
        helper = Helper()
        
        # Store original image in session
        original_encoded = helper.encode_image(image)
        activation_function = request.form.get('activation_function', 'relu')
        
        local_session.clear()
        local_session.add({
            'original_image': original_encoded,
            'original_size': f"{image.size[0]}x{image.size[1]}"
        })
        
        return render_template('loading_custom.html', 
                             activation_function=activation_function)
    
    except Exception as err:
        print(f"Image upload error: {err}")
        flash('Error processing image')
        return redirect(url_for('index'))

@socketio.on('start_custom_process')
def handle_custom_process(data):
    try:
        helper = Helper()
        original_image = helper.decode_image(data['original_image'])
        image = Image.open(BytesIO(original_image))
        
        if image.size != (32,32):
            image = image.resize((32,32), Image.LANCZOS)
            
        transform = transforms.ToTensor()
        input_tensor = transform(image).unsqueeze(0).to(image_processing.device)
        
        result = image_processing.process_custom_image(input_tensor, data['activation_function'])
        
        result['original_image'] = data['original_image']
        result['activation_function'] = data['activation_function']
        
        local_session.add(result)
        
        emit('complete', {'result': result})
    except Exception as err:
        print(f"Cutstom Image Process Error: {err}")
        emit('error', str(err))


@app.route('/result_custom')
def result_custom():
    try:
        results = local_session.get_results()
        if not results or len(results) < 2:  # Need both original and processed results
            return redirect(url_for('index'))
            
        return render_template('result_custom.html', 
                             original=results[0],
                             result=results[1])
                             
    except Exception as err:
        print(f"Result custom error: {err}")
        return redirect(url_for('index'))
# Encode Decode Image

@app.template_filter('b85decode')
def b85decode_filter(b85_string):
    """Convert base85 to bytes"""
    return base64.b85decode(b85_string)

@app.template_filter('b64encode')
def b64encode_filter(data):
    """Convert bytes to base64"""
    return base64.b64encode(data).decode()

if __name__ == '__main__':
    socketio.run(app, debug=True)
