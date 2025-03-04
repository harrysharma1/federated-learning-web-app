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
from views.single_cifar import register_routes
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

register_routes(app=app, helper=Helper(), socketio=socketio, image_processing=image_processing)



# Multiple Choices

@app.route('/handle_data_multiple', methods=['GET','POST'])
def handle_data_multiple():
    if request.method == 'POST':
        start_cifar_index = int(request.form['start_cifar_index'])
        end_cifar_index = int(request.form['end_cifar_index'])
        activation_function = request.form['activation_function']
        
        return render_template("loading_multiple.html", start_cifar_index=start_cifar_index, end_cifar_index=end_cifar_index, activation_function=activation_function)
    else:
        return redirect(url_for('index'))
    
@socketio.on('start_processing')
def handle_process(data):
    try:
        start_cifar_index = data['start_index']
        end_cifar_index = data['end_index']
        activation_function = data['activation_function']
        total = end_cifar_index - start_cifar_index + 1
        local_session.clear()
        results = []
        for i, cifar_id in enumerate(range(start_cifar_index, end_cifar_index + 1)):
            result = image_processing.process_single_image(cifar_id, activation_function)
            results.append({'cifar_id': cifar_id, **result})
            result_with_id = {'cifar_id': cifar_id, **result}
            local_session.add(result_with_id)
            progress = ((i + 1)/total) * 100
            emit('progress', {
                'progress': progress,
                'current_result': result,
                'curr_id': cifar_id
            })
            time.sleep(0.1)
        
        # Send results directly to client
        emit('complete', {'results': results})
        
    except Exception as err:
        print(f"Processing error: {err}")
        emit('error', str(err))

@socketio.on('convert_image')
def convert_image(data):
    helper = Helper()
    try:
        base85_str = data['image']
        bytes_data = helper.decode_image(base85_str)
        base64_str = base64.b64encode(bytes_data).decode()
        emit('image_converted', {
            'image' : base64_str
        }) 
    except Exception as err:
        emit ('error', str(err))

@app.route('/chart')
def chart():
    helper = Helper()
    try:
        results = local_session.get_results()
        if not results:
            return redirect(url_for('index'))
        
        # Add original images
        for result in results:
            cifar_id = result['cifar_id']
            original_img = image_processing.dst[cifar_id][0]
            result['original_image'] = helper.encode_image(original_img)
        
        return render_template('chart_multiple.html', results=results)
    except Exception as e:
        print(f"Chart error: {e}")
        return redirect(url_for('index'))    

# Multiple Random

@app.route('/handle_data_random_range', methods=['GET', 'POST'])
def handle_data_random_range():  # Changed function name to match URL
    if request.method == 'POST':
        # Generate random range between 1-15 images
        random = Random()
        range_size = random.randint(1, 15)
        start_index = random.randint(0, 49999 - range_size)
        end_index = start_index + range_size - 1
        activation_function = random.choice(['relu', 'sigmoid', 'tanh'])
        
        return render_template("loading_multiple.html", 
                             start_cifar_index=start_index,
                             end_cifar_index=end_index,
                             activation_function=activation_function)
    else:
        return redirect(url_for('index'))

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
        
        # Resize to 32x32 if needed
        if image.size != (32, 32):
            image = image.resize((32, 32), Image.LANCZOS)
        
        # Convert to tensor
        transform = transforms.ToTensor()
        input_tensor = transform(image).unsqueeze(0).to(image_processing.device)
        
        # Process image using existing pipeline
        result = image_processing.process_single_image(0, data['activation_function'])
        
        # Add original image to result
        result['original_image'] = data['original_image']
        
        # Store in session
        local_session.add(result)
        
        emit('complete', {'result': result})
        
    except Exception as err:
        print(f"Custom process error: {err}")
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
