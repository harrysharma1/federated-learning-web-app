from flask import flash, redirect, render_template, request, url_for
from flask.views import MethodView
from flask_socketio import emit
from io import BytesIO
from PIL import Image
from torchvision import transforms


class CustomImageView(MethodView):
    def __init__(self, helper, local_session):
        self.helper = helper
        self.local_session = local_session
        
    def get(self):
        return redirect(url_for('index'))
    
    def post(self):
        try:                     
            if 'image_file' not in request.files:
                return redirect(url_for('index'))
        
            file = request.files['image_file']
            if file.filename == '' or not self.allowed_files(file.filename):
                flash('No selected file or invalid file type')
                return redirect(url_for('index'))
            image = Image.open(file.stream)
            
            original_encoded = self.helper.encode_image(image)
            activation_function = request.form.get('activation_function', 'sigmoid')
            noise_scale = float(request.form['noise_scale_custom'])/100
            self.local_session.clear()
            self.local_session.add(
                {
                    'original_image' : original_encoded,
                    'original_size' : f"{image.size[0]}x{image.size[1]}"
                }
            )
            return render_template('loading_custom.html', activation_function=activation_function, local_session=self.local_session, noise_scale=noise_scale)
        except Exception as err:
            print(f"Error occurred uploading image: {err}")
            import traceback
            traceback.print_exc()
            return redirect(url_for('index'))
    
    def allowed_files(self, filename):
        ALLOWED_FILES = {'png', 'jpg', 'jpeg'}
        return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_FILES
    
class ResultCustomView(MethodView):
    def __init__(self, local_session):
        self.local_session = local_session
        
    def get(self):
        try:
            results = self.local_session.get_results()
            if not results or len(results) < 2: 
                return redirect(url_for('index'))
            
            return render_template('result_custom.html', original=results[0], result=results[1])
        except Exception as err:
            print(f"Error processing image result: {err}")
            return redirect(url_for('index'))
        
def register_socket_io_handlers(socketio, image_processing, local_session, helper):
    @socketio.on('start_custom_process')
    def handle_custom_process(data):
        try:
            original_image = helper.decode_image(data['original_image'])
            noise_scale = float(data.get('noise_scale', 0))
            image = Image.open(BytesIO(original_image))
            
            if image.size != (32,32):
                image = image.resize((32,32), Image.LANCZOS)
                
            transform = transforms.ToTensor()
            
            result = image_processing.process_custom_image(image, data['activation_function'], noise_scale)
            
            result['original_image'] = data['original_image']
            result['activation_function'] = data['activation_function']
            result['noise'] = data['noise_scale']
            
            local_session.add(result)
            
            emit('complete', {'result': result})
        except Exception as err:
            print(f"Custom Image Process Error: {err}")
            emit('error', str(err))

class CustomImageViewRegister:
    def register_routes(self, app, helper, socketio, image_processing, local_session):
        app.add_url_rule('/interactive/handle_file_upload', 
                        view_func=CustomImageView.as_view('handle_file_upload', helper, local_session))
        
        app.add_url_rule('/interactive/result_custom', 
                        view_func=ResultCustomView.as_view('result_custom', local_session))
        
        register_socket_io_handlers(socketio, image_processing, local_session, helper)