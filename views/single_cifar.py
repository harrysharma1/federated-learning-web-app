from random import Random
import secrets
from flask import json, redirect, render_template, request, url_for
from flask.views import MethodView
from flask_socketio import emit

class SingleCifarView(MethodView):    
    def post(self):
        cifar_index = int(request.form['cifar_index'])
        activation_function = request.form['activation_function']
        noise_scale = float(request.form['noise_scale_single'])/100
        return render_template("loading_single.html", cifar_index=cifar_index, activation_function=activation_function, noise_scale=noise_scale) 

    def get(self):
        return redirect(url_for('index'))

class SingleRandomCifarView(MethodView):
    def post(self):
        random = Random()
        cifar_index = random.randint(0,49999)
        activation_function = secrets.choice(['relu','sigmoid','tanh'])
        noise_scale = float(random.randint(0.0,1.0))
        return render_template("loading_single.html", 
                               cifar_index=cifar_index, 
                               activation_function=activation_function,
                               noise_scale=noise_scale) 
    
    def get(self):
        return redirect(url_for('index'))   

class ResultView(MethodView):
    def __init__(self, helper, image_processing):
        super().__init__()
        self.image_processing = image_processing
        self.helper = helper
    
    def get(self):
        try:
        # Get and validate data
            data_str = request.args.get('data')
            if not data_str:
                raise ValueError("No data provided")
                
            data = json.loads(data_str)
            cifar_index = data.get('cifar_index')
            if cifar_index is None:
                raise ValueError("No CIFAR index provided")
            
            # Get and encode original image
            original_img = self.image_processing.dst[cifar_index][0]
            data['original_image'] = self.helper.encode_image(original_img)
            
            # Validate reconstructed image exists
            if 'image' not in data:
                raise ValueError("No reconstructed image in data")

            return render_template('result_single.html', result=data)
        except Exception as err:
            print(f"Error in result route: {err}")
            print(f"Request args: {request.args}")
            return redirect(url_for('index'))
        
def register_socket_io_handlers(socketio, image_processing):
    @socketio.on('start_single_process')
    def handle_single_process(data):
        cifar_index = data['cifar_index']
        activation_function = data['activation_function']
        noise_scale = float(data.get('noise_scale', 0))
        
        try:
            result = image_processing.process_single_image(cifar_index, activation_function, noise_scale)
            emit('complete',{'result':result})
        except Exception as err:
            emit('error', str(err))
            
class SingleCifarViewRegister():
    def register_routes(self, app, helper, socketio, image_processing):
        app.add_url_rule('/interactive/handle_data_single', 
                        view_func=SingleCifarView.as_view('handle_data_single'))
        
        app.add_url_rule('/interactive/handle_data_single_random', 
                        view_func=SingleRandomCifarView.as_view('handle_data_single_random'))
        
        app.add_url_rule('/interactive/result', 
                        view_func=ResultView.as_view('result', helper, image_processing))
        
        register_socket_io_handlers(socketio, image_processing)