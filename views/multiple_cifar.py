import base64
from random import Random
import secrets
import time
from flask import Blueprint, redirect, render_template, request, url_for
from flask.views import MethodView
from flask_socketio import emit

class MultipleCifarView(MethodView):
    def post(self):
        start_cifar_index = int(request.form['start_cifar_index'])
        end_cifar_index = int(request.form['end_cifar_index'])
        activation_function = request.form['activation_function']
        noise_scale = float(request.form['noise_scale_multiple'])/100 
        return render_template("loading_multiple.html", start_cifar_index=start_cifar_index, end_cifar_index=end_cifar_index, activation_function=activation_function, noise_scale=noise_scale) 

    def get(self):
        return redirect(url_for('index'))

class RandomRangeCifarView(MethodView):
    def post(self):
        random = Random()
        range_size = random.randint(1, 15)
        start_index = random.randint(0, 49999 - range_size)
        end_index = start_index + range_size - 1
        noise_scale = float(random.randint(0.0,1.0))
        activation_function = secrets.choice(['relu', 'sigmoid', 'tanh'])
        
        return render_template("loading_multiple.html", 
                             start_cifar_index=start_index,
                             end_cifar_index=end_index,
                             activation_function=activation_function,
                             noise_scale = noise_scale)

class ChartView(MethodView):
    def __init__(self, helper, local_session, image_processing):
        super().__init__()
        self.helper = helper 
        self.local_session = local_session
        self.image_processing = image_processing
        
    def get(self):
        try:
            results = self.local_session.get_results()
            if not results:
                return redirect(url_for('index'))
            
            # Add original images
            for result in results:
                cifar_id = result['cifar_id']
                original_img = self.image_processing.dst[cifar_id][0]
                result['original_image'] = self.helper.encode_image(original_img)
            
            return render_template('chart_multiple.html', results=results)
        except Exception as e:
            print(f"Chart error: {e}")
            return redirect(url_for('index'))   

def register_socket_io_handlers(socketio, image_processing, local_session, helper):    
    @socketio.on('start_processing')
    def handle_process(data):
        try:
            start_cifar_index = data['start_index']
            end_cifar_index = data['end_index']
            activation_function = data['activation_function']
            total = end_cifar_index - start_cifar_index + 1
            noise_scale = data['noise_scale']
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
        try:
            base85_str = data['image']
            bytes_data = helper.decode_image(base85_str)
            base64_str = base64.b64encode(bytes_data).decode()
            emit('image_converted', {
                'image' : base64_str
            }) 
        except Exception as err:
            emit ('error', str(err))

class MultipleCifarViewRegister():
    def register_routes(self, app, helper, socketio, local_session, image_processing):
        app.add_url_rule('/interactive/handle_data_multiple', 
                        view_func=MultipleCifarView.as_view('handle_data_multiple'))
        
        app.add_url_rule('/interactive/handle_data_random_range', 
                        view_func=RandomRangeCifarView.as_view('handle_data_random_range'))
        
        app.add_url_rule('/interactive/chart', 
                        view_func=ChartView.as_view('chart', helper, local_session, image_processing))
        
        register_socket_io_handlers(socketio=socketio, image_processing=image_processing, local_session=local_session, helper=helper)