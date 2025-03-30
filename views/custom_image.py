from flask import flash, redirect, render_template, request, url_for
from flask.views import MethodView
from flask_socketio import emit
from io import BytesIO
from PIL import Image


class CustomImageView(MethodView):
    """Defining Class-Based Views for CustomImage routing.

    Args:
        MethodView : Dispatches request methods to the corresponding instance methods. For example, if you implement a get method, it will be used to handle GET requests.
        This can be useful for defining a REST API. methods is automatically set based on the methods defined on the class.
    """
    def __init__(self, helper, local_session):
        """Init function for CustomImageView class

        Args:
            helper (Helper): Instance of Helper Class.
            local_session (LocalSession): Instance of LocalSession Class.
        """
        self.helper = helper
        self.local_session = local_session
        
    def get(self):
        """Return items of the GET request at URL '/interactive/handle_file_upload'. 

        Returns:
            index.html: In this case, it redirects to the home page of the web application.
        """
        return redirect(url_for('index'))
    
    def post(self):
        """Return items of the POST request at URL '/interactive/handle_file_upload'.

        Returns:
            loading_custom.html : Sends data as local_session to send through a WebSocket connection.
        """
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
        """Specifications for allowed file types in the back-end.

        Args:
            filename (str): String representation of file.

        Returns:
            bool: Returns True if file includes '.' and an allowed filetype.
        """
        ALLOWED_FILES = {'png', 'jpg', 'jpeg'}
        return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_FILES
    
class ResultCustomView(MethodView):
    """Defining Class-Based Views for ResultCustom routing.

    Args:
        MethodView: Dispatches request methods to the corresponding instance methods. For example, if you implement a get method, it will be used to handle GET requests.
        This can be useful for defining a REST API. methods is automatically set based on the methods defined on the class.
    """
    def __init__(self, local_session):
        """Init function for ResultCustomView class.
        Args:
            local_session (LocalSession): Instance of LocalSession Class.
        """
        self.local_session = local_session
        
    def get(self):
        """Return items of the GET request at URL '/interactive/result_custom'.

        Returns:
            result_custom.html: Retrieves data from WebSocket to display information of the simulated attack result.
        """
        try:
            results = self.local_session.get_results()
            if not results or len(results) < 2: 
                return redirect(url_for('index'))
            
            return render_template('result_custom.html', original=results[0], result=results[1])
        except Exception as err:
            print(f"Error processing image result: {err}")
            return redirect(url_for('index'))
        
def register_socket_io_handlers(socketio, image_processing, local_session, helper):
    """Registering WebSocket handler for class based method.

    Args:
        socketio (SocketIO): Instance of SocketIO class, to register a WebSocket connection.
        image_processing (ImageProcessing): Instance of ImageProcessing class, to reduce to 32x32 and train on custom image gradient.
        local_session (LocalSession): Instance of LocalSession class.
        helper (Helper): Instance of Helper class.
    """
    @socketio.on('start_custom_process')
    def handle_custom_process(data):
        """Handling the WebSocket

        Args:
            data (Dict): Dictionary for passing data between WebSocket connections.
        """
        try:
            original_image = helper.decode_image(data['original_image'])
            noise_scale = float(data.get('noise_scale', 0))
            image = Image.open(BytesIO(original_image))
            
            if image.size != (32,32):
                image = image.resize((32,32), Image.LANCZOS)
                
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
    """Registering all Class-based views in this file.
    """
    def register_routes(self, app, helper, socketio, image_processing, local_session):
        """Registering the HTTP routes in regards to these Class-based views.

        Args:
            app (Flask): Instance of the Flask class, to actually add routes to application.
            helper (Helper): Instance of the Helper class, to provide utility functions.
            socketio (SocketIO): Instance of the SocketIO class, to provide WebSocket handling.
            image_processing (ImageProcessing): Instance of the ImageProcessing class, to process and train on custom images.
            local_session (LocalSession): Instance of the LocalSession class, to provide local store for passing data along in WebSocket.
        """
        app.add_url_rule('/interactive/handle_file_upload', 
                        view_func=CustomImageView.as_view('handle_file_upload', helper, local_session))
        
        app.add_url_rule('/interactive/result_custom', 
                        view_func=ResultCustomView.as_view('result_custom', local_session))
        
        register_socket_io_handlers(socketio, image_processing, local_session, helper)