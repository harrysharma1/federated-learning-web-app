import base64
from io import BytesIO
from random import Random
import time
from flask import Flask, flash, json, redirect, render_template, request, url_for
from flask_socketio import SocketIO, emit
import secrets
from src.dlg import ImageProcessing
from src.utils import Helper, LocalSession
from views.single_cifar import SingleCifarViewRegister
from views.multiple_cifar import MultipleCifarViewRegister
from views.index import IndexViewRegister
from views.misc import MiscViewRegister
from views.custom_image import CustomImageViewRegister

# Create Necessary objects
app = Flask(__name__)
app.config.update(
    SECRET_KEY=secrets.token_hex(32),
)
socketio = SocketIO(app)
local_session = LocalSession()
image_processing = ImageProcessing()

# Register All Views for Web Application
IndexViewRegister().register_routes(app=app)
MiscViewRegister().register_routes(app=app)
SingleCifarViewRegister().register_routes(app=app, helper=Helper(), socketio=socketio, image_processing=image_processing)
MultipleCifarViewRegister().register_routes(app=app, socketio=socketio, image_processing=image_processing, helper=Helper(), local_session=local_session)
CustomImageViewRegister().register_routes(app=app, socketio=socketio, image_processing=image_processing,local_session=local_session, helper=Helper())


# Encode Decode Image
@app.template_filter('b85decode')
def b85decode_filter(b85_string):
    """Provide Base85 decode feature to be accessible to templates.

    Args:
        b85_string (str): Base85 encoded string.

    Returns:
        bytes: The decoded bytes from the Base85 encoded string.
    """
    return base64.b85decode(b85_string)

@app.template_filter('b64encode')
def b64encode_filter(data):
    """Provides Base64 encode feature accessible to templates.

    Args:
        data (str): Normally passed string.

    Returns:
        bytes: The Base64 encoded representation of the input string.
    """
    return base64.b64encode(data).decode()

if __name__ == '__main__':
    socketio.run(app)
