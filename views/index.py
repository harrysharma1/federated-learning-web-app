from flask import Blueprint, redirect, render_template, request, url_for
from flask.views import MethodView

index_blueprint = Blueprint('index', __name__)

class IndexView(MethodView):
    def get(self):
        return render_template("index.html")
    
class InteractiveView(MethodView):
    def get(self):
        return render_template("interactive.html")

class IndexViewRegister():
    def register_routes(self, app):
        app.add_url_rule('/', view_func=IndexView.as_view('index'))
class InteractiveViewRegister():
    def register_routes(self, app):
        app.add_url_rule('/interactive', view_func=InteractiveView.as_view('interacttive'))