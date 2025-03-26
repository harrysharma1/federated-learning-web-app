from flask import redirect, render_template, request, url_for
from flask.views import MethodView

class InteractiveView(MethodView):
    def get(self):
        return render_template("interactive.html")
   
class BackgroundView(MethodView):
    def get(self):
        return render_template("background.html")

class AlgorithmView(MethodView):
    def get(self):
        return render_template("algorithms.html")

class CitationView(MethodView):
    def get(self):
        return render_template("citations.html")

class MiscViewRegister():
    def register_routes(self, app):
        app.add_url_rule("/background", view_func=BackgroundView.as_view('background'))
        app.add_url_rule('/interactive', view_func=InteractiveView.as_view('interacttive'))
        app.add_url_rule('/algorithms', view_func=AlgorithmView.as_view('algorithms'))
        app.add_url_rule('/citations', view_func=CitationView.as_view('citations'))
        