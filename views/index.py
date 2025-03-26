from flask import render_template
from flask.views import MethodView


class IndexView(MethodView):
    def get(self):
        return render_template("index.html")
    
class IndexViewRegister():
    def register_routes(self, app):
        app.add_url_rule('/', view_func=IndexView.as_view('index'))
        
