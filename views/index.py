from flask import render_template
from flask.views import MethodView


class IndexView(MethodView):
    """Defining Class-Based Views for Index routing.

    Args:
        MethodView: Dispatches request methods to the corresponding instance methods. For example, if you implement a get method, it will be used to handle GET requests.
        This can be useful for defining a REST API. methods is automatically set based on the methods defined on the class.
    """
    def get(self):
        """Return items of the GET request at URL '/'. 

        Returns:
            index.html: Directing application to homepage.
        """
        return render_template("index.html")
    
class IndexViewRegister():
    """Registering all Class-based views in this file.
    """
    def register_routes(self, app):
        """Registering the HTTP routes in regards to these Class-based views.

        Args:
            app (Flask): Instance of the Flask class, to actually add routes to application.
        """
        app.add_url_rule('/', view_func=IndexView.as_view('index'))
        
