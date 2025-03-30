from flask import render_template
from flask.views import MethodView

class InteractiveView(MethodView):
    """Defining Class-Based Views for Interactive routing.

    Args:
        MethodView: Dispatches request methods to the corresponding instance methods. For example, if you implement a get method, it will be used to handle GET requests.
        This can be useful for defining a REST API. methods is automatically set based on the methods defined on the class.
    """
    def get(self):
        """Return items of the GET request at URL '/interactive'. 

        Returns:
            interactive.html: Directing application to interactive page.
        """
        return render_template("interactive.html")
   
class BackgroundView(MethodView):
    """Defining Class-Based Views for Background routing.

    Args:
        MethodView: Dispatches request methods to the corresponding instance methods. For example, if you implement a get method, it will be used to handle GET requests.
        This can be useful for defining a REST API. methods is automatically set based on the methods defined on the class.
    """
    def get(self):
        """Return items of the GET request at URL '/background'. 

        Returns:
            background.html: Directing application to the background page.
        """
        return render_template("background.html")

class AlgorithmView(MethodView):
    """Defining Class-Based Views for Algorithm routing.

    Args:
        MethodView: Dispatches request methods to the corresponding instance methods. For example, if you implement a get method, it will be used to handle GET requests.
        This can be useful for defining a REST API. methods is automatically set based on the methods defined on the class.
    """
    def get(self):
        """Return items of the GET request at URL '/algorithm'. 

        Returns:
            algorithms.html: Directing application to the algorithms page.
        """
        return render_template("algorithms.html")

class CitationView(MethodView):
    """Defining Class-Based Views for Citation routing.

    Args:
        MethodView: Dispatches request methods to the corresponding instance methods. For example, if you implement a get method, it will be used to handle GET requests.
        This can be useful for defining a REST API. methods is automatically set based on the methods defined on the class.
    """
    def get(self):
        """Return items of the GET request at URL '/citation'. 

        Returns:
            citations.html: _description_
        """
        return render_template("citations.html")
    
class ExperimentResultView(MethodView):
    """Defining Class-Based Views for ExperimentResult routing.

    Args:
        MethodView: Dispatches request methods to the corresponding instance methods. For example, if you implement a get method, it will be used to handle GET requests.
        This can be useful for defining a REST API. methods is automatically set based on the methods defined on the class.
    """
    def get(self):
        """Return items of the GET request at url '/experiment'

        Returns:
            experiment_result.html: _description_
        """
        return render_template("experiment_result.html")
    
class MiscViewRegister():
    """Registering all Class-based views in this file.
    """
    def register_routes(self, app):
        """Registering the HTTP routes in regards to these Class-based views.

        Args:
            app (Flask): Instance of the Flask class, to actually add routes to application.
        """
        app.add_url_rule("/background", view_func=BackgroundView.as_view('background'))
        app.add_url_rule('/interactive', view_func=InteractiveView.as_view('interacttive'))
        app.add_url_rule('/algorithms', view_func=AlgorithmView.as_view('algorithms'))
        app.add_url_rule('/citations', view_func=CitationView.as_view('citations'))
        app.add_url_rule('/experiment', view_func=ExperimentResultView.as_view('experiment'))
        