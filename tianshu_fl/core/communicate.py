from flask import Flask
from werkzeug.serving import run_simple
from werkzeug.middleware.dispatcher import DispatcherMiddleware


app = Flask(__name__)


@app.route("/")
def test_flask():
    return 'hello world'

@app.route("/jobs", methods=['GET'])
def acquire_job_list():
    return ''






def start_communicate_server(api_version, ip, port):

    app.url_map.strict_slashes = False
    run_simple(hostname=ip, port=port, application=app, threaded=True)

