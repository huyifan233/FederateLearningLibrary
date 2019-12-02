
import os, pickle, json
from flask import Flask, send_from_directory, request
from werkzeug.serving import run_simple
from werkzeug.middleware.dispatcher import DispatcherMiddleware
from tianshu_fl.utils.utils import JobEncoder, return_data_decorator
from tianshu_fl.entity.runtime_config import CONNECTED_TRAINER_LIST
from tianshu_fl.core.job_manager import JobManager



API_VERSION = "/api/v1"
JOB_PATH = os.path.abspath(".")+"\\res\\jobs"
BASE_MODEL_PATH = os.path.abspath(".")+"\\res\\models"
INIT_MODEL_PARS = "avg_pars_0"

app = Flask(__name__)



@app.route("/test/<name>")
@return_data_decorator
def test_flask_server(name):

    return name, 200



@app.route("/register/<ip>/<port>/<client_id>", methods=['POST'], endpoint='register_trainer')
@return_data_decorator
def register_trainer(ip, port, client_id):
    trainer_host = ip+":"+port
    if trainer_host not in CONNECTED_TRAINER_LIST:
        job_list = JobManager.get_job_list(JOB_PATH)
        for job in job_list:
            job_model_client_dir = BASE_MODEL_PATH + "\\models_{}\\models_{}".format(job.get_job_id(), client_id)
            if not os.path.exists(job_model_client_dir):
                os.makedirs(job_model_client_dir)
        CONNECTED_TRAINER_LIST.append(trainer_host)
        return 'register_success', 200
    else:
        return 'already connected', 201


@app.route("/offline/<ip>/<port>", methods=['PUT'], endpoint='offline')
@return_data_decorator
def offline(ip, port):
    trainer_host = ip+":"+port
    if trainer_host in CONNECTED_TRAINER_LIST:
        CONNECTED_TRAINER_LIST.remove(trainer_host)
        return 'offline success', 200
    return 'already offline', 201


@app.route("/jobs", methods=['GET'], endpoint='acquire_job_list')
@return_data_decorator
def acquire_job_list():
    job_str_list = []
    job_list = JobManager.get_job_list(JOB_PATH)
    for job in job_list:
        job_str = json.dumps(job, cls=JobEncoder)
        job_str_list.append(job_str)
    return job_str_list, 200


@app.route("/modelpars/<job_id>", methods=['GET'], endpoint='acquire_init_model_pars')
@return_data_decorator
def acquire_init_model_pars(job_id):
    print(job_id)
    init_model_pars_dir = BASE_MODEL_PATH+"\\models_{}\\tmp_aggregate_pars".format(job_id)
    return send_from_directory(init_model_pars_dir, INIT_MODEL_PARS, as_attachment=True)



@app.route("/modelpars/<client_id>/<job_id>", methods=['POST'], endpoint='submit_model_parameter')
@return_data_decorator
def submit_model_parameter(client_id, job_id, fed_avg):
    tmp_parameter_file = request.files['tmp_parameter_file']
    model_pars_dir = BASE_MODEL_PATH+"\\models_{}\\models_{}".format(job_id, client_id)
    if not os.path.exists(model_pars_dir):
        os.makedirs(model_pars_dir)
    model_pars_path = BASE_MODEL_PATH+"\\models_{}\\models_{}\\tmp_parameter_{}".format(job_id, client_id, fed_avg)
    with open(model_pars_path, "wb") as f:
        with open(tmp_parameter_file, "rb") as tmp_f:
            for line in tmp_f.readlines():
                f.write(line)

    return 'submit_success', 200



@app.route("/aggregatepars", methods=['GET'], endpoint='get_aggregate_parameter')
@return_data_decorator
def get_aggregate_parameter():
    return ''




def start_communicate_server(api_version, ip, port):
    app.url_map.strict_slashes = False
    run_simple(hostname=ip, port=port, application=app, threaded=True)

