import pickle, os
from flask import Flask, request
from werkzeug.serving import run_simple
from tianshu_fl.utils.utils import return_data_decorator


app = Flask(__name__)

BASE_MODEL_PATH = os.path.abspath(".")+"\\res\\models"


@return_data_decorator
@app.route("/", methods=['GET'])
def test_client():
    return "Hello tianshu_fl client"

@return_data_decorator
@app.route("/aggregatepars", methods=['POST'])
def submit_aggregate_pars():

    recv_aggregate_files = request.files
    job_ids = request.data

    for job_id in job_ids:
        tmp_aggregate_file = recv_aggregate_files['tmp_aggregate_{}'.format(job_id)]
        job_base_model_dir = BASE_MODEL_PATH + "\\models_{}\\tmp_aggregate_pars".format(job_id)
        latest_num = len(os.listdir(job_base_model_dir)) - 1
        latest_tmp_aggretate_file_path = job_base_model_dir + "\\avg_pars_{}".format(latest_num)
        with open(latest_tmp_aggretate_file_path, "wb") as f:
            with open(tmp_aggregate_file, "rb") as tmp_f:
                for line in tmp_f.readlines():
                    f.write(line)

    return "ok", 200



def start_communicate_client(client_ip, client_port):
    app.url_map.strict_slashes = False
    run_simple(hostname=client_ip, port=client_port, application=app, threaded=True)