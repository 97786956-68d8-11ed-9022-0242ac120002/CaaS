import json
import os

from flask import Flask, request, make_response, jsonify
from flask_httpauth import HTTPBasicAuth
from werkzeug.security import generate_password_hash, check_password_hash

from _version import VERSION
from _core import betweenness, CPU_PROG, GPU_PROG

app = Flask("CaaS")
auth = HTTPBasicAuth()
users = {
    "admin": generate_password_hash(os.getenv("CAAS_PWD"))
}


@auth.verify_password
def verify_password(username, password):
    if username in users and check_password_hash(users.get(username), password):
        return username
    return None


@auth.error_handler
def unauthorized():
    return make_response(jsonify({
        'error': 'auth failed'
    }), 403)


@app.route('/')
def index():
    return make_response(jsonify({
        'service': 'CaaS',
        'version': VERSION
    }), 200)


@app.route('/betweenness', methods=['POST', 'GET'])
@auth.login_required
def enrich():
    if request.method == 'GET':
        return 'cannot enrich database through HTTP_GET'
    elif request.method == 'POST':
        raw_data = request.get_data()
        data = json.loads(raw_data)
        if isinstance(data, dict) and "cfg" in data.keys():
            return make_response(jsonify(betweenness(data["cfg"])), 200)


if __name__ == '__main__':
    if os.path.exists(CPU_PROG) and os.path.exists(GPU_PROG):
        app.run()
    else:
        print("Compile with "
              "\n\t`nvcc BC.parallel.gpu.cu -o BC.parallel.gpu.elf -std=c++11`"
              "\n\t`g++ BC.parallel.cpu.cpp -o BC.parallel.cpu.elf -std=c++11`")
