from __future__ import print_function
import flask


app = flask.Flask(__name__)


@app.route("/ping", methods=["GET"])
def ping():
    return flask.Response(response="\n", status=200, mimetype="application/json")


@app.route("/invocations", methods=["POST"])
def transformation():
    return flask.Response(
        response={"result": "ok"}, status=200, mimetype="application/json"
    )
