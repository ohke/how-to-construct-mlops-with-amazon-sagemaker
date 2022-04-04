# c.f. https://github.com/aws/amazon-sagemaker-examples/blob/main/advanced_functionality/scikit_bring_your_own/container/decision_trees/predictor.py

# This is the file that implements a flask server to do inferences. It's the file that you will modify to
# implement the scoring for your own algorithm.

from __future__ import print_function

import os
import flask
import torch
from io import BytesIO
from PIL import Image
from torchvision.transforms.functional import to_tensor, normalize

from model import Net

prefix = "/opt/ml/"
model_path = os.path.join(prefix, "model")

# A singleton for holding the model. This simply loads the model and holds it.
# It has a predict function that does a prediction based on the model and the input data.


class ModelService(object):
    model = None  # Where we keep the model when it's loaded

    @classmethod
    def get_model(cls):
        if cls.model == None:
            net = Net()
            net.load_state_dict(torch.load(os.path.join(model_path, "mnist_cnn.pt")))
            net.eval()
            cls.model = net
        return cls.model

    @classmethod
    def predict(cls, input: torch.FloatTensor):
        model = cls.get_model()
        return model(input)


# The flask app for serving predictions
app = flask.Flask(__name__)


@app.route("/ping", methods=["GET"])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    health = ModelService.get_model() is not None  # You can insert a health check here

    status = 200 if health else 404
    return flask.Response(response="\n", status=status, mimetype="application/json")


@app.route("/invocations", methods=["POST"])
def transformation():
    data = None

    # Convert from JPEG to tensor
    if flask.request.content_type == "image/jpeg":
        image = Image.open(BytesIO(flask.request.data)).convert("L").resize((28, 28))
        # NOTE: (1, 1, 28, 28), torch.float32
        data = normalize(
            to_tensor(image) - 1.0 * (-1.0), (0.1307,), (0.3081,)
        ).unsqueeze(0)
    else:
        return flask.Response(
            response="This predictor only supports JPEG data",
            status=415,
            mimetype="text/plain",
        )

    print("Invoked with {} records".format(data.shape[0]))

    # Do the prediction
    predictions = ModelService.predict(data)

    # Convert from tensor to JSON
    return flask.jsonify({k: v for k, v in enumerate(predictions[0].tolist())})


if __name__ == "__main__":
    app.run("0.0.0.0", 8080, debug=True)
