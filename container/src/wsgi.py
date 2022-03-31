# https://github.com/aws/amazon-sagemaker-examples/blob/main/advanced_functionality/scikit_bring_your_own/container/decision_trees/wsgi.py

import inference

# This is just a simple wrapper for gunicorn to find your app.
# If you want to change the algorithm file, simply change "predictor" above to the
# new file.

app = inference.app
