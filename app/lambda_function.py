"""
AWS Lambda handler that exposes the Canvas endpoint to other applications,
typically behind an API Gateway HTTP API (POST /predict).

Request body (raw application, messy values are fine):
    {"age": "34", "employment_status": "Salaried", "annual_income": "$85,000",
     "credit_score": "720", "existing_debt": "12.5k", "loan_amount": "20K",
     "employment_length": "5 years", "loan_purpose": "Home Improvement"}

Environment variables:
    ENDPOINT_NAME   name of the SageMaker endpoint
    TRANSFORM_DIR   folder that holds the canvas_transforms/*.py files
                    (default: ./canvas_transforms inside the deployment package)

Packaging: include preprocess.py, this file and the canvas_transforms folder in
the zip, and add the AWS-managed "AWS SDK for pandas" Lambda layer for pandas/numpy.
"""

import json
import os

os.environ.setdefault(
    "TRANSFORM_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "canvas_transforms"),
)

import boto3  # noqa: E402

from preprocess import applications_to_csv  # noqa: E402

ENDPOINT_NAME = os.environ["ENDPOINT_NAME"]
runtime = boto3.client("sagemaker-runtime")


def _response(status, payload):
    return {
        "statusCode": status,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(payload),
    }


def lambda_handler(event, context):
    try:
        body = event.get("body") or "{}"
        application = json.loads(body) if isinstance(body, str) else body
    except json.JSONDecodeError:
        return _response(400, {"error": "Request body must be JSON"})

    csv_row = applications_to_csv([application])
    result = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="text/csv",
        Accept="application/json",
        Body=csv_row.encode("utf-8"),
    )
    prediction = json.loads(result["Body"].read().decode("utf-8"))
    if isinstance(prediction, dict) and "predictions" in prediction:
        prediction = prediction["predictions"][0]
    elif isinstance(prediction, list):
        prediction = prediction[0]

    return _response(200, {
        "decision": prediction.get("predicted_label"),
        "probability": prediction.get("probability"),
        "details": prediction,
    })
