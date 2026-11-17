"""
Send one or more raw loan applications to the deployed Canvas endpoint.

Usage:
    export AWS_REGION=us-east-1
    export ENDPOINT_NAME=loan-eligibility-endpoint
    python app/invoke_endpoint.py app/sample_application.json
"""

import json
import os
import sys

import boto3

from preprocess import applications_to_csv


def predict(applications, endpoint_name, region=None):
    client = boto3.client("sagemaker-runtime", region_name=region)
    body = applications_to_csv(applications)
    response = client.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="text/csv",
        Accept="application/json",
        Body=body.encode("utf-8"),
    )
    result = json.loads(response["Body"].read().decode("utf-8"))
    # Canvas classification models return predicted_label, probability,
    # labels and probabilities. Handle both a bare list and {"predictions": [...]}.
    predictions = result.get("predictions", result) if isinstance(result, dict) else result
    return body, predictions


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "sample_application.json")
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    applications = data if isinstance(data, list) else [data]

    endpoint = os.environ.get("ENDPOINT_NAME", "canvas-new-deployment-11-16-2026-8-24-PM")
    body, predictions = predict(applications, endpoint, os.environ.get("AWS_REGION", "us-east-1"))

    print("Payload sent to endpoint (cleaned CSV):")
    print(body)
    print("Prediction(s):")
    print(json.dumps(predictions, indent=2))


if __name__ == "__main__":
    main()
