import os

from titanic.api.auth import verify_token


JAEGER_ENDPOINT = os.getenv("JAEGER_ENDPOINT", "http://jaeger.{{ cookiecutter.developer_redhat_username }}-dev.svc.cluster.local:4318/v1/traces")


def infer() -> list:
    return [0]
