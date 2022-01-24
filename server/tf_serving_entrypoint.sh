#!/bin/bash

tensorflow_model_server --port=8500 --rest_api_port="${PORT}" --model_name="leaves_classifier" --model_base_path="./leaves_classifier" "$@"


