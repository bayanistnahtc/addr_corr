FROM repo.infranet.co:5000/ml-base:1.1.9

COPY ./requirements.txt /

RUN pip install -r requirements.txt

COPY ./app /app

WORKDIR /app

RUN rm -rf multiproc-tmp && mkdir multiproc-tmp
ENV prometheus_multiproc_dir=multiproc-tmp

ARG CONFIG_DIR_IN_CONTAINER=/app_configs
ENV config_dir=$CONFIG_DIR_IN_CONTAINER

ARG CONFIG_NAME=app_config.yaml
ENV config_name=$CONFIG_NAME

CMD ["uvicorn", "app:app", "--host=0.0.0.0", "--port=8080"]
