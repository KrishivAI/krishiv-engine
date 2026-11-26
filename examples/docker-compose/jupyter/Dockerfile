# SPDX-License-Identifier: Apache-2.0
# JupyterLab with the PySpark Connect client (`pyspark-client`: no JVM, no Spark jars).
FROM python:3.12-slim
COPY requirements.txt /tmp/requirements.txt
# The home folder is writable by any uid: compose.yaml runs the container as the host user
# (JUPYTER_UID), who has no account in the image.
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    && useradd --uid 1000 --create-home --shell /bin/bash jovyan \
    && chmod 1777 /home/jovyan
COPY --chmod=755 start.sh /usr/local/bin/start.sh
ENV HOME=/home/jovyan
USER 1000:1000
WORKDIR /home/jovyan/work
EXPOSE 8888
CMD ["start.sh"]
