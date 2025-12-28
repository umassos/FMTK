FROM nvcr.io/nvidia/tritonserver:24.04-py3

# Install system libs if needed
# RUN apt-get update && apt-get install -y ...

# Install python libs
COPY requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt

# If fmtk is a local package, copy and install it
# COPY fmtk /opt/fmtk
# ENV PYTHONPATH="${PYTHONPATH}:/opt/fmtk"