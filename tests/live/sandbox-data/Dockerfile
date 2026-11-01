# A data-analysis sandbox image for the live tests (tests/live/test_live_sandbox_data.py):
# Python 3.12 with pandas, openpyxl, matplotlib and statsmodels, the versions tested.
#
#   docker build -t trellis-sandbox-data tests/live/sandbox-data
#
# Where the build cannot reach PyPI, download the wheels first and build without an index:
#
#   pip download -d tests/live/sandbox-data/wheels --only-binary=:all: --python-version 3.12 \
#     --platform manylinux_2_28_x86_64 --platform manylinux_2_17_x86_64 \
#     --platform manylinux2014_x86_64 --platform manylinux_2_5_x86_64 --platform manylinux1_x86_64 \
#     pandas==3.1.0 openpyxl==3.1.5 matplotlib==3.11.2 statsmodels==0.15.0 "wrapt<2.6.0.dev0"
#   docker build --network none --build-arg PIP_NO_INDEX=1 -t trellis-sandbox-data tests/live/sandbox-data
FROM python:3.12-slim
ARG PIP_NO_INDEX=
RUN --mount=type=bind,target=/context \
    pip install --no-cache-dir --find-links /context/wheels \
      pandas==3.1.0 openpyxl==3.1.5 matplotlib==3.11.2 statsmodels==0.15.0 \
 && mkdir -p /workspace
ENV MPLBACKEND=Agg
WORKDIR /workspace
