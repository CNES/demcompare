FROM ubuntu:22.04
LABEL maintainer="CNES"

## cars-mesh installation Dockerfile example

# Avoid apt install interactive questions.
ARG DEBIAN_FRONTEND=noninteractive

# Install Ubuntu python dependencies (ignore pinning versions)
# hadolint ignore=DL3008
RUN apt-get update \
  && apt-get install --no-install-recommends -y --quiet \
  git \
  make \
  python3-pip \
  python3-dev \
  python3-venv \
  libgomp1 \
  libx11-6 \
  libgl1 \
  libegl1 \
  libusb-1.0-0 \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

#  Install cars-mesh
WORKDIR /cars-mesh
COPY . /cars-mesh

RUN make clean && make install

# source venv/bin/activate in docker mode
ENV VIRTUAL_ENV='/cars-mesh/venv'
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Clean pip cache
RUN python -m pip cache purge

## run cars-mesh command
ENTRYPOINT ["cars-mesh"]
CMD ["-h"]
