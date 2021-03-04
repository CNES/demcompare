FROM ubuntu:20.04
LABEL maintainer="CNES"

# Avoid apt install interactive questions.
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
  && apt-get install --no-install-recommends -y --quiet \
  git=1:2.25.1-1ubuntu3 \
  make=4.2.1-1.2 \
  python3-pip=20.0.2-5ubuntu1.1 \
  python3-dev=3.8.2-0ubuntu2 \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

# DEMcompare install
WORKDIR /demcompare
COPY . /demcompare
RUN python3 -m pip --no-cache-dir install /demcompare/.

# Auto args completion
RUN register-python-argcomplete demcompare >> ~/.bashrc

# Go in tests directory to be able to launch tests easily
WORKDIR /demcompare/tests/

# launch demcompare
ENTRYPOINT ["demcompare"]
CMD ["-h"]
