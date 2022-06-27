FROM ubuntu:20.04
LABEL maintainer="Chloe Thenoz (Magellium), Lisa Vo Thanh (Magellium)"

## mesh_3d installation Dockerfile example
## Hack it !

# Avoid apt install interactive questions.
ARG DEBIAN_FRONTEND=noninteractive

# Install Ubuntu python dependencies
RUN apt-get update \
  && apt-get install --no-install-recommends -y --quiet \
  git=1:2.25.1-1ubuntu3 \
  make=4.2.1-1.2 \
  python3-pip=20.0.2-5ubuntu1.6 \
  python3-dev=3.8.2-0ubuntu2 \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

#  Install mesh_3d
WORKDIR /mesh_3d
COPY . /mesh_3d

# Version will be automatic with git versioning and tags
RUN python3 -m pip --no-cache-dir install /mesh_3d/. \
  # # Auto args completion
  && register-python-argcomplete mesh_3d >> ~/.bashrc

# launch demcompare
ENTRYPOINT ["mesh_3d"]
CMD ["-h"]
