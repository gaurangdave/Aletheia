# See here for image contents: https://github.com/microsoft/vscode-dev-containers/tree/v0.245.2/containers/python-3-anaconda/.devcontainer/base.Dockerfile

FROM mcr.microsoft.com/vscode/devcontainers/anaconda:0-3

# [Choice] Node.js version: none, lts/*, 16, 14, 12, 10
ARG NODE_VERSION="none"

ARG LOCAL=/usr/local
ARG HOME=/home/vscode
ARG PLATFORM=/workspace/platform
ENV SHELL /bin/bash

RUN if [ "${NODE_VERSION}" != "none" ]; then su vscode -c "umask 0002 && . /usr/local/share/nvm/nvm.sh && nvm install ${NODE_VERSION} 2>&1"; fi

# Copy environment.yml (if found) to a temp location so we update the environment. Also
# copy "noop.txt" so the COPY instruction does not fail if no environment.yml exists.
COPY environment.yml* .devcontainer/noop.txt /tmp/conda-tmp/
RUN if [ -f "/tmp/conda-tmp/environment.yml" ]; then umask 0002 && /opt/conda/bin/conda env update -n base -f /tmp/conda-tmp/environment.yml; fi \
    && rm -rf /tmp/conda-tmp


# Installing Go Lang
ARG GO_VERSION=1.16.3
RUN wget https://golang.org/dl/go${GO_VERSION}.linux-amd64.tar.gz \
    && tar -C ${LOCAL} -xzf go${GO_VERSION}.linux-amd64.tar.gz \
    && rm go${GO_VERSION}.linux-amd64.tar.gz
ENV GOPATH ${HOME}/go
ENV PATH $PATH:${LOCAL}/go/bin:$GOPATH/bin

# Installing Protocol Buffer Compiler
ARG PROTOC_VERSION=3.15.8
RUN wget https://github.com/protocolbuffers/protobuf/releases/download/v${PROTOC_VERSION}/protoc-${PROTOC_VERSION}-linux-x86_64.zip \
    && sudo unzip protoc-${PROTOC_VERSION}-linux-x86_64.zip -d ${LOCAL} bin/protoc \
    && sudo unzip -o protoc-${PROTOC_VERSION}-linux-x86_64.zip -d ${LOCAL}/.local/ include/* \
    && rm protoc-${PROTOC_VERSION}-linux-x86_64.zip
# ENV PROTOC_PATH ${HOME}/go    
# ENV PATH $PATH:${HOME}/.local/bin/protoc

# Installing EVANs CLI
ARG EVANS_CLI_VERSION=0.9.3
RUN wget https://github.com/ktr0731/evans/releases/download/${EVANS_CLI_VERSION}/evans_linux_amd64.tar.gz \
    && tar -C /usr/local/bin/ -xzf evans_linux_amd64.tar.gz \
    && rm evans_linux_amd64.tar.gz

# Installing Protocol Buffer Go Plugin
ARG PROTOC_GEN_GO_VERSION=1.26.0
RUN go install google.golang.org/protobuf/cmd/protoc-gen-go@v${PROTOC_GEN_GO_VERSION}
ENV PATH $PATH:${HOME}/go/bin/protoc-gen-go

ARG PROTOC_GEN_GO_GRPC_VERSION=1.1.0
RUN go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@v${PROTOC_GEN_GO_GRPC_VERSION}
ENV PATH $PATH:${HOME}/go/bin/protoc-gen-go-grpc

RUN go get google.golang.org/grpc
RUN go get go.mongodb.org/mongo-driver/mongo

ENV PATH $PATH:$(go env GOPATH)/bin

# Exporting path for all go libraries.
ENV PATH $PATH:${HOME}/go/bin

# 
# RUN sudo chown -R node:node /home/node/

# [Optional] Uncomment this section to install additional OS packages.
# RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
#     && apt-get -y install --no-install-recommends <your-package-list-here>
# Additional libraries and frameworks
RUN npm install -g typescript
# RUN npm install -g tsc
RUN npm install -g firebase-tools
RUN npm install -g @angular/cli
RUN npm install -g @nestjs/cli

# [Optional] Uncomment this section to install additional OS packages.

# RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
#     && apt-get -y install --no-install-recommends <your-package-list-here>
# Install JAVA

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    openjdk-11-jre

# Prints installed java version, just for checking
RUN java --version

# Custom Environment Variables
ENV FIRESTORE_EMULATOR_HOST=localhost:9002