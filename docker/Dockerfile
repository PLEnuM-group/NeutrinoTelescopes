FROM ubuntu:23.10 as base
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && \
    apt-get -y --no-install-recommends install \
    build-essential \
    gcc \
    git \
    libssl-dev \
    python3-dev \
    python3-full \
    python3-pip \
    curl pipx

RUN curl -fsSL https://deb.nodesource.com/setup_20.x && \
    apt-get install -y nodejs

RUN apt-get install -y --no-install-recommends cmake \
    wget libgsl-dev pkg-config libhdf5-serial-dev libboost-all-dev && \
    apt-get autoremove --purge && \
    rm -rf /var/lib/apt/lists/* 
    #    wget https://github.com/Kitware/CMake/releases/download/v3.20.3/cmake-3.20.3.tar.gz && \
    #    tar -zxvf cmake-3.20.3.tar.gz && \
    #    cd cmake-3.20.3 && ./bootstrap && make -j4 && make install && \
ENV PATH="/root/.local/bin:${PATH}/"
RUN pipx install conan

    #conan profile new default --detect && conan profile update settings.compiler.libcxx=libstdc++11 default


#COPY build_proposal_tables.py /opt/PROPOSAL/build_proposal_tables.py
#RUN PYTHONPATH=$PYTHONPATH:/opt/PROPOSAL/build/src/pyPROPOSAL python3 /opt/PROPOSAL/build_proposal_tables.py
#RUN wget https://nodejs.org/dist/v14.17.0/node-v14.17.0-linux-x64.tar.xz && \
#    mkdir -p /usr/local/lib/nodejs && \
#    tar -xJvf node-v14.17.0-linux-x64.tar.xz -C /usr/local/lib/nodejs && \
#    apt-get update && apt-get -y --no-install-recommends install npm && rm -rf /var/lib/apt/lists

#ADD "https://www.random.org/cgi-bin/randbyte?nbytes=10&format=h" skipcache
#RUN mkdir -p /usr/local/lib/MCEq && cd /usr/local/lib/MCEq && \
#    git clone -b next_1_3_X https://github.com/afedynitch/MCEq.git . && \
#    pip install -e .[CUDA]
#COPY mceq_db_lext_dpm191_v131.h5 /usr/local/lib/MCEq/
#RUN python -c "from MCEq.core import MCEqRun"


RUN mkdir -p /usr/local/cfitsio/source && cd /usr/local/cfitsio/source && \
    wget http://heasarc.gsfc.nasa.gov/FTP/software/fitsio/c/cfitsio-4.1.0.tar.gz && \
    tar -xvf cfitsio-4.1.0.tar.gz && cd cfitsio-4.1.0 && \
    ./configure --prefix /usr/local && make -j4 && make install && \
    rm -rf /usr/local/cfitsio/source && \
    mkdir -p /usr/local/photospline/source && cd /usr/local/photospline/source && \
    git clone https://github.com/icecube/photospline.git . && \
    cmake . -DCMAKE_INSTALL_PREFIX=/usr/local && make install && \
    mkdir -p /usr/local/LeptonInjector/source && cd /usr/local/LeptonInjector/source && \
    git clone https://github.com/icecube/LeptonInjector.git . && \
    cd .. && mkdir build && cd build && \
    cmake -DCMAKE_INSTALL_PREFIX=/usr/local ../source && \
    make -j4 && make install
#COPY LeptonInjectorPatch.diff /usr/local/LeptonInjector/source
#RUN cd /usr/local/LeptonInjector/source && git apply LeptonInjectorPatch.diff && \
    

ENV LD_LIBRARY_PATH="/usr/local/cuda-11/lib64/:${LD_LIBRARY_PATH}/"
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN pip install proposal
RUN pip install jupyterlab awkward numba seaborn tqdm jupyterlab_code_formatter black scipy numpy h5py

RUN curl -fsSL https://install.julialang.org | sh -s -- --yes --default-channel release
ENV PATH="/root/.juliaup/bin:$PATH"

RUN mkdir -p /opt/SQuIDS && \
    cd /opt/SQuIDS && \
    git clone https://github.com/jsalvado/SQuIDS.git . && \
    ./configure && make && make install && \
    mkdir -p /opt/nuSQuIDS && \
    cd /opt/nuSQuIDS && \
    git clone https://github.com/arguelles/nuSQuIDS.git . && \
    ./configure --with-python-bindings --with-squids=/usr/local/ && \
    make && make install && LD_LIBRARY_PATH=/opt/conda/lib/:$LD_LIBRARY_PATH make python && make python-install && \
    julia -e "using Pkg; Pkg.Registry.add(\"General\"); Pkg.Registry.add(RegistrySpec(url=\"https://github.com/PLEnuM-group/julia-registry.git\")); Pkg.add([\"PhysicsTools\", \"PhotonPropagation\"]); Pkg.add(url=\"https://github.com/PLEnuM-group/NeutrinoTelescopes.git\");"


RUN echo "export PATH=${VIRTUAL_ENV}/bin:${PATH}" >> /root/.bashrc && \
    echo "export PYTHONPATH=${PYTHONPATH}:/opt/PROPOSAL/build/src/pyPROPOSAL:/usr/local/lib/:/workspaces/repos/hyperion:/workspaces/repos/olympus:/usr/local/lib/python3/dist-packages/:/opt/pyprob" >> /root/.bashrc && \
    echo "export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/cuda-11/lib64/:/opt/LeptonInjector/install:${LD_LIBRARY_PATH}" >> /root/.bashrc && \
    echo "export EARTH_PARAMS=/usr/local/LeptonInjector/source" >> /root/.bashrc 


ENTRYPOINT [ "/bin/bash" ]
#CMD tensorboard --port 8008 --logdir=/tmp/tensorboard --bind_all & \
#    PYTHONPATH=${PYTHONPATH}:/opt/PROPOSAL/build/src/pyPROPOSAL:/usr/lib/nuSQuIDS/resources/python/bindings/ jupyter lab --port=8888 --no-browser --ip=0.0.0.0 --allow-root --notebook-dir=/app