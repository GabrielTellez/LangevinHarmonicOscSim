#/bin/bash
node=$1
port=8879 
USER=
HOST=
ssh -N -L $port:$1:$port ${USER}@${HOST}

