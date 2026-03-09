wget https://github.com/numactl/numactl/releases/download/v2.0.16/numactl-2.0.16.tar.gz
tar xzf numactl-2.0.16.tar.gz
cd numactl-2.0.16
# Build to a local prefix
./configure --prefix=$HOME/.local
make
make install
# Point compiler and linker to it (add to ~/.bashrc for persistence)
export CPATH=$HOME/.local/include:$CPATH
export LIBRARY_PATH=$HOME/.local/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$HOME/.local/lib:$LD_LIBRARY_PATH


source .skyrl-venv/bin/activate
uv sync --active --extra fsdp


