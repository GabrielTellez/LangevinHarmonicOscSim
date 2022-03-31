# Scripts to run jupyter notebook kernel remotely on a cluster

* Run on the cluster:
`sbatch remote-jupyter.sh`
Check the file `jupyter_log/jupyter-notebook-e*.log` to get the node (node_name) where the job is running

* On the local machine edit `remote-ssh-tunnel.sh` to add your username (USER=xxx) and hostname (HOST=xxx) of the managing node of the cluster, then
run:
`remote-ssh-tunnel.sh node_name`

* Open localhost on port 8879 to open jupyter. The URL is found on `jupyter_log/jupyter-notebook-e*.log`

