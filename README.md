# hvd-benchmark

Directory organization

```bash
.
├── __init__.py
├── benchmarks 
│   ├── launcher.sh
│   ├── scripts
│   │   └── tf_cnn_benchmarks               [Models from TF models repo](https://github.com/tensorflow/models)
│   └── sm
│       ├── setup.py
│       └── sm_openmpi.py                    Helper function to run mpi in sagemaker
├── build_sagemaker_container.py             Helper function to build container with a custom script
├── docker
│   ├── Dockerfile                           HVD Dockerfile
│   ├── change-hostname.sh
│   ├── changehostname.c
│   └── parallel_updater.patch
|
├── install.sh                               Run benchmarks
├── requirements.txt
├── run_benchmarks.py
└── utils.py
```
