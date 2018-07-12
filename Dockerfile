FROM tensorflow/tensorflow:latest-gpu

MAINTAINER Sami Ben Guedria (BGS)

RUN apt update && apt install -y git
RUN pip uninstall -y tensorflow-gpu
RUN pip install tf-nightly-gpu
RUN git clone https://github.com/ElementAI/tf_benchmarks.git

WORKDIR /notebooks/tf_benchmarks/scripts/tf_cnn_benchmarks
RUN chmod +x gpu_bench.py && ln gpu_bench.py gpu_bench


#ENTRYPOINT [ "/root/gpu_bench" ]
