FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime

## Include the following line if you have a requirements.txt file.
RUN mkdir /physionet
COPY ./ /physionet
WORKDIR /physionet

RUN pip install -r requirements.txt

#CMD ["/bin/bash"]

