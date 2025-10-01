FROM bitnami/spark:3.5.5

USER root

COPY jars/hadoop-aws-3.3.2.jar /opt/spark/jars/
COPY jars/aws-java-sdk-bundle-1.11.901.jar /opt/spark/jars/

RUN useradd -u 1001 -ms /bin/bash spark
USER spark
ENV USER spark
ENV HOME /home/spark
