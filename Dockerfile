FROM php:apache
COPY . /var/www/html
RUN chmod a+rwx -R /var/www/html
WORKDIR /var/www/html
RUN apt-get update \
  && apt-get install -y python3-pip python3-dev \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python \
  && pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
RUN chmod a+rwx -R /var/www
