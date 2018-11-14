FROM python:3.4
RUN pip install Flask==0.10.1 uWSGI==2.0.13 flask_uploads flask_wtf numpy tensorflow ipython pillow
WORKDIR /app
COPY app /app
COPY cmd.sh /
RUN groupadd -r uwsgi && useradd -r -g uwsgi uwsgi && chown -R uwsgi:uwsgi /app
EXPOSE 9090 9191
USER uwsgi
CMD ["/cmd.sh"]
