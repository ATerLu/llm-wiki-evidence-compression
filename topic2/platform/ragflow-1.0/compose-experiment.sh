#!/bin/sh
set -eu

cd /mnt/f/llm_wiki_research/topic2/platform/ragflow-1.0/release/docker
export SVR_WEB_HTTP_PORT=127.0.0.1:18780
export SVR_WEB_HTTPS_PORT=127.0.0.1:18743
export SVR_HTTP_PORT=127.0.0.1:19380
export ADMIN_SVR_HTTP_PORT=127.0.0.1:19381
export ES_PORT=127.0.0.1:19200
export EXPOSE_MYSQL_PORT=127.0.0.1:13306
export MINIO_PORT=127.0.0.1:19000
export MINIO_CONSOLE_PORT=127.0.0.1:19001
export KVROCKS_PORT=127.0.0.1:16379
export EXPOSE_NATS_PORT=127.0.0.1:14222
export EXPOSE_NATS_MONITORING_PORT=127.0.0.1:18222
export COMPOSE_PROFILES=cpu,elasticsearch,metadata-mysql
export RAGFLOW_IMAGE=infiniflow/ragflow:v1.0.0@sha256:ed8dff41a7530ed8fb037fe7479a7c535429d42fa8d00aec28b54bcbc056934d
exec docker compose --ansi never -p topic2-ragflow -f docker-compose.yml "$@"
