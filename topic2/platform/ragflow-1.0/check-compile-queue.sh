#!/bin/sh
set -eu

dataset=$1
case "$dataset" in
    *[!0-9a-f]*) exit 2 ;;
esac
[ "${#dataset}" -eq 32 ]

query() {
    docker exec -i topic2-ragflow-mysql-1 sh -c '
    export MYSQL_PWD="$MYSQL_ROOT_PASSWORD"
    exec mysql --host=127.0.0.1 --port=3306 -u root -D rag_flow -B
'
}

if [ "${2:-}" = "--requeue-stale-claim" ]; then
    query <<EOF
START TRANSACTION;
SELECT state, current_phase, claim_expires_at IS NOT NULL
INTO @state, @phase, @has_lease
FROM knowledge_compile_docs WHERE dataset_id='$dataset' FOR UPDATE;
SELECT COUNT(*), COUNT(DISTINCT doc_id) INTO @inflight, @distinct_docs
FROM knowledge_compile_docs_inflight WHERE dataset_id='$dataset';
SET @recover = @state='pending' AND @phase='completed' AND @has_lease=1
               AND @inflight=32 AND @distinct_docs=32;
SELECT COALESCE(MAX(queue_position),0) INTO @offset
FROM knowledge_compile_docs_backlog WHERE dataset_id='$dataset';
SELECT doc_id, queue_position, event_type FROM knowledge_compile_docs_inflight
WHERE dataset_id='$dataset' AND @recover ORDER BY queue_position;
INSERT INTO knowledge_compile_docs_backlog
    (dataset_id, doc_id, queue_position, event_type, variants, task_types)
SELECT dataset_id, doc_id, queue_position + @offset, event_type, variants, task_types
FROM knowledge_compile_docs_inflight WHERE dataset_id='$dataset' AND @recover;
SET @moved=ROW_COUNT();
DELETE FROM knowledge_compile_docs_inflight WHERE dataset_id='$dataset' AND @moved=32;
SET @deleted=ROW_COUNT();
UPDATE knowledge_compile_docs
SET claim_owner='', claim_token='', claim_expires_at=NULL
WHERE dataset_id='$dataset' AND @moved=32 AND @deleted=32;
SET @released=ROW_COUNT();
SET @finish = IF(@moved=32 AND @deleted=32 AND @released=1, 'COMMIT', 'ROLLBACK');
PREPARE finish FROM @finish;
EXECUTE finish;
DEALLOCATE PREPARE finish;
SELECT @state AS prior_state, @phase AS prior_phase, @moved AS moved, @released AS released, @finish AS result;
EOF
    exit
fi

query <<EOF
SELECT state, current_phase, TIMESTAMPDIFF(SECOND, NOW(3), claim_expires_at) AS lease_seconds,
       (SELECT COUNT(*) FROM knowledge_compile_docs_inflight WHERE dataset_id='$dataset') AS inflight,
       (SELECT COUNT(*) FROM knowledge_compile_docs_backlog WHERE dataset_id='$dataset') AS backlog
FROM knowledge_compile_docs WHERE dataset_id='$dataset';
SELECT 'inflight', COUNT(*), COUNT(DISTINCT doc_id), MIN(queue_position), MAX(queue_position)
FROM knowledge_compile_docs_inflight WHERE dataset_id='$dataset';
EOF
