#!/usr/bin/env bash
# Real-CDC Coach demo: CockroachDB changefeed -> Debezium -> Kafka -> Coach.
#
# Brings up Kafka + Kafka Connect, registers the Debezium CockroachDB source
# on the spending_signals table, and prints how to run the app against it.
# CockroachDB itself is whatever the app already uses; point at it with the
# env vars below.
#
#   CRDB_HOST             default host.docker.internal (local host node)
#   CRDB_PORT             default 26257
#   CRDB_USER             default root
#   CRDB_DB               default defaultdb
#   CHANGEFEED_KAFKA_URI  default kafka://localhost:29092 (URI the CRDB
#                         process itself pushes the changefeed to;
#                         kafka://kafka:9092 when CRDB runs in containers)
#   CRDB_CONTAINERS       the CockroachDB containers the changefeed runs on.
#                         Default: every running cockroachdb/cockroach
#                         container. Name them (space separated) on a
#                         machine that runs more than one cluster.
#
# Kafka joins the network of those containers, and so does Connect when it
# reaches the database by a container's name. The containers themselves join
# nothing. Take this stack down before the database's own
# (docker compose -f docker-compose.cdc.yml down): a network cannot be
# removed while one of these two is still on it.
#
# Connector plugin resolution order:
#   1. ./connect-plugins already populated
#   2. copy from ~/idea_workspace/debezium-cockroachdb-examples (local dev)
#   3. download from Maven Central
set -euo pipefail
cd "$(dirname "$0")"

echo "== 0/5 cockroachdb"
# The connector must watch the SAME database the app uses, so this script
# never creates its own CockroachDB. It does make sure one is running:
# a stopped repo container gets started, and if nothing is listening at
# all, the repo compose brings one up (data lives on a named volume).
if [ -n "${CRDB_HOST:-}" ]; then
    echo "   using CRDB_HOST=${CRDB_HOST} from the environment"
elif docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^banko-cockroachdb$'; then
    echo "   banko-cockroachdb already running"
elif (echo > /dev/tcp/localhost/26257) 2>/dev/null; then
    echo "   using the CockroachDB already listening on localhost:26257"
    echo "   (make sure this is the cluster your app points at)"
elif docker ps -a --format '{{.Names}}' 2>/dev/null | grep -q '^banko-cockroachdb$'; then
    docker start banko-cockroachdb >/dev/null
    echo "   started existing banko-cockroachdb container"
else
    (cd "$(git -C . rev-parse --show-toplevel 2>/dev/null || echo ../../..)" \
        && docker compose up -d cockroachdb)
    echo "   brought up the repo compose cockroachdb"
fi

# When CockroachDB runs in docker (the repo compose container or any other
# stack, like the chaos demo's cluster), the changefeed sink URI must be
# reachable FROM THE CRDB NODES, so it has to be the in-network kafka:9092;
# kafka://localhost:29092 would make each node dial itself. Step 2 puts
# Kafka on the nodes' own network under that name.
if [ -n "${CRDB_CONTAINERS+set}" ]; then
    CRDB_CONTAINERS="$(printf '%s\n' $CRDB_CONTAINERS)"
else
    CRDB_CONTAINERS="$(docker ps --format '{{.Names}} {{.Image}}' 2>/dev/null \
        | awk '$2 ~ /cockroachdb\/cockroach/ {print $1}')"
fi
if echo "$CRDB_CONTAINERS" | grep -q '^banko-cockroachdb$'; then
    CRDB_DEFAULT_HOST="banko-cockroachdb"
    KAFKA_URI_DEFAULT="kafka://kafka:9092"
elif [ -n "$CRDB_CONTAINERS" ]; then
    # Foreign cluster (chaos demo etc.): JDBC goes through the host's
    # published 26257 (haproxy on the chaos stack), the sink stays
    # in-network.
    CRDB_DEFAULT_HOST="host.docker.internal"
    KAFKA_URI_DEFAULT="kafka://kafka:9092"
else
    CRDB_DEFAULT_HOST="host.docker.internal"
    KAFKA_URI_DEFAULT="kafka://localhost:29092"
fi
CRDB_HOST="${CRDB_HOST:-$CRDB_DEFAULT_HOST}"
CRDB_PORT="${CRDB_PORT:-26257}"
CRDB_USER="${CRDB_USER:-root}"
CRDB_DB="${CRDB_DB:-defaultdb}"
CHANGEFEED_KAFKA_URI="${CHANGEFEED_KAFKA_URI:-$KAFKA_URI_DEFAULT}"
CONNECTOR_VERSION="${CONNECTOR_VERSION:-3.6.0.Final}"
# gvproxy keeps forwards for other stacks (a 5-node cluster's admin UIs
# occupy 8080-8084, for example). If the port is held by anything that is
# not our own connect container, hop to the next free one.
CONNECT_PORT="${CONNECT_PORT:-8084}"
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^banko-cdc-connect$'; then
    # Reuse whatever port the running container already publishes;
    # recreating it on the default collides with whoever took 8084.
    EXISTING_PORT="$(docker port banko-cdc-connect 8083/tcp 2>/dev/null \
        | head -1 | awk -F: '{print $NF}')"
    [ -n "$EXISTING_PORT" ] && CONNECT_PORT="$EXISTING_PORT"
else
    while (echo > /dev/tcp/localhost/${CONNECT_PORT}) 2>/dev/null; do
        echo "   port ${CONNECT_PORT} is taken; trying $((CONNECT_PORT+1))"
        CONNECT_PORT=$((CONNECT_PORT+1))
    done
fi
export CONNECT_PORT
EXAMPLES_REPO="${EXAMPLES_REPO:-$HOME/idea_workspace/debezium-cockroachdb-examples}"

echo "== 1/5 connector plugin"
if ls connect-plugins/debezium-connector-cockroachdb/*.jar >/dev/null 2>&1; then
    echo "   already present"
elif ls "$EXAMPLES_REPO"/crdb-to-crdb/connect-plugins/debezium-connector-cockroachdb/*.jar >/dev/null 2>&1; then
    mkdir -p connect-plugins
    cp -R "$EXAMPLES_REPO/crdb-to-crdb/connect-plugins/debezium-connector-cockroachdb" connect-plugins/
    echo "   copied from $EXAMPLES_REPO"
else
    ZIP="debezium-connector-cockroachdb-${CONNECTOR_VERSION}-plugin.zip"
    URL="https://repo1.maven.org/maven2/io/debezium/debezium-connector-cockroachdb/${CONNECTOR_VERSION}/${ZIP}"
    mkdir -p connect-plugins
    curl -fsSL -o "/tmp/${ZIP}" "$URL"
    unzip -q -o "/tmp/${ZIP}" -d connect-plugins/
    echo "   downloaded ${CONNECTOR_VERSION} from Maven Central"
fi

echo "== 2/5 kafka + connect"
if ! docker compose -f docker-compose.cdc.yml up -d; then
    cat <<'HINT'
Compose failed. If the error says "proxy already running", podman's port
forwarder is stuck (it happens after sleep or reboot). Recover with:
    podman machine stop && podman machine start
    docker compose -f docker-compose.cdc.yml down
    ./run-cdc-demo.sh
Databases live on named volumes; nothing is lost by the restart.
HINT
    exit 1
fi
if [ -n "$CRDB_CONTAINERS" ]; then
    # The changefeed runs on the CockroachDB nodes, so each of them has to
    # reach kafka:9092. Kafka joins their network under that name, and the
    # nodes are left as they are. This script used to join every node to
    # the cdc network instead. A container on two networks can come back
    # from a restart with its interfaces in the other order, and whatever
    # shapes traffic on eth0 (the chaos demo's latency between regions)
    # then shapes the wrong network.
    networks_of() {
        docker inspect -f '{{range $name, $_ := .NetworkSettings.Networks}}{{println $name}}{{end}}' "$1" 2>/dev/null \
            | grep -v '^$' || true
    }
    on_network() { networks_of "$2" | grep -qx "$1"; }
    reaches_kafka() { docker exec "$1" getent hosts kafka >/dev/null 2>&1; }
    for c in $CRDB_CONTAINERS; do
        if ! docker inspect "$c" >/dev/null 2>&1; then
            echo "   $c is not a container on this machine. Check CRDB_CONTAINERS."
            exit 1
        fi
    done
    # The default networks (docker's bridge, podman's podman) resolve no
    # names, so the broker could not be found there under any alias.
    KAFKA_NETWORKS=""
    for net in $(for c in $CRDB_CONTAINERS; do networks_of "$c"; done | sort -u); do
        case "$net" in cdc-demo_default|bridge|podman|host|none) continue ;; esac
        docker network connect --alias kafka "$net" banko-cdc-kafka 2>/dev/null || true
        if on_network "$net" banko-cdc-kafka; then
            KAFKA_NETWORKS="$KAFKA_NETWORKS $net"
        fi
    done
    if [ -n "$KAFKA_NETWORKS" ]; then
        echo "   kafka is on:$KAFKA_NETWORKS"
    fi
    for c in $CRDB_CONTAINERS; do
        shared=""
        for net in $KAFKA_NETWORKS; do
            if on_network "$net" "$c"; then shared="$net"; fi
        done
        if [ -n "$shared" ]; then
            if on_network cdc-demo_default "$c"; then
                # Put there by an earlier run of this script.
                docker network disconnect cdc-demo_default "$c" 2>/dev/null || true
                echo "   $c: taken off the cdc network. If it restarted while it was on both, restart it once more."
            fi
            if ! reaches_kafka "$c"; then
                echo "   $c cannot find kafka on $shared: that network resolves no names, and its changefeed"
                echo "      will not deliver. If nothing shapes this node's traffic, join it by hand:"
                echo "      docker network connect cdc-demo_default $c"
            fi
            continue
        fi
        # Its own network resolves no names: the cdc network is the only way
        # to the broker. Once there it stays, so a live changefeed is not cut.
        if ! on_network cdc-demo_default "$c"; then
            docker network connect cdc-demo_default "$c" 2>/dev/null || true
        fi
        if on_network cdc-demo_default "$c"; then
            echo "   $c: its own network resolves no names, so it is on the cdc network."
            echo "      Anything that shapes this node's traffic by interface name can pick the wrong one after a restart."
        else
            echo "   $c could not join the cdc network, and cannot find kafka: its changefeed will not deliver."
        fi
    done
    # The connector opens its own connection to the database. When that is
    # one of these containers, reached by its name, the connector has to be
    # on that container's network too.
    if echo "$CRDB_CONTAINERS" | grep -qx "$CRDB_HOST"; then
        for net in $(networks_of "$CRDB_HOST"); do
            case "$net" in cdc-demo_default|bridge|podman|host|none) continue ;; esac
            docker network connect "$net" banko-cdc-connect 2>/dev/null || true
            if on_network "$net" banko-cdc-connect; then
                echo "   connect is on: $net (it reaches $CRDB_HOST by name)"
            fi
        done
    fi
fi
printf "   waiting for connect"
for _ in $(seq 1 60); do
    if curl -fs http://localhost:${CONNECT_PORT}/connectors >/dev/null 2>&1; then
        echo " ... up"
        break
    fi
    printf "."
    sleep 2
done
curl -fs http://localhost:${CONNECT_PORT}/connectors >/dev/null 2>&1 || {
    echo "connect never came up; check: docker logs banko-cdc-connect"
    exit 1
}

echo "== 3/5 rangefeed on ${CRDB_HOST}:${CRDB_PORT}/${CRDB_DB}"
if [ -n "$CRDB_CONTAINERS" ]; then
    NODE="$(echo "$CRDB_CONTAINERS" | head -1)"
    printf "   waiting for the node"
    for _ in $(seq 1 30); do
        if docker exec "$NODE" cockroach sql --insecure \
            -e "SELECT 1" >/dev/null 2>&1; then
            break
        fi
        printf "."
        sleep 2
    done
    echo ""
    docker exec "$NODE" cockroach sql --insecure \
        -e "SET CLUSTER SETTING kv.rangefeed.enabled = true" >/dev/null
    echo "   enabled (via $NODE)"
else
    echo "   (host node: run SET CLUSTER SETTING kv.rangefeed.enabled = true;)"
fi

echo "== 4/5 register connector"
CONFIG=$(sed -e "s|\${CRDB_HOST}|$CRDB_HOST|" \
             -e "s|\${CRDB_PORT}|$CRDB_PORT|" \
             -e "s|\${CRDB_USER}|$CRDB_USER|" \
             -e "s|\${CRDB_DB}|$CRDB_DB|" \
             -e "s|\${CHANGEFEED_KAFKA_URI}|$CHANGEFEED_KAFKA_URI|" \
             spending-signals-source.json)
curl -fs -X DELETE http://localhost:${CONNECT_PORT}/connectors/banko-spending-signals-source >/dev/null 2>&1 || true
if ! echo "$CONFIG" | curl -fs -X POST -H 'Content-Type: application/json' \
    --data @- http://localhost:${CONNECT_PORT}/connectors >/dev/null; then
    echo "   connect refused the connector; check: docker logs banko-cdc-connect"
    exit 1
fi
sleep 3
echo "== 5/5 connector status"
curl -fs http://localhost:${CONNECT_PORT}/connectors/banko-spending-signals-source/status | python3 -m json.tool

cat <<'EOF'

Next:
  1. Run the app with the Kafka transport on:
       COACH_KAFKA_ENABLED=true KAFKA_BOOTSTRAP_SERVERS=localhost:29092 \
       AI_SERVICE=watsonx CDC_WEBHOOK_HMAC_SECRET=dev-only-secret \
       DATABASE_URL=<same db as CRDB_DB above> uv run banko-ai run
     (Banko in a container on the cluster's network: KAFKA_BOOTSTRAP_SERVERS=kafka:9092)
  2. Open http://localhost:5000/coach
  3. Insert a signal row and watch CockroachDB stream it into a nudge.
     The coach page shows the signed-in user's nudges, so target the
     username you signed up with:
       INSERT INTO spending_signals
         (user_id, signal_type, severity, payload, idempotency_key)
       VALUES
         ((SELECT user_id FROM users WHERE username = 'YOUR-USERNAME'),
          'budget_threshold', 'warn',
          '{"category": "dining", "pct_used": 0.82, "monthly_budget": 400.0,
            "spent_so_far": 328.0, "days_remaining": 9}',
          'demo:' || gen_random_uuid()::STRING);
EOF
