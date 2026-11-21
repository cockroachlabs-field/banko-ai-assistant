"""scripts/coach/cdc-demo/run-cdc-demo.sh decides which container joins which
network. Run here against a docker that only keeps notes: the script's
decisions are what matter, and they went wrong twice without a test (nodes
joined to a second network lost the chaos demo's latency shaping after a
restart; nodes left alone left the connector unable to find the database)."""

import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts" / "coach" / "cdc-demo"
CDC = "cdc-demo_default"
NODE_IMAGE = "cockroachdb/cockroach:v26.4.1"
DEBEZIUM = "3.7.1.Final"            # the Connect image and the connector plugin, by default

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")

# A docker that answers from a JSON file and writes down what it was asked to
# change. Containers have networks; a network resolves names or does not; the
# broker is "kafka" on the cdc network and wherever it joined under that alias.
DOCKER = r'''#!__PYTHON__
import json, sys
from pathlib import Path

path = Path(__file__).with_name("state.json")
state = json.loads(path.read_text())
args = sys.argv[1:]
boxes = state["containers"]


def save():
    path.write_text(json.dumps(state))


def out(lines):
    sys.stdout.write("".join(f"{line}\n" for line in lines))


if args[:1] == ["ps"]:
    names = sorted(boxes) if "-a" in args else sorted(n for n in boxes if boxes[n].get("running", True))
    fmt = args[args.index("--format") + 1]
    out(f"{n} {boxes[n]['image']}" if "Image" in fmt else n for n in names)
elif args[:1] == ["inspect"]:
    name = args[-1]
    if name not in boxes:
        sys.exit(1)
    if "-f" in args:
        out(boxes[name]["networks"])
elif args[:2] == ["network", "connect"]:
    rest = args[2:]
    alias = None
    if rest[0] == "--alias":
        alias, rest = rest[1], rest[2:]
    net, name = rest
    if name not in boxes or net in boxes[name]["networks"] or (alias and net in state["no_alias"]):
        sys.exit(1)
    boxes[name]["networks"].append(net)
    if alias:
        state["aliases"].setdefault(net, []).append(alias)
    state["changes"].append(["connect", net, name, alias])
    save()
elif args[:2] == ["network", "disconnect"]:
    net, name = args[2:]
    if name not in boxes or net not in boxes[name]["networks"]:
        sys.exit(1)
    boxes[name]["networks"].remove(net)
    state["changes"].append(["disconnect", net, name, None])
    save()
elif args[:1] == ["exec"]:
    name, command = args[1], args[2:]
    if command[:2] == ["getent", "hosts"]:
        host = command[2]
        found = any(net in state["dns"] and (host in state["aliases"].get(net, []))
                    for net in boxes[name]["networks"])
        sys.exit(0 if found else 2)
elif args[:1] == ["port"]:
    out(["0.0.0.0:18084"])
elif args[:1] == ["restart"]:
    state["changes"].append(["restart", args[1], None, None])
    save()
elif args[:1] == ["compose"]:
    import os
    state["compose_env"] = {k: v for k, v in os.environ.items() if k in ("DEBEZIUM_VERSION", "CONNECT_PORT")}
    save()
'''

CURL = r'''#!__PYTHON__
import json, sys
from pathlib import Path
args = sys.argv[1:]
if any(a.endswith("/status") for a in args):
    print('{"name": "banko-spending-signals-source", "connector": {"state": "RUNNING"}}')
elif "POST" in args:
    sys.stdin.read()
elif "-o" in args:                                   # a download: keep the address, leave a file behind
    path = Path(__file__).with_name("state.json")
    state = json.loads(path.read_text())
    state.setdefault("downloads", []).append(args[-1])
    path.write_text(json.dumps(state))
    Path(args[args.index("-o") + 1]).write_text("a plugin archive")
'''

# unzip -q -o <archive> -d <dir>: what the plugin archive from Maven Central holds, as far as the script cares.
UNZIP = r'''#!__PYTHON__
import re, sys
from pathlib import Path
archive, target = sys.argv[3], Path(sys.argv[5])
version = re.search(r"debezium-connector-cockroachdb-(.+)-plugin\.zip$", archive).group(1)
plugin = target / "debezium-connector-cockroachdb"
plugin.mkdir(parents=True, exist_ok=True)
(plugin / f"debezium-connector-cockroachdb-{version}.jar").write_text("")
'''


def _box(image: str, *networks: str) -> dict:
    return {"image": image, "networks": list(networks)}


def _run(tmp_path, containers: dict, *, env: dict | None = None, dns=("crdb-chaos", "banko-network", CDC),
         no_alias=("bridge",), plugin_version: str | None = DEBEZIUM):
    """Run the script against the pretend docker. Returns (exit code, output, changes, containers).
    plugin_version is the connector already in connect-plugins, None for an empty directory."""
    work = tmp_path / "cdc-demo"
    shutil.copytree(SCRIPT_DIR, work, ignore=shutil.ignore_patterns("connect-plugins"))
    if plugin_version:
        plugin = work / "connect-plugins" / "debezium-connector-cockroachdb"
        plugin.mkdir(parents=True)
        (plugin / f"debezium-connector-cockroachdb-{plugin_version}.jar").write_text("")
    tools = tmp_path / "bin"
    tools.mkdir()
    boxes = {"banko-cdc-kafka": _box("confluentinc/cp-kafka:7.4.0", CDC),
             "banko-cdc-connect": _box(f"quay.io/debezium/connect:{DEBEZIUM}", CDC), **containers}
    (tools / "state.json").write_text(json.dumps({
        "containers": boxes, "dns": list(dns), "no_alias": list(no_alias),
        "aliases": {CDC: ["kafka", "connect"]}, "changes": []}))
    for name, body in (("docker", DOCKER), ("curl", CURL), ("unzip", UNZIP), ("sleep", "#!/bin/sh\n")):
        tool = tools / name
        tool.write_text(body.replace("__PYTHON__", sys.executable))
        tool.chmod(tool.stat().st_mode | stat.S_IXUSR)
    environ = {"PATH": f"{tools}{os.pathsep}{os.environ['PATH']}", "HOME": str(tmp_path),
               "EXAMPLES_REPO": str(tmp_path / "none"), **(env or {})}
    done = subprocess.run(["bash", str(work / "run-cdc-demo.sh")], env=environ, capture_output=True, text=True,
                          timeout=120)
    state = json.loads((tools / "state.json").read_text())
    return done.returncode, done.stdout + done.stderr, [tuple(c) for c in state["changes"]], state["containers"]


def test_the_repo_compose_database_is_reached_by_name_from_both_sides(tmp_path):
    """docker compose up -d, then the script: the changefeed runs on the node
    and needs the broker, the connector needs the node, each by name. The
    pipeline's containers join the node's network; the node joins nothing."""
    code, said, changes, boxes = _run(tmp_path, {"banko-cockroachdb": _box(NODE_IMAGE, "banko-network")})
    assert code == 0, said
    assert ("connect", "banko-network", "banko-cdc-kafka", "kafka") in changes
    assert ("connect", "banko-network", "banko-cdc-connect", None) in changes
    assert boxes["banko-cockroachdb"]["networks"] == ["banko-network"]


def test_another_stacks_cluster_is_left_exactly_as_it_was(tmp_path):
    """The chaos demo shapes each node's traffic on its first interface: a
    node that joins a second network can come back from a restart with the
    two in the other order. The connector goes through the host there."""
    nodes = {f"us-east-1a-n{i}": _box(NODE_IMAGE, "crdb-chaos") for i in (1, 2, 3)}
    code, said, changes, boxes = _run(tmp_path, nodes)
    assert code == 0, said
    assert changes == [("connect", "crdb-chaos", "banko-cdc-kafka", "kafka")]
    assert all(boxes[n]["networks"] == ["crdb-chaos"] for n in nodes)


def test_a_node_an_earlier_run_joined_to_the_cdc_network_is_taken_off_it(tmp_path):
    code, said, changes, boxes = _run(tmp_path, {"us-east-1a-n1": _box(NODE_IMAGE, "crdb-chaos", CDC)})
    assert code == 0, said
    assert boxes["us-east-1a-n1"]["networks"] == ["crdb-chaos"]
    assert "restart it once more" in said


def test_a_node_whose_network_resolves_no_names_uses_the_cdc_network_and_is_not_bounced(tmp_path):
    """docker's default bridge resolves no names, so that node can only find
    the broker on the cdc network. It joins once; the next run, with it
    already there, must not take it off and put it back under a live
    changefeed."""
    code, said, changes, boxes = _run(tmp_path, {"lonely-roach": _box(NODE_IMAGE, "bridge")}, dns=(CDC,))
    assert code == 0, said
    assert changes == [("connect", CDC, "lonely-roach", None)]
    assert "cdc network" in said

    again = tmp_path / "again"
    again.mkdir()
    code, said, changes, boxes = _run(again, {"lonely-roach": _box(NODE_IMAGE, "bridge", CDC)}, dns=(CDC,))
    assert code == 0, said
    assert changes == []


def test_a_container_that_does_not_exist_is_named_not_met_with_silence(tmp_path):
    """Under set -euo pipefail a grep that printed nothing ended the script
    right after compose came up, with no word about why."""
    code, said, changes, _ = _run(tmp_path, {}, env={"CRDB_CONTAINERS": "roach-typo", "CRDB_HOST": "localhost"})
    assert code != 0 and "roach-typo" in said
    assert changes == []


def test_a_node_that_still_cannot_find_the_broker_is_said_so(tmp_path):
    """A network someone made without name resolution: kafka joins it, and
    the node still cannot find it. The script says what to do; it does not
    report success."""
    code, said, changes, boxes = _run(tmp_path, {"quiet-roach": _box(NODE_IMAGE, "no-dns-net")},
                                      dns=(CDC,), env={"CRDB_HOST": "host.docker.internal"})
    assert code == 0, said
    assert "quiet-roach" in said and "cannot find kafka" in said


def test_named_containers_keep_the_script_away_from_every_other_cluster(tmp_path):
    """A machine that runs more than one cluster: CRDB_CONTAINERS names the
    one the changefeed runs on, and nothing else is touched."""
    boxes = {"banko-cockroachdb": _box(NODE_IMAGE, "banko-network"),
             **{f"us-east-1a-n{i}": _box(NODE_IMAGE, "crdb-chaos") for i in (1, 2)}}
    code, said, changes, after = _run(tmp_path, boxes, env={"CRDB_CONTAINERS": "banko-cockroachdb"})
    assert code == 0, said
    assert all("crdb-chaos" not in change for change in changes)
    assert after["us-east-1a-n1"]["networks"] == ["crdb-chaos"]


# ---- which Debezium, and which CockroachDB ---------------------------------

def _plugin_and_downloads(tmp_path) -> tuple[list[str], list[str]]:
    jars = sorted(p.name for p in (tmp_path / "cdc-demo" / "connect-plugins").rglob("debezium-connector-cockroachdb-*.jar"))
    return jars, json.loads((tmp_path / "bin" / "state.json").read_text()).get("downloads", [])


def test_connect_and_the_connector_are_debezium_3_7_1_final(tmp_path):
    """One version for the Connect image and the plugin it loads."""
    compose = (SCRIPT_DIR / "docker-compose.cdc.yml").read_text()
    assert f"quay.io/debezium/connect:${{DEBEZIUM_VERSION:-{DEBEZIUM}}}" in compose
    code, said, _, _ = _run(tmp_path, {"us-east-1a-n1": _box(NODE_IMAGE, "crdb-chaos")}, plugin_version=None)
    assert code == 0, said
    jars, downloads = _plugin_and_downloads(tmp_path)
    assert jars == [f"debezium-connector-cockroachdb-{DEBEZIUM}.jar"]
    assert downloads == ["https://repo1.maven.org/maven2/io/debezium/debezium-connector-cockroachdb/"
                         f"{DEBEZIUM}/debezium-connector-cockroachdb-{DEBEZIUM}-plugin.zip"]
    assert DEBEZIUM in said


def test_a_connector_of_another_version_is_replaced_not_used(tmp_path):
    """Whatever was in connect-plugins used to be taken as it was, and so
    was whatever the examples repository held: a snapshot build ran where a
    release was meant to."""
    code, said, _, _ = _run(tmp_path, {"us-east-1a-n1": _box(NODE_IMAGE, "crdb-chaos")}, plugin_version="3.7.2-SNAPSHOT")
    assert code == 0, said
    jars, downloads = _plugin_and_downloads(tmp_path)
    assert jars == [f"debezium-connector-cockroachdb-{DEBEZIUM}.jar"] and len(downloads) == 1
    assert "3.7.2-SNAPSHOT" in said                         # it says what it found and did not use


def test_the_examples_repository_is_used_only_for_the_version_asked_for(tmp_path):
    def examples(version: str) -> Path:
        repo = tmp_path / f"examples-{version}"
        plugin = repo / "crdb-to-crdb" / "connect-plugins" / "debezium-connector-cockroachdb"
        plugin.mkdir(parents=True)
        (plugin / f"debezium-connector-cockroachdb-{version}.jar").write_text("")
        return repo

    node = {"us-east-1a-n1": _box(NODE_IMAGE, "crdb-chaos")}
    first = tmp_path / "snapshot"
    first.mkdir()
    code, said, _, _ = _run(first, node, plugin_version=None, env={"EXAMPLES_REPO": str(examples("3.7.2-SNAPSHOT"))})
    assert code == 0, said
    assert _plugin_and_downloads(first) == ([f"debezium-connector-cockroachdb-{DEBEZIUM}.jar"], _plugin_and_downloads(first)[1])
    assert len(_plugin_and_downloads(first)[1]) == 1         # downloaded: the examples held another version

    second = tmp_path / "same"
    second.mkdir()
    code, said, _, _ = _run(second, node, plugin_version=None, env={"EXAMPLES_REPO": str(examples(DEBEZIUM))})
    assert code == 0, said
    assert _plugin_and_downloads(second) == ([f"debezium-connector-cockroachdb-{DEBEZIUM}.jar"], [])   # copied


def test_one_setting_moves_the_image_and_the_plugin_together(tmp_path):
    code, said, _, _ = _run(tmp_path, {"us-east-1a-n1": _box(NODE_IMAGE, "crdb-chaos")},
                            env={"DEBEZIUM_VERSION": "3.8.0.Alpha2"})
    assert code == 0, said
    jars, downloads = _plugin_and_downloads(tmp_path)
    assert jars == ["debezium-connector-cockroachdb-3.8.0.Alpha2.jar"] and "3.8.0.Alpha2" in downloads[0]
    assert json.loads((tmp_path / "bin" / "state.json").read_text())["compose_env"]["DEBEZIUM_VERSION"] == "3.8.0.Alpha2"


def test_a_running_connect_is_restarted_when_its_plugin_changed(tmp_path):
    """Connect loads its plugins when it starts. Swapped under a running
    one, the new connector would sit on disk and the old one keep running."""
    code, said, changes, _ = _run(tmp_path, {"us-east-1a-n1": _box(NODE_IMAGE, "crdb-chaos")},
                                  plugin_version="3.6.0.Final")
    assert code == 0, said
    assert ("restart", "banko-cdc-connect") in [c[:2] for c in changes]
    again = tmp_path / "again"
    again.mkdir()
    code, said, changes, _ = _run(again, {"us-east-1a-n1": _box(NODE_IMAGE, "crdb-chaos")})
    assert code == 0 and "restart" not in [c[0] for c in changes]     # nothing changed: nothing restarted


def test_a_single_node_started_here_is_cockroachdb_26_4_1():
    root = SCRIPT_DIR.parents[2]
    for compose in ("docker-compose.yml", "docker-compose.airgap.yml"):
        assert "cockroachdb/cockroach:${COCKROACHDB_VERSION:-v26.4.1}" in (root / compose).read_text(), compose
    assert "cockroachdb/cockroach:v26.4.1" in (root / "docs" / "DOCKER.md").read_text()
