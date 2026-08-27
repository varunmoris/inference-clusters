"""Mutating live E2E — FSx for Lustre opt-in path.

Module-scoped fixture flips `enable_fsx=true`, reapplies, yields; reverts on teardown so
subsequent test sessions see the base state. All tests share one enable+revert cycle
(FSx has an hourly cost floor). Assertions cover the full chain: TF outputs populated,
FS + DRA reach AVAILABLE, CSI driver placed correctly, static PV/PVC bind, a consumer
pod pinned to the FSx AZ does an RWX round-trip against /models, and — via a
`FsxHydrate` CR against the platform-installed KRO RGD — a track-deploy-time
hydration Job byte-warms a seeded prefix and touches the sentinel workload
initContainers gate on (the lhnealreilly blocking-review path on d7cfd9c).
"""

import contextlib
import json
import time
import uuid
from collections.abc import Generator

import pytest
from pytest_jupyter_deploy.deployment import EndToEndDeployment
from pytest_jupyter_deploy.kubernetes.kubectl import run_kubectl

from tests.e2e import _serving_helpers as h

FSX_NAMESPACE = "kube-system"
# `app=fsx-csi-controller` / `app=fsx-csi-node` are the POD-TEMPLATE labels the chart
# stamps; the parent Deployment/DaemonSet themselves carry only `app.kubernetes.io/*`
# labels. Use the POD label for pod-level assertions and the DaemonSet's own NAME for
# the daemonset-level query.
FSX_CSI_CONTROLLER_LABEL = "app=fsx-csi-controller"
FSX_CSI_NODE_DS_NAME = "fsx-csi-node"
FSX_CONSUMER_POD = "fsx-consumer-e2e"


@pytest.fixture(scope="module")
def fsx_enabled(e2e_deployment: EndToEndDeployment) -> Generator[EndToEndDeployment, None, None]:
    """Enable FSx once for the module, then revert to base state at teardown.

    Both the enable and the revert are full reconfigure+reapply passes and each takes
    non-trivial minutes; wrapping them in a module-scoped fixture runs them exactly twice
    regardless of how many tests consume it. The revert is in `finally` so a mid-test
    failure still tears the file system down."""
    e2e_deployment.ensure_deployed()
    e2e_deployment.update_override_value("enable_fsx", True)
    # FSx PERSISTENT_2 file-system creation is slow (~10-20 min) and the DRA add another
    # few — give the apply a generous ceiling.
    e2e_deployment.ensure_deployed_with([], timeout_seconds=2400)
    try:
        yield e2e_deployment
    finally:
        e2e_deployment.update_override_value("enable_fsx", False)
        e2e_deployment.ensure_deployed_with([], timeout_seconds=2400)


def _await_fsx_available(region: str, fs_id: str, timeout_s: int = 1800) -> None:
    """Poll DescribeFileSystems until Lifecycle=AVAILABLE; defence-in-depth for a
    mid-run describe race, fail loud on terminal Lifecycles."""
    deadline = time.time() + timeout_s
    lifecycle = "UNKNOWN"
    while time.time() < deadline:
        systems = h.fsx_client(region).describe_file_systems(FileSystemIds=[fs_id])["FileSystems"]
        assert systems, f"DescribeFileSystems returned no results for {fs_id}"
        lifecycle = systems[0].get("Lifecycle", "UNKNOWN")
        if lifecycle == "AVAILABLE":
            return
        if lifecycle in ("FAILED", "DELETING", "MISCONFIGURED"):
            raise AssertionError(f"FSx file system {fs_id} entered terminal Lifecycle={lifecycle}")
        time.sleep(15)
    raise AssertionError(f"FSx file system {fs_id} never reached AVAILABLE (last Lifecycle={lifecycle})")


def _await_dra_available(region: str, fs_id: str, timeout_s: int = 900) -> dict:
    """Poll DescribeDataRepositoryAssociations until Lifecycle=AVAILABLE; return the assoc."""
    deadline = time.time() + timeout_s
    lifecycle = "UNKNOWN"
    while time.time() < deadline:
        associations = h.fsx_client(region).describe_data_repository_associations(
            Filters=[{"Name": "file-system-id", "Values": [fs_id]}],
        )["Associations"]
        if associations:
            association = dict(associations[0])
            lifecycle = str(association.get("Lifecycle", "UNKNOWN"))
            if lifecycle == "AVAILABLE":
                return association
            if lifecycle in ("FAILED", "DELETING", "MISCONFIGURED"):
                raise AssertionError(f"DRA on {fs_id} entered terminal Lifecycle={lifecycle}")
        time.sleep(10)
    raise AssertionError(f"DRA on {fs_id} never reached AVAILABLE (last Lifecycle={lifecycle})")


@pytest.mark.mutating
def test_fsx_outputs_and_control_plane(fsx_enabled: EndToEndDeployment) -> None:
    """Terraform outputs flip populated when FSx is on, and the FS + DRA reach AVAILABLE.

    This is the control-plane half of the mutating cycle: it never contacts the cluster.
    A break here (empty outputs, stuck lifecycle, DRA pointing at the wrong prefix) means
    the Terraform wiring in platform_fsx.tf regressed."""
    # Terraform's tostring(true) yields the lowercase "true" (not Python's "True").
    assert h.jd_output(fsx_enabled, "fsx_enabled") == "true", (
        "fsx_enabled output must be 'true' after flipping enable_fsx=true"
    )
    fs_id = h.jd_output(fsx_enabled, "fsx_file_system_id")
    dns = h.jd_output(fsx_enabled, "fsx_dns_name")
    mount_name = h.jd_output(fsx_enabled, "fsx_mount_name")
    az = h.jd_output(fsx_enabled, "fsx_availability_zone")
    dra_path = h.jd_output(fsx_enabled, "fsx_data_repository_path")
    model_store = h.jd_output(fsx_enabled, "model_store_bucket")
    region = h.jd_output(fsx_enabled, "region")

    assert fs_id.startswith("fs-"), f"expected fsx_file_system_id to look like fs-<id>, got {fs_id!r}"
    assert dns.endswith("amazonaws.com"), f"expected FSx DNS to be an AWS hostname, got {dns!r}"
    assert mount_name, "fsx_mount_name output must be non-empty"
    # AZ is <region><letter>, e.g. us-west-2a — must be in the deployment region.
    assert az.startswith(region), f"FSx AZ ({az}) must be inside the deployment region ({region})"
    # DRA points at s3://<model_store>/models/ (trailing slash matters).
    assert dra_path == f"s3://{model_store}/models/", (
        f"DRA path must point at s3://<model_store>/models/, got {dra_path!r}"
    )

    _await_fsx_available(region, fs_id)
    dra = _await_dra_available(region, fs_id)
    assert dra.get("DataRepositoryPath") == f"s3://{model_store}/models/", dra
    # DRA maps Lustre root to the S3 `models/` prefix. The FSx CSI PV mounts
    # Lustre root at the pod's mountpoint (typically /models), so an S3 object
    # `models/foo.bin` appears at pod path `/models/foo.bin` — same layout as
    # the S3-mount PV. Any other value (notably `/models`) double-nests the
    # imported content and silently no-ops hydration.
    assert dra.get("FileSystemPath") == "/", (
        f"DRA FileSystemPath must be / (so Lustre root maps directly to the "
        f"S3 models/ prefix), got {dra.get('FileSystemPath')!r}"
    )


@pytest.mark.mutating
def test_fsx_csi_driver_installed_and_placed(
    fsx_enabled: EndToEndDeployment,
    kubernetes_cluster_login: None,
) -> None:
    """The aws-fsx-csi-driver Helm release is up: controller on system MNG, node-plugin DaemonSet ready.

    The controller Deployment is a control-loop pod → must land on the tainted system MNG.
    The node-plugin is a DaemonSet that tolerates all taints so a Karpenter GPU/CPU node
    can mount FSx PVCs; it must have desired == ready across every current node."""
    h.assert_pods_by_selector_on_system_mng(FSX_NAMESPACE, FSX_CSI_CONTROLLER_LABEL, "aws-fsx-csi-driver controller")

    # DaemonSet lookup by NAME (the DS's own metadata labels don't include the
    # pod-template `app=fsx-csi-node` label — a -l selector on it returns zero items).
    node_ds = run_kubectl(
        "get",
        "daemonset",
        FSX_CSI_NODE_DS_NAME,
        "-n",
        FSX_NAMESPACE,
        "-o",
        "jsonpath={.status.desiredNumberScheduled},{.status.numberReady}",
        check=True,
    ).stdout.strip()
    desired, _, ready = node_ds.partition(",")
    assert desired and desired == ready, (
        f"aws-fsx-csi-driver node DaemonSet not fully ready (desired={desired!r}, ready={ready!r})"
    )
    assert int(desired) >= 1, "expected at least one FSx CSI node-plugin pod scheduled"


@pytest.mark.mutating
def test_fsx_pv_pvc_bound_and_wired(
    fsx_enabled: EndToEndDeployment,
    kubernetes_cluster_login: None,
) -> None:
    """The storage chart's FSx PV + PVC render, bind, and carry the correct volumeHandle.

    Guards the chart-side wiring: the mountname/dnsname/fs-id from Terraform must flow
    through platform_storage.tf → the storage helm_release → the fsx-mount.yaml template
    and back out as a bound PV that the FSx CSI driver would actually mount.
    """
    workload_ns = h.jd_output(fsx_enabled, "workload_namespace")
    fs_id = h.jd_output(fsx_enabled, "fsx_file_system_id")
    mount_name = h.jd_output(fsx_enabled, "fsx_mount_name")
    dns = h.jd_output(fsx_enabled, "fsx_dns_name")

    # PVC name is baked into the chart values (fsx.claimName), not currently exposed as a
    # Terraform output. The value is stable ("model-store-fsx") and matches values.yaml.
    pvc_name = "model-store-fsx"

    pvc_phase = run_kubectl(
        "get",
        "pvc",
        pvc_name,
        "-n",
        workload_ns,
        "-o",
        "jsonpath={.status.phase}",
        check=True,
    ).stdout.strip()
    assert pvc_phase == "Bound", f"expected PVC {workload_ns}/{pvc_name} to be Bound, got {pvc_phase!r}"

    pv = json.loads(
        run_kubectl(
            "get",
            "pv",
            pvc_name,
            "-o",
            "json",
            check=True,
        ).stdout
    )
    csi = pv["spec"].get("csi", {})
    assert csi.get("driver") == "fsx.csi.aws.com", f"PV CSI driver must be fsx.csi.aws.com, got {csi}"
    assert csi.get("volumeHandle") == f"{fs_id}::{mount_name}", (
        f"PV volumeHandle must be <fs-id>::<mount-name>, got {csi.get('volumeHandle')!r}"
    )
    assert csi.get("volumeAttributes", {}).get("dnsname") == dns
    assert csi.get("volumeAttributes", {}).get("mountname") == mount_name
    assert "flock" in pv["spec"].get("mountOptions", []), (
        "FSx PV must mount with flock (POSIX file locks — SafeTensors mmap, sqlite, torch)"
    )
    access_modes = pv["spec"].get("accessModes", [])
    assert access_modes == ["ReadWriteMany"], f"FSx PV must be RWX, got accessModes={access_modes}"


@pytest.mark.mutating
def test_fsx_consumer_pod_mounts_and_readwrites(
    fsx_enabled: EndToEndDeployment,
    kubernetes_cluster_login: None,
) -> None:
    """Data-plane end-to-end: pod mounts /models RWX, sees a DRA-imported object at the
    documented path, does a POSIX read/write round-trip.

    Two things covered:
      1. DRA-visibility — the test seeds s3://<model_store>/models/<probe_dir>/probe.txt
         BEFORE applying the pod. The DRA's auto_import (NEW/CHANGED events) propagates
         the write into Lustre; the pod polls for the file at pod-path
         /models/<probe_dir>/probe.txt (the DOCUMENTED path) and asserts the content
         matches. A regression to DRA `file_system_path = "/models"` would surface the
         file at /models/models/<probe_dir>/probe.txt and time out this poll — that's
         the ~30-min roborev bug we walked past twice before catching. Guarded here now.
      2. POSIX RWX — SG rules allow Lustre RPC through, the CSI driver hands the mount
         to the pod, `flock` mount options are honored, the Lustre backing accepts a
         write. Pinned to the FSx AZ (single-AZ FS).
    """
    workload_ns = h.jd_output(fsx_enabled, "workload_namespace")
    zone = h.jd_output(fsx_enabled, "fsx_availability_zone")
    model_store = h.jd_output(fsx_enabled, "model_store_bucket")
    image = h.client_image(fsx_enabled)

    # Per-run unique probe dir + content. Uniqueness matters because two runs against
    # the same deployment (retry-after-failure) shouldn't reuse a stale Lustre entry.
    # Alphanumeric only — string.Template substitution in apply_resource() balks on
    # anything shell-active, and Lustre paths must stay clean.
    probe_id = uuid.uuid4().hex[:12]
    probe_dir = f"e2e-dra-probe-{probe_id}"
    probe_key = f"models/{probe_dir}/probe.txt"
    probe_content = f"dra-probe-{probe_id}"

    run_kubectl(
        "delete",
        "pod",
        FSX_CONSUMER_POD,
        "-n",
        workload_ns,
        "--ignore-not-found",
        "--wait=false",
        check=False,
    )
    # Seed BEFORE the pod applies. auto_import events propagate within seconds; the
    # pod's polling loop tolerates a few minutes as a defensive margin.
    h.s3_put_object(model_store, probe_key, probe_content.encode())
    try:
        h.apply_resource(
            "fsx-consumer.yaml",
            image=image,
            namespace=workload_ns,
            claim_name="model-store-fsx",
            zone=zone,
            probe_dir=probe_dir,
            probe_content=probe_content,
        )
        # First-time mount can be slow: Karpenter must provision a CPU node in the FSx AZ
        # AND the FSx CSI node plugin must attach the Lustre client — budget 10 min.
        run_kubectl(
            "wait",
            "--for=jsonpath={.status.phase}=Succeeded",
            f"pod/{FSX_CONSUMER_POD}",
            "-n",
            workload_ns,
            "--timeout=600s",
            check=True,
        )
        logs = run_kubectl("logs", FSX_CONSUMER_POD, "-n", workload_ns, check=True).stdout
        assert "[fsx-consumer] OK" in logs, (
            f"consumer pod completed but did not print the OK sentinel; logs tail:\n{logs[-2000:]}"
        )
        # Confirm the pod actually ran in the FSx AZ (defence against a stale/no-op
        # affinity — a broken selector would fail earlier as Unschedulable, but a matching
        # label on the wrong zone value would silently pass).
        node = run_kubectl(
            "get",
            "pod",
            FSX_CONSUMER_POD,
            "-n",
            workload_ns,
            "-o",
            "jsonpath={.spec.nodeName}",
            check=True,
        ).stdout.strip()
        assert node, "consumer pod has no nodeName — scheduling never completed"
        node_zone = run_kubectl(
            "get",
            "node",
            node,
            "-o",
            r"jsonpath={.metadata.labels.topology\.kubernetes\.io/zone}",
            check=True,
        ).stdout.strip()
        assert node_zone == zone, (
            f"consumer pod landed in {node_zone!r} but the FSx file system is in {zone!r} "
            "— every mount would pay inter-AZ transfer; check the nodeAffinity block"
        )
    finally:
        run_kubectl(
            "delete",
            "pod",
            FSX_CONSUMER_POD,
            "-n",
            workload_ns,
            "--ignore-not-found",
            "--wait=false",
            check=False,
        )
        # DRA auto_export_policy is empty, so the probe object stays in S3 (not
        # written to by Lustre). Delete it here so the model_store bucket is clean
        # for the next test run.
        with contextlib.suppress(Exception):
            h.s3_delete_object(model_store, probe_key)


@pytest.mark.mutating
def test_fsx_hydrate_cr_warms_prefix_and_writes_sentinel(
    fsx_enabled: EndToEndDeployment,
    kubernetes_cluster_login: None,
) -> None:
    """Track-deploy-time hydration path end-to-end: apply a FsxHydrate CR, KRO
    expands it to a Job that byte-warms a seeded prefix and drops the sentinel
    workload initContainers gate on.

    Replaces the earlier `jd up`-time TF-managed Job that never had weights to
    warm (lhnealreilly blocking review on d7cfd9c: "the onboarding is done after
    the cluster is already live … we would need this hydration done as an async
    job triggered when the deploy happens on the track resources"). The RGD
    lives in the storage chart (fsx-hydrate-rgd.yaml, gated on fsx.enabled), so
    the primitive is available the moment `enable_fsx=true` — a track just
    applies a CR of kind `FsxHydrate` next to its workload.

    Coverage:
      1. RGD is installed and Active — the storage chart wired it up correctly.
      2. Seeding s3://<model_store>/models/<prefix>/data.bin (bytes, not just
         namespace) + DRA auto_import surfaces it in Lustre.
      3. FsxHydrate CR triggers a Job that lands on the FSx AZ (nodeAffinity).
      4. Job Completes within its activeDeadlineSeconds.
      5. Sentinel `.hydrated-<slug>` exists at /models/ (workload initContainers'
         gating file).
      6. `lfs hsm_state` on the seeded file returns something other than
         "released" (bytes are actually on Lustre OSTs, not just metadata).
      7. Deleting the CR cascades the child Job away (the KRO reconcile property).
    """
    workload_ns = h.jd_output(fsx_enabled, "workload_namespace")
    zone = h.jd_output(fsx_enabled, "fsx_availability_zone")
    model_store = h.jd_output(fsx_enabled, "model_store_bucket")

    # 1. RGD installed by the storage chart is Active. If this fails, either the
    #    chart didn't render the template (fsx.enabled not passed) or KRO didn't
    #    reconcile it (helm_release.storage lost its helm_release.kro dep_on).
    run_kubectl(
        "wait",
        "--for=jsonpath={.status.state}=Active",
        "resourcegraphdefinition/fsx-hydrate",
        "--timeout=120s",
        check=True,
    )

    # 2. Per-run unique prefix so re-runs against the same deployment don't
    #    collide on stale Lustre entries / stale sentinels.
    run_id = uuid.uuid4().hex[:12]
    prefix = f"hydrate-e2e-{run_id}"
    slug = prefix  # no `/` or `_` in the prefix → SLUG == prefix
    cr_name = f"hydrate-{run_id}"  # DNS-1123
    job_name = f"fsx-hydrate-{cr_name}"  # RGD's Job name template
    seed_key = f"models/{prefix}/data.bin"
    # 4 KiB body — the DRT metadata layer only needs a file present; hsm_restore
    # cost is proportional to bytes but we only need it to complete once.
    seed_body = b"x" * 4096
    sentinel_path = f"/models/.hydrated-{slug}"

    def _dump_hydrate_diagnostics(reason: str) -> str:
        """kubectl-describe + kubectl-logs + recent events for the Job and any pod
        it produced. Called on wait-timeout so the pytest failure captures WHY the
        Job didn't Complete (dnf install failure, image pull error, Karpenter
        starvation) rather than just "wait timed out". Returned as a single string
        the AssertionError body embeds; nothing is asserted against it here."""
        parts = [f"[hydrate-diag] {reason}"]
        # Job describe: shows retry count, active/failed pods, condition/events.
        parts.append(
            "--- kubectl describe job ---\n"
            + run_kubectl("describe", "job", job_name, "-n", workload_ns, check=False).stdout
        )
        # Every pod the Job ever produced (including CrashLoopBackOff retries).
        pods = run_kubectl(
            "get",
            "pods",
            "-n",
            workload_ns,
            "-l",
            f"batch.kubernetes.io/job-name={job_name}",
            "-o",
            "jsonpath={.items[*].metadata.name}",
            check=False,
        ).stdout.split()
        for p in pods or []:
            parts.append(
                f"--- kubectl describe pod {p} ---\n"
                + run_kubectl("describe", "pod", p, "-n", workload_ns, check=False).stdout
            )
            parts.append(
                f"--- kubectl logs {p} (previous + current, tail) ---\n"
                + run_kubectl("logs", p, "-n", workload_ns, "--previous", "--tail=200", check=False).stdout
                + "\n---\n"
                + run_kubectl("logs", p, "-n", workload_ns, "--tail=200", check=False).stdout
            )
        # Recent namespace events — image pull failures, PVC bind stalls, etc.
        parts.append(
            "--- kubectl get events (last 30) ---\n"
            + run_kubectl("get", "events", "-n", workload_ns, "--sort-by=.lastTimestamp", check=False).stdout[-4000:]
        )
        # RGD status — did KRO reconcile the CR?
        parts.append(
            "--- kubectl get resourcegraphdefinition/fsx-hydrate -o yaml ---\n"
            + run_kubectl("get", "resourcegraphdefinition/fsx-hydrate", "-o", "yaml", check=False).stdout
        )
        return "\n\n".join(parts)

    h.s3_put_object(model_store, seed_key, seed_body)
    try:
        # 3. Apply the CR — KRO reconciles it into a Job.
        h.apply_resource(
            "fsx-hydrate-cr.yaml",
            namespace=workload_ns,
            cr_name=cr_name,
            prefix=prefix,
        )

        # 4. Job reaches Complete. First-time hydration budget: Karpenter must
        #    provision a CPU node in the FSx AZ (2-3 min), FSx CSI attaches the
        #    Lustre client (30-60s), the pod dnf-installs lustre2.15-client
        #    (~30s), then hsm_restore + hsm_state polling on a 4 KiB file
        #    finishes in seconds. 15-min ceiling is generous but keeps a stuck
        #    Job from wedging the mutating suite.
        try:
            run_kubectl(
                "wait",
                "--for=condition=Complete",
                f"job/{job_name}",
                "-n",
                workload_ns,
                "--timeout=900s",
                check=True,
            )
        except Exception as e:
            raise AssertionError(
                f"hydration Job {job_name} did not reach condition=Complete within 15m.\n"
                f"{_dump_hydrate_diagnostics('Job wait timeout')}\n"
                f"original error: {e}"
            ) from e

        # Confirm the Job landed on the FSx AZ (defence against a stale
        # affinity block — a wrong-AZ node would still schedule but every mount
        # would pay inter-AZ transfer).
        job_pod = run_kubectl(
            "get",
            "pods",
            "-n",
            workload_ns,
            "-l",
            f"batch.kubernetes.io/job-name={job_name}",
            "-o",
            "jsonpath={.items[0].spec.nodeName}",
            check=True,
        ).stdout.strip()
        assert job_pod, f"hydration Job {job_name} produced no pod"
        job_zone = run_kubectl(
            "get",
            "node",
            job_pod,
            "-o",
            r"jsonpath={.metadata.labels.topology\.kubernetes\.io/zone}",
            check=True,
        ).stdout.strip()
        assert job_zone == zone, (
            f"hydration Job landed in {job_zone!r} but FSx is in {zone!r} — "
            "AZ affinity in fsx-hydrate-rgd.yaml regressed"
        )

        # 5. Probe via a short-lived busybox pod pinned to the FSx AZ: assert
        #    the sentinel exists. The Job only touches the sentinel after every
        #    file has been fully read (Lustre HSM restore is synchronous on
        #    read), so sentinel presence is the entire contract — no separate
        #    hsm_state check needed.
        probe_pod = f"fsx-hydrate-probe-{run_id}"
        h.apply_resource(
            "fsx-hydrate-probe.yaml",
            pod=probe_pod,
            namespace=workload_ns,
            image=h.client_image(fsx_enabled),
            zone=zone,
            claim_name="model-store-fsx",
            sentinel_path=sentinel_path,
        )
        try:
            try:
                run_kubectl(
                    "wait",
                    "--for=jsonpath={.status.phase}=Succeeded",
                    f"pod/{probe_pod}",
                    "-n",
                    workload_ns,
                    "--timeout=300s",
                    check=True,
                )
            except Exception as e:
                describe = run_kubectl("describe", "pod", probe_pod, "-n", workload_ns, check=False).stdout
                logs = run_kubectl("logs", probe_pod, "-n", workload_ns, "--tail=200", check=False).stdout
                raise AssertionError(
                    f"probe pod {probe_pod} did not reach Succeeded within 5m.\n"
                    f"--- describe ---\n{describe}\n--- logs ---\n{logs}\n"
                    f"original error: {e}"
                ) from e
            logs = run_kubectl("logs", probe_pod, "-n", workload_ns, check=True).stdout
            assert "[probe] sentinel OK" in logs, (
                f"probe pod completed but sentinel {sentinel_path} was missing; logs:\n{logs[-2000:]}"
            )
        finally:
            run_kubectl(
                "delete",
                "pod",
                probe_pod,
                "-n",
                workload_ns,
                "--ignore-not-found",
                "--wait=false",
                check=False,
            )

        # 7. Deleting the CR cascades the Job away — KRO reconcile lifecycle.
        run_kubectl(
            "delete",
            "fsxhydrate",
            cr_name,
            "-n",
            workload_ns,
            "--timeout=120s",
            check=True,
        )
        cascaded = False
        for _ in range(24):  # ~2 min for the cascade
            got = run_kubectl(
                "get",
                "job",
                job_name,
                "-n",
                workload_ns,
                "--ignore-not-found",
                "-o",
                "name",
                check=False,
            )
            if not got.stdout.strip():
                cascaded = True
                break
            time.sleep(5)
        assert cascaded, "deleting the FsxHydrate CR must cascade-delete its child Job (KRO reconcile)"
    finally:
        # CR + probe pod may have leaked past an assertion — best-effort cleanup.
        run_kubectl(
            "delete",
            "fsxhydrate",
            cr_name,
            "-n",
            workload_ns,
            "--ignore-not-found",
            "--wait=false",
            check=False,
        )
        # DRA auto_export is off — the seed object stays in S3 until we clear it.
        with contextlib.suppress(Exception):
            h.s3_delete_object(model_store, seed_key)
