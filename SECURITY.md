# Reporting Security Vulnerabilities

Report suspected vulnerabilities in Grana privately using NVIDIA's
[security vulnerability submission form](https://www.nvidia.com/object/submit-security-vulnerability.html)
or email [psirt@nvidia.com](mailto:psirt@nvidia.com). For sensitive email,
use the [NVIDIA public PGP key](https://www.nvidia.com/en-us/security/pgp-key/).
Include the repository URL, affected commit or version, vulnerability type,
reproduction steps or a proof of concept, and the potential security impact.
Do not publish vulnerability details in a public GitHub issue or pull request.

See [NVIDIA's PSIRT policies](https://www.nvidia.com/en-us/security/psirt-policies/)
for coordinated disclosure information and
[NVIDIA Product Security](https://www.nvidia.com/en-us/security/) for other
security-related concerns.

## Scope

This policy covers the Python training, inference, cache, and annotation
workflows and scripts in this source release.

Grana is intended for controlled research environments with trusted operators,
code, and inputs. Users and operators are responsible for assessing the risks
described below and securing their own deployments, including network isolation,
access controls, and the provenance of code, datasets, models, and configurations.

## Deployment boundaries

Grana's inference and annotation-review servers are local research tools.
They require an IPv4 loopback bind and do not provide application-level user
authentication or TLS. Access a remote server through an authenticated SSH
port forward. If you place a gateway in front of a service, keep the Grana
backend on loopback and enforce authentication, authorization, and TLS at the
gateway; only trusted users should reach it. Do not bypass the server's
loopback or request-validation checks to expose it publicly.

Inference review media is served from an explicit file allowlist. Only add
recordings approved for the intended reviewers. Review pages and API responses
may contain private task descriptions, images, annotations, or model outputs.
Restrict access to the host, forwarded ports, and artifact directories.

## Dependency warnings

The following dependency vulnerabilities may affect deployments depending on
exposed services, enabled functionality, and input sources. Users and operators
must evaluate these conditions in their own environments and apply the controls
described below.

### Ray jobs and data readers

Ray intentionally executes submitted Python jobs with the privileges of its
worker processes. Access to its job submission and control interfaces must
therefore be limited to trusted users and hosts. Untrusted access can result in
arbitrary code execution. See the upstream discussion of
[CVE-2023-48022](https://www.anyscale.com/blog/update-on-ray-cves-cve-2023-6019-cve-2023-6020-cve-2023-6021-cve-2023-48022-cve-2023-48023).

Grana pins Ray 2.55.0 with Train V1. The supplied launchers start a single-host
cluster with `address="local"`, `_node_ip_address="127.0.0.1"`,
`dashboard_host="127.0.0.1"`, and `include_dashboard=False`. This configures
loopback control services and disables the dashboard; it does not isolate every
worker or agent socket. Online W&B retains outbound access for log uploads.
Restrict all Ray head, worker, and agent endpoints
with host and network access controls, and verify their actual reachability in
the deployment environment. Protecting Grana's HTTP server does not also protect
separate Ray ports. Browser access must be included in this assessment:
[CVE-2025-62593](https://github.com/ray-project/ray/security/advisories/GHSA-q279-jhrf-cc6v)
describes a DNS rebinding attack against Ray job interfaces, including on private
networks. Keep untrusted browsing separate from hosts and sessions that can
reach these interfaces.

Ray 2.55.0 fixes executable deserialization of Arrow tensor extension metadata
in [CVE-2026-41486](https://github.com/ray-project/ray/security/advisories/GHSA-mw35-8rx3-xf9r).
Keep its default JSON-only metadata reader: do not enable
`RAY_DATA_AUTOLOAD_CLOUDPICKLE_TENSOR_METADATA=1`. This compatibility switch
restores executable deserialization and is rejected by Grana's environment
checker. The switch is read when Ray's tensor extension module is first
imported; changing it afterward does not update that module's cached setting.
Ray tensor files written with cloudpickle metadata by versions 2.49–2.54 are
rejected by default and require conversion from a trusted source before use
with this configuration. Grana's ordinary cache format does not require this
compatibility switch.

The supplied Grana training and evaluation paths do not call
`ray.data.read_webdataset()`. Extensions that use this reader must treat input
archives as trusted code: in affected Ray versions, the default decoder can
deserialize executable Python objects. Do not feed it untrusted WebDataset TAR
archives. See
[CVE-2026-57516](https://github.com/ray-project/ray/security/advisories/GHSA-hhrp-gw25-jr43).

### Image decoding and rendering

Only decode images and recordings from trusted, controlled data pipelines.
Pillow has documented image-processing vulnerabilities,
including memory corruption when decoding crafted
[PSD images](https://github.com/python-pillow/Pillow/security/advisories/GHSA-cfh3-3jmp-rvhc)
and memory disclosure or process crashes when decoding crafted
[TIFF images](https://github.com/python-pillow/Pillow/security/advisories/GHSA-c6hw-5grf-ff96).
Other optional image handlers and rendering tools can introduce additional
risks when processing attacker-controlled files.

The inference CLI's image-list input rejects classic TIFF and BigTIFF headers,
including Pillow's reversed-version variants, before invoking Pillow. It checks
the content of the same open file handle used for decoding, regardless of filename
extension, and retains other Pillow-supported RGB formats. The annotation evidence
reader restricts Pillow to PNG/JPEG/WebP before opening images. The Python predictor,
HTTP client, and prediction renderer reject PIL objects whose format is TIFF before
loading their pixels; they cannot protect decoding already performed by the caller
or infer the source format of a supplied array or copied PIL image.

These checks reduce TIFF decoding exposure at these input boundaries; they do not
fix Pillow, cover every image vulnerability, or restrict FFmpeg's video decoders.
RGB mode and a supported format do not establish that a file is safe. Treat
datasets, image paths, and their source files as trusted inputs across all workflows.
Converting an untrusted image to another format still parses the original file
and requires an appropriately isolated processing environment.

### Checkpoints and filesystem references

Use model weights, checkpoint indexes, configurations, and filesystem reference
documents only from trusted sources. Accelerate's affected checkpoint-loading
functions can follow malicious shard paths or block on named pipes; see
[CVE-2026-69112](https://github.com/huggingface/accelerate/issues/4067).
In affected fsspec versions, opening a malicious Kerchunk reference document
through `ReferenceFileSystem` can execute code through template evaluation; see
[CVE-2026-104851](https://github.com/fsspec/filesystem_spec/security/advisories/GHSA-27vj-qcqg-25rc).
Apply these trust requirements to extensions and other tools sharing the
environment.

## Credentials and artifacts

- Keep API keys, access tokens, and W&B credentials outside the repository.
  Supply credentials through the documented environment variables or an
  external secret manager, and avoid including them in command arguments,
  configuration files, prompts, logs, screenshots, or reports.
- Keep private datasets, robot recordings, cache payloads, checkpoints, and
  machine-specific paths out of commits and release archives. Treat local
  annotation receipts and review artifacts as private unless approved for
  sharing.
- API annotation sends selected visual evidence and task context to the
  configured provider. Use an approved provider endpoint and only submit
  material you are authorized to share. Configure W&B destinations and
  checkpoint uploads according to the same data-sharing requirements.
- Load datasets, checkpoints, model code, and configuration only from sources
  you trust. A local research environment is not a sandbox for untrusted
  artifacts. Keep dependency versions and their licenses under review when
  changing the runtime.
- If a credential is exposed, revoke and rotate it. Removing the latest file
  or commit does not remove copies from repository history or external logs.

Grana produces model predictions and annotation suggestions; it does not
provide a safety controller for robot actuation. Integrations that use these
outputs to command hardware need their own operator controls and safety checks.
