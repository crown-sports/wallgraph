# Security policy

The latest published 0.x version receives security fixes on a best-effort basis. Earlier versions may require upgrading; this experimental project has no support SLA.

Report vulnerabilities through [GitHub private vulnerability reporting](https://github.com/crown-sports/wallgraph/security/advisories/new). Include the affected version, minimal synthetic reproduction, impact, and suggested mitigation. Do not post exploit details, credentials, private drawings, datasets, or weights in public issues.

Ordinary correctness bugs belong in the bug issue form. Maintainers will review private reports, coordinate a fix when appropriate, and describe the affected versions in a security advisory. No response deadline is promised.

The library operates on local files and caller-supplied models. Applications that accept uploads or expose network endpoints must implement their own access controls, resource limits, and isolation. The repository does not provide a hosted inference service.
