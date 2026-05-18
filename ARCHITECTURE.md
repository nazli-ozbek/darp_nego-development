## Project Architecture

Canonical code paths:

- Runtime entry:
  - `darp_nego.runners.run_negotiation`
  - `darp_nego.runners.config`
- Core domain-facing modules:
  - `darp_nego.core.darp`
  - `darp_nego.core.negotiator`
  - `darp_nego.core.outcome`
  - `darp_nego.core.utils`
  - `darp_nego.core.learning`
- Protocols:
  - `darp_nego.protocols.common`
  - `darp_nego.protocols.heuristic`
  - `darp_nego.protocols.learning`
- Logging:
  - `darp_nego.logging`
- Metrics package:
  - `metrics/*`
- Experiment/analysis scripts:
  - `experiments/*_impl.py`

Notes:

- `darp_nego/protocol/` is kept only for non-code artifacts
  (`prenegotiation_specification.md`, archived `.zip`).
