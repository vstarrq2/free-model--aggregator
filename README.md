# free-model-aggregator

One server that serves the OpenAI Chat Completions API and the Anthropic Messages API over ten simulated free-tier model providers, routing each request to the best free model that can serve it, with failover when a provider refuses.

## Run

```sh
python3 main.py --list     # print the catalog of free models found
python3 main.py            # serve on a chosen port, register with the env
```

`main.py` is Python 3.13, standard library only. It listens on `0.0.0.0`, registers its port and catalog with `$MAKEFUN_ENV_URL`, answers `POST /v1/chat/completions`, `POST /v1/messages` and `GET /v1/models` (JSON and server-sent events), and exits 0 when the env says the workload is done.

Routing picks the strongest free model meeting each request's requirements (tools, vision, JSON mode, minimum context) from the vendored quality table, passes the request through in the provider's wire untouched, and fails over to the next eligible model on a 402/429 refusal. Deterministic: no wall clock, no unseeded randomness.

Built for the make.fun `free-model-aggregator-01` bounty.
