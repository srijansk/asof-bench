"""The code that the systems' adapters share: the stream's episodes and the one episode template
every store system ingests (``stream``), the Anthropic client that every adapter builds
(``llm``), the log of each model call's usage (``usage``), and the plumbing of a run, which is
the output directory's lock, run.json's records, the network-check preflight and the exit
statuses (``run``).

``stream``, ``usage`` and ``run`` use only the standard library; ``llm`` imports ``anthropic`` and
``httpx`` only inside the function that builds the client, so the package imports where neither is
installed, as on the repository's own environment and in CI.
"""
