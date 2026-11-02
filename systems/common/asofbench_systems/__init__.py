"""The code that the systems' adapters share: the stream's episodes and the one episode template
every store system ingests (``stream``), the Anthropic client that every adapter builds
(``llm``), and the log of each model call's usage (``usage``).

``stream`` and ``usage`` use only the standard library; ``llm`` imports ``anthropic`` and
``httpx`` only inside the function that builds the client, so the package imports where neither is
installed, as on the repository's own environment and in CI.
"""
