# de-spii

<p align="center">
  <img src="docs/assets/banner.png" alt="de-spii logo" width="600"/>
</p>

**Privacy middleware for DSPy and LLM pipelines.**

De-spii is a Python package for detecting and redacting Personally Identifiable Information (PII) from user prompts before they are sent to cloud-based LLMs. It ensures privacy by stripping sensitive data locally, forwarding redacted queries upstream, and then reconstructing the original content from placeholders after the response.

---

## Roadmap

* [ ] Initial release with regex + spaCy + LLM hybrid redaction
* [ ] DSPy + LangChain pipeline integration demos
* [ ] Add pluggable detector registry
* [ ] PyPI release
* [ ] Docker image with preloaded spaCy models
