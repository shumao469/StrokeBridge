# Deployment / 部署

## What is supplied

`demo.html` is an offline, aggregate-only interactive evidence viewer. `streamlit_app.py` is a prepared Streamlit implementation of the same research scope. Neither accepts patient uploads, loads model weights or produces clinical predictions.

A GitHub repository URL is not a deployed application URL. A cloud demo is reported only after installation, startup, access and content checks pass. No Streamlit cloud deployment is implied by this source release.

## Local Streamlit environment

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m streamlit run streamlit_app.py --server.address 127.0.0.1 --server.port 8501
```

For Windows use `.venv\Scripts\python.exe`. Keep this environment separate from frozen experiment environments. Access `http://127.0.0.1:8501` only after the server is actually running. The local default does not open LAN/public ports, create a tunnel or disable XSRF/CORS protections.

## Streamlit Community Cloud

In the user's own authenticated Streamlit workspace, choose the authorized GitHub repository/branch and entry file `streamlit_app.py`. Use a compatible Python version and platform-approved server configuration. Do not publish credentials, raw ECG, patient-level records, hospital documents, model weights or historical launchers. The `archive/` directory is unnecessary for the minimum online application and should be excluded from that deployment bundle.

Before reporting a URL, check dependency installation, app health, all tabs, aggregate numbers against JSON, synthetic-only contract tests, undefined precision under all rejection, absence of upload/prediction interfaces, and unauthenticated access scope. Record actual URL, commit and dependency environment. Do not fabricate a `streamlit.app` address.

Official guidance: [Streamlit deployment preparation](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app), [deployment](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy). The direct UI dependency is pinned to [Streamlit 1.50.0](https://pypi.org/project/streamlit/1.50.0/); this is not a transitive dependency lock or a claim that every environment has been tested.

## Publication boundaries

The public source export contains code, source hashes, synthetic examples and aggregate research results only. Institutional/third-party rights remain in force. No clinical deployment, patient-specific result, future-risk model validation or journal acceptance is implied.

