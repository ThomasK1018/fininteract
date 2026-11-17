"""Publish only the prepared website directory to a separate GitHub Pages repo.

Dry run is the default. --publish creates/updates ThomasK1018/fininteract-project
and enables Pages. --status reads deployment status. GitHub authentication is
obtained from the existing Git credential helper, kept in memory, and never
written to disk or printed. The research repository is not pushed by this tool.
"""
from pathlib import Path
import argparse
import base64
import hashlib
import json
import os
import re
import subprocess
import requests

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "website"
STATE = ROOT / "paper" / "build_acl" / "github_publish_state.json"
OWNER, NAME = "ThomasK1018", "fininteract-project"
REPO = f"{OWNER}/{NAME}"
URL = "https://thomask1018.github.io/fininteract-project/"
FILES = ["index.html", "styles.css", "app.js", ".nojekyll", "LICENSE",
         "DATA_LICENSE.txt", "README.md", "assets/favicon.svg",
         "assets/fininteract.pdf", "assets/data/fininteract_v1.jsonl",
         "assets/data/fininteract_v1.1.jsonl", "assets/data/examples.json",
         "assets/data/dataset-manifest.json", "assets/data/verified-results.json"]

def credential():
    env = dict(os.environ, GCM_INTERACTIVE="never", GIT_TERMINAL_PROMPT="0", GIT_ASKPASS="")
    result = subprocess.run(["git", "credential", "fill"],
                            input="protocol=https\nhost=github.com\n\n",
                            text=True, capture_output=True, env=env, timeout=30)
    values = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    token = values.get("password")
    if result.returncode or not token:
        raise SystemExit("GitHub authentication unavailable. Sign in using Git's credential helper.")
    return token

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    if not args.status:
        for name in FILES:
            path = SITE / name
            if not path.is_file():
                raise SystemExit(f"Missing release file: {name}")
            if path.suffix != ".pdf":
                content = path.read_text(encoding="utf-8")
                if re.search(r"sk-(?:or-|ant-)?[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{20,}|[/\\]Users[/\\]", content):
                    raise SystemExit(f"Potential private content in {name}; inspect before publication.")
        print(json.dumps({"repository": REPO, "site_url": URL, "files": FILES,
                          "total_bytes": sum((SITE / f).stat().st_size for f in FILES)}, indent=2))
    if not (args.publish or args.status):
        return
    session = requests.Session()
    session.headers.update({"Authorization": "Bearer " + credential(),
                            "Accept": "application/vnd.github+json",
                            "X-GitHub-Api-Version": "2022-11-28"})
    def api(method, endpoint, data=None, allowed=(200, 201, 202, 204)):
        response = session.request(method, "https://api.github.com" + endpoint,
                                   json=data, timeout=45)
        if response.status_code not in allowed:
            try:
                reason = response.json().get("message", "request failed")
            except ValueError:
                reason = "request failed"
            raise SystemExit(f"GitHub {method} {endpoint}: {response.status_code}: {reason}")
        return response.json() if response.content else {}
    user = api("GET", "/user")
    if user["login"].lower() != OWNER.lower():
        raise SystemExit("Authenticated account does not match the intended repository owner.")
    if args.status:
        result = api("GET", f"/repos/{REPO}/pages")
        print(json.dumps({k: result.get(k) for k in ("status", "html_url", "source", "https_enforced")}, indent=2))
        return
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    repo = api("GET", f"/repos/{REPO}", allowed=(200, 404))
    if repo.get("message") == "Not Found":
        repo = api("POST", "/user/repos", {"name": NAME, "private": False,
                   "description": "FinInteract: Findings of ACL, EACL 2027 project page and versioned benchmark downloads",
                   "homepage": URL, "auto_init": True})
        state = {"repository_id": repo["id"], "repository": REPO, "site_url": URL}
        STATE.parent.mkdir(exist_ok=True)
        STATE.write_text(json.dumps(state, indent=2), encoding="utf-8")
    elif state.get("repository_id") != repo.get("id"):
        raise SystemExit("Repository already exists without a matching local publication state; refusing to overwrite it.")
    branch = repo["default_branch"]
    head = api("GET", f"/repos/{REPO}/git/ref/heads/{branch}")["object"]["sha"]
    parent = api("GET", f"/repos/{REPO}/git/commits/{head}")
    entries = []
    for name in FILES:
        raw = (SITE / name).read_bytes()
        blob = api("POST", f"/repos/{REPO}/git/blobs",
                   {"encoding": "base64", "content": base64.b64encode(raw).decode("ascii")})
        entries.append({"path": name, "mode": "100644", "type": "blob", "sha": blob["sha"]})
    tree = api("POST", f"/repos/{REPO}/git/trees", {"base_tree": parent["tree"]["sha"], "tree": entries})
    if tree["sha"] != parent["tree"]["sha"]:
        commit = api("POST", f"/repos/{REPO}/git/commits",
                     {"message": "Publish FinInteract research project page and versioned benchmark assets",
                      "tree": tree["sha"], "parents": [head]})
        current = api("GET", f"/repos/{REPO}/git/ref/heads/{branch}")["object"]["sha"]
        if current != head:
            raise SystemExit("Remote branch changed during upload; no branch update performed.")
        api("PATCH", f"/repos/{REPO}/git/refs/heads/{branch}", {"sha": commit["sha"], "force": False})
        head = commit["sha"]
    state["commit_sha"] = head
    state["file_sha256"] = {name: hashlib.sha256((SITE/name).read_bytes()).hexdigest() for name in FILES}
    STATE.write_text(json.dumps(state, indent=2), encoding="utf-8")
    pages = api("GET", f"/repos/{REPO}/pages", allowed=(200, 404))
    if pages.get("message") == "Not Found":
        pages = api("POST", f"/repos/{REPO}/pages", {"source": {"branch": branch, "path": "/"}, "build_type": "legacy"})
    elif pages.get("source") != {"branch": branch, "path": "/"}:
        raise SystemExit("Existing Pages configuration differs; inspect it before changing hosting.")
    print(json.dumps({"repository": repo["html_url"], "commit_sha": head,
                      "site_url": pages.get("html_url", URL), "status": pages.get("status", "pending")}, indent=2))

if __name__ == "__main__":
    main()
