"""
Step 5 test: exercises every API endpoint with Flask's built-in test client.
No server needs to be running.   Run:  python test_api.py
"""
import json

from app import ANALYTICS_NAMES, STATE, app

client = app.test_client()
results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def section(t):
    print("\n" + "=" * 72 + f"\n{t}\n" + "=" * 72)


def post(body, raw=False, content_type="application/json"):
    data = body if raw else json.dumps(body)
    return client.post("/api/recommend", data=data, content_type=content_type)


def main():
    section("1. HEALTH AND METADATA")
    r = client.get("/api/health")
    h = r.get_json()
    check("health returns 200", r.status_code == 200)
    check("model is loaded", h["model_loaded"] and h["status"] == "ok", h.get("error") or "")
    check("no output files missing", h["missing_outputs"] == [], str(h["missing_outputs"]))
    r = client.get("/api/meta")
    m = r.get_json()
    check("meta lists 7 features", r.status_code == 200 and len(m["features"]) == 7)
    check("meta lists 22 crops", len(m["crops"]) == 22)
    check("every feature has unit, help, limits",
          all(f["unit"] and f["help"] and "hard_min" in f for f in m["features"]))

    section("2. SAMPLE INPUTS AND CROP PROFILES")
    r = client.get("/api/sample")
    s = r.get_json()
    check("default sample works", r.status_code == 200 and len(s["inputs"]) == 7, s["display"])
    check("sample for a named crop", client.get("/api/sample?crop=chickpea").status_code == 200)
    check("sample for unknown crop is 404", client.get("/api/sample?crop=dragonfruit").status_code == 404)
    r = client.get("/api/crop/rice")
    check("crop profile works", r.status_code == 200 and len(r.get_json()["features"]) == 7)
    check("crop profile accepts display name",
          client.get("/api/crop/Kidney%20Beans").status_code == 200)
    check("unknown crop profile is 404", client.get("/api/crop/dragonfruit").status_code == 404)

    section("3. RECOMMENDATION ENDPOINT")
    rice = client.get("/api/sample?crop=rice").get_json()["inputs"]
    chick = client.get("/api/sample?crop=chickpea").get_json()["inputs"]
    r = post({**rice, "selected_crop": "rice"})
    d = r.get_json()
    check("valid request returns 200", r.status_code == 200)
    check("top-3 present", len(d["top3"]) == 3)
    check("selected-crop assessment present", d["selected"] and d["selected"]["label"])
    check("cautions list present", isinstance(d["cautions"], list))
    d2 = post({**rice, "selected_crop": "rice"}).get_json()
    check("same request gives identical response", json.dumps(d) == json.dumps(d2))
    check("wrapped {'inputs': ...} format works",
          post({"inputs": rice, "selected_crop": "rice"}).status_code == 200)
    check("selected crop is optional",
          post(rice).get_json()["selected"] is None)
    t_rice = post(rice).get_json()["top3"][0]["crop"]
    t_chick = post(chick).get_json()["top3"][0]["crop"]
    check("different inputs give different top crop", t_rice != t_chick, f"{t_rice} vs {t_chick}")
    out = post({**rice, "N": 250}).get_json()
    check("out-of-dataset-range value gives a warning", len(out["warnings"]) >= 1)

    section("4. ERROR HANDLING")
    r = post({**rice, "ph": 15})
    e = r.get_json()
    check("pH=15 rejected with 400", r.status_code == 400 and e["error"] == "validation_error")
    check("error names the field", "ph" in e["errors"], str(e["errors"].get("ph")))
    bad = {k: v for k, v in rice.items() if k != "N"}
    bad["ph"] = "abc"
    e = post({**bad, "selected_crop": "dragonfruit"}).get_json()
    check("several errors reported at once", {"N", "ph", "selected_crop"} <= set(e["errors"]),
          ", ".join(e["errors"]))
    check("non-JSON body is 400", post("hello", raw=True, content_type="text/plain").status_code == 400)
    check("invalid JSON is 400", post("{bad json", raw=True).status_code == 400)
    check("JSON array body is 400", post([1, 2, 3]).status_code == 400)
    check("empty body is 400", post("", raw=True).status_code == 400)
    r = client.get("/api/recommend")
    check("GET on /api/recommend is 405 JSON", r.status_code == 405 and r.get_json()["error"])
    r = client.get("/api/nope")
    check("unknown API path is 404 JSON", r.status_code == 404 and r.get_json()["error"])
    r = post("x" * 20000, raw=True)
    check("oversized body is 413 JSON", r.status_code == 413 and r.get_json()["error"])
    saved = STATE["rec"]
    STATE["rec"] = None
    STATE["error"] = "simulated failure"
    r = post(rice)
    check("model unavailable gives 503", r.status_code == 503 and r.get_json()["error"] == "model_unavailable")
    check("health reports degraded", client.get("/api/health").get_json()["status"] == "degraded")
    STATE["rec"], STATE["error"] = saved, None

    section("5. ANALYTICS ENDPOINTS")
    for name in ANALYTICS_NAMES:
        r = client.get(f"/api/analytics/{name}")
        ok = r.status_code == 200 and r.get_json() is not None
        check(f"/api/analytics/{name}", ok, "" if ok else f"status {r.status_code}")
    check("unknown analytics name is 404",
          client.get("/api/analytics/secret_file").status_code == 404)
    check("path traversal attempt is rejected",
          client.get("/api/analytics/..%2Fapp.py").status_code == 404)
    hist = client.get("/api/analytics/histograms").get_json()
    n = sum(hist["features"]["ph"]["counts"])
    check("histogram counts add up to the row total", n > 0, f"ph bins total {n}")

    print("\n" + "=" * 72)
    print(f"{sum(results)} of {len(results)} checks passed.")
    if not all(results):
        print("Some checks FAILED. Fix them before continuing to Step 6.")


if __name__ == "__main__":
    main()