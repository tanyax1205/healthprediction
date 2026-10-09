#!/usr/bin/env bash
# End-to-end smoke test against a running MultiDx API (default http://localhost:8000).
set -euo pipefail

BASE="${BASE_URL:-http://localhost:8000}"
fail=0

check() {
  local desc="$1" url="$2"
  local code
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 15 "${BASE}${url}")
  if [[ "$code" == "200" ]]; then
    echo "  PASS  ${desc}  (HTTP ${code})"
  else
    echo "  FAIL  ${desc}  (HTTP ${code})"
    fail=1
  fi
}

echo "Smoke-testing MultiDx API at ${BASE}"
check "health        " "/health"
check "version       " "/version"
check "tasks         " "/tasks"
check "task heart    " "/tasks/heart"

echo "  -- single predictions --"
code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 30 -X POST \
  "${BASE}/predict/heart" -H 'Content-Type: application/json' \
  -d '{"values":{"age":63,"sex":1,"trestbps":145,"chol":233,"fbs":1,"restecg":0,"thalach":150,"exang":0,"oldpeak":2.3}}')
[[ "$code" == "200" ]] && echo "  PASS  predict heart   (HTTP ${code})" || { echo "  FAIL  predict heart   (HTTP ${code})"; fail=1; }

code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 30 -X POST \
  "${BASE}/predict/symptoms" -H 'Content-Type: application/json' \
  -d '{"values":{"itching":1,"skin_rash":1,"cough":1}}')
[[ "$code" == "200" ]] && echo "  PASS  predict symptoms (HTTP ${code})" || { echo "  FAIL  predict symptoms (HTTP ${code})"; fail=1; }

echo "  -- invalid input -> 422 --"
code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 15 -X POST \
  "${BASE}/predict/heart" -H 'Content-Type: application/json' -d '{"values":{}}')
[[ "$code" == "422" ]] && echo "  PASS  validation 422 (HTTP ${code})" || { echo "  FAIL  validation 422 (HTTP ${code})"; fail=1; }

echo "  -- batch CSV --"
printf 'age,sex,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal,cp\n63,1,145,233,1,0,150,0,2.3,1,0,3,3\n' > /tmp/multidx_smoke.csv
code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 30 -X POST \
  "${BASE}/predict/heart/batch" -F "file=@/tmp/multidx_smoke.csv")
[[ "$code" == "200" ]] && echo "  PASS  batch CSV     (HTTP ${code})" || { echo "  FAIL  batch CSV     (HTTP ${code})"; fail=1; }

if [[ "$fail" == "0" ]]; then echo "ALL CHECKS PASSED"; else echo "SOME CHECKS FAILED"; exit 1; fi