---
name: verify-change
description: Batch bounded source search, run focused pytest acceptance and check final patch freshness with compact evidence.
---

Use run_skill_script with skill_name="verify-change", file_path="scripts/check.py" and string-list args.

- ["context", "literal_symbol", "another_symbol"]: one bounded tracked-source search; returns compact hits and a full log path outside the repository. Read relevant implementation rather than searching repeatedly.
- ["pytest", "tests/test_example.py::test_behavior"]: focused acceptance, default 30-second timeout. Checks actual collected/passed/failed/skipped counts, production/config changes, modified existing tests/test configuration, and workspace stability. Full report/logs stay outside the repo. Read acceptance.accepted and reasons; outer status="ok" only means execution succeeded.
- ["inspect"]: compact current diff plus whether previous passing acceptance is still fresh. Useful once before submission, not as an initialization ritual.

At most two pytest executions per task workspace. Identical unchanged evidence is reused. After failed tests, change implementation once and retry the same selectors. The script refuses a third execution and changing selectors after a failure. Correct assertion coverage is still your responsibility. No tests, all skipped, runner errors, timeouts, source-less patches and protected-test modifications do not pass acceptance. Temporary repro files inside the workspace block acceptance; put them in /tmp/test_issue.py and pass that absolute selector instead.

This is deterministic checking when invoked, not a global interceptor for native tools or submit_patch. It never edits source or silently removes files. Submit best supported source patch promptly; disclose failures instead of claiming unverified success.
