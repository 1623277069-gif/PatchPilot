"""Behavioral fixtures for actual Git/pytest acceptance, without model calls."""
import json, os, subprocess, sys, tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent
SCRIPT=ROOT/'agent/skills/verify-change/scripts/check.py'
PY=sys.executable
rows=[]
def fixture():
 r=Path(tempfile.mkdtemp(prefix='repair-fixture-'))
 for n,s in {'.gitignore':'__pycache__/\n.pytest_cache/\n','source.py':'VALUE = 0\n','test_source.py':'from source import VALUE\ndef test_value(): assert VALUE == 2\n','test_skip.py':'import pytest\n@pytest.mark.skip(reason="fixture")\ndef test_skip(): assert False\n','empty.py':'print("nothing")\n'}.items(): (r/n).write_text(s)
 for args in [('init','-q'),('add','.'),('-c','user.name=Fixture','-c','user.email=fixture@example.invalid','commit','-qm','initial')]: subprocess.run(['git','-C',str(r),*args],check=True)
 return r

def call(r,*args):
 p=subprocess.run([str(PY),str(SCRIPT),*args],env=dict(os.environ,GEMMA_EDIT_WORKSPACE=str(r)),text=True,capture_output=True,check=False)
 return json.loads(p.stdout)
def check(name,ok,data):
 rows.append({'case':name,'passed':bool(ok),'report':data});assert ok,(name,data)
r=fixture();(r/'source.py').write_text('VALUE = 2\n')
a=call(r,'pytest','test_source.py');check('source_repair_passes',a['acceptance']['accepted'],a)
b=call(r,'pytest','test_source.py');check('identical_evidence_reused',b.get('reused_evidence') and b['acceptance']['executions']==1,b)
(r/'source.py').write_text('VALUE = 3\n');b=call(r,'inspect');check('stale_pass_invalidated',not b['acceptance']['accepted'] and not b['acceptance']['fresh'],b)
r=fixture();(r/'source.py').write_text('VALUE = 1\n')
a=call(r,'pytest','test_source.py');check('failure_detected',not a['acceptance']['accepted'] and a['test']['status']=='failed',a)
b=call(r,'pytest','test_skip.py');check('cannot_swap_failing_test',b['status']=='error',b)
(r/'source.py').write_text('VALUE = 2\n');a=call(r,'pytest','test_source.py');check('single_source_retry_passes',a['acceptance']['accepted'] and a['acceptance']['executions']==2,a)
(r/'source.py').write_text('VALUE = 4\n');a=call(r,'pytest','test_source.py');check('third_execution_blocked',a['status']=='error' and 'limit' in a['message'],a)
r=fixture();(r/'test_source.py').write_text('def test_value(): assert True\n');a=call(r,'pytest','test_source.py');check('test_only_patch_rejected',not a['acceptance']['accepted'] and len(a['acceptance']['reasons'])>=2,a)
r=fixture();(r/'source.py').write_text('VALUE = 2\n');(r/'test_source.py').write_text('def test_value(): assert True\n');a=call(r,'pytest','test_source.py');check('modified_existing_test_rejected',not a['acceptance']['accepted'],a)
for label,selector,status in [('all_skipped','test_skip.py','all_skipped'),('no_tests','empty.py','no_tests')]:
 r=fixture();(r/'source.py').write_text('VALUE = 2\n');a=call(r,'pytest',selector);check(label,not a['acceptance']['accepted'] and a['test']['status']==status,a)
r=fixture();(r/'source.py').write_text('VALUE = 2\n');(r/'repro_check.py').write_text('print(1)\n');a=call(r,'pytest','test_source.py');check('workspace_scratch_rejected',not a['acceptance']['accepted'],a)
r=fixture();(r/'source.py').write_text('VALUE = 2\n');tmp=Path(tempfile.mkdtemp())/'test_issue.py';tmp.write_text('from source import VALUE\ndef test_issue(): assert VALUE == 2\n');a=call(r,'pytest',str(tmp));check('external_repro_supported',a['acceptance']['accepted'],a)
r=fixture();(r/'source.py').write_text('VALUE = 2\n');tmp=Path(tempfile.mkdtemp())/'test_issue.py';tmp.write_text('from source import VALUE\ndef test_issue(): assert VALUE == 2\n');a=call(r,'pytest',str(tmp));tmp.write_text('def test_issue(): assert True\n');a=call(r,'inspect');check('external_test_tamper_invalidates_freshness',not a['acceptance']['accepted'],a)
(r/'source.py').write_text('VALUE = 3\n');a=call(r,'pytest',str(tmp));check('external_test_tamper_blocks_retry',a['status']=='error',a)
r=fixture();a=call(r,'context','VALUE');check('bounded_context',len(a['context']['excerpt'])<=3500 and 'source.py' in a['context']['excerpt'],a)
print(json.dumps({'checks':len(rows),'passed':sum(r['passed'] for r in rows)}))
