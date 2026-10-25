"""Standard-library checks for preserving source locks and fit-budget history."""
import ast,hashlib,json,shutil,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
SOURCE=Path(__file__).with_name('a17r_resume_after_environment.py')
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write(path,value):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value),encoding='utf-8')
class RecoveryContract(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory(prefix='a17r_recovery_',dir=ROOT/'unified_project/runs/phase_A17R_20261021_v1')
  self.root=Path(self.temp.name);self.base=self.root/'base';self.run=self.root/'native';self.here=self.root/'work'
  for p in [self.base,self.run/'code',self.here]:p.mkdir(parents=True,exist_ok=True)
  self.source=self.here/'a17r_model_analysis.py';self.source.write_text('# fixed before fitting\n',encoding='utf-8');self.target=self.run/'code/a17r_model_analysis.py';self.lock=self.run/'analysis_source_lock.json'
  write(self.base/'runs/phase_A17R_20261021_v1/D_source_manifest.json',{'files':[{'file':self.source.name,'sha256':digest(self.source)}]})
  tree=ast.parse(SOURCE.read_text(encoding='utf-8'));fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='pin_analysis_without_overwrite')
  scope=dict(HERE=self.here,RUN=self.run,BASE=self.base,sha=digest,read=read,write=write,shutil=shutil)
  exec(compile(ast.Module(body=[fn],type_ignores=[]),str(SOURCE),'exec'),scope);self.pin=scope[fn.name]
 def tearDown(self):self.temp.cleanup()
 def test_first_lock_before_any_fit(self):
  self.pin([]);self.assertEqual(self.target.read_bytes(),self.source.read_bytes());self.assertEqual(read(self.lock),{str(self.target):digest(self.source)})
 def test_conflicting_lock_preserves_existing_source(self):
  self.target.write_text('# prior version\n',encoding='utf-8');old=self.target.read_bytes();write(self.lock,{str(self.target):digest(self.target)})
  with self.assertRaises(AssertionError):self.pin([])
  self.assertEqual(self.target.read_bytes(),old)
 def test_matching_lock_permits_readonly_check_after_fit(self):
  self.pin([]);before=self.lock.read_bytes();self.pin([{'number':1,'status':'started'}]);self.assertEqual(self.lock.read_bytes(),before)
 def test_first_lock_forbidden_after_fit(self):
  with self.assertRaises(AssertionError):self.pin([{'number':1,'status':'complete'}])
  self.assertFalse(self.target.exists());self.assertFalse(self.lock.exists())
 def test_working_source_drift_stops(self):
  self.source.write_text('# drift\n',encoding='utf-8')
  with self.assertRaises(AssertionError):self.pin([])
  self.assertFalse(self.target.exists());self.assertFalse(self.lock.exists())
if __name__=='__main__':unittest.main(verbosity=2)
