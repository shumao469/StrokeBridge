"""Platform-only A24 path/runtime adapters for the already-running Ubuntu WSL."""
from pathlib import Path
import ast
import re
import sys

def linux_path(value):
    text=str(value)
    match=re.match(r'^([A-Za-z]):[\\/](.*)$',text)
    if match:
        drive=match.group(1).lower()
        tail=match.group(2).replace('\\','/')
        return Path('/mnt')/drive/tail
    return Path(value)

def install_shared_path_adapter(shared):
    original=shared.sha
    def sha_linux(path):
        return original(linux_path(path))
    shared.sha=sha_linux
    return sha_linux

def install_wsl_vendor(vendor_root,safe_root):
    """Expose only pure-Python vendored helpers; use WSL-native tokenizers/safetensors/regex binaries."""
    root=Path(vendor_root)
    root_text=str(root)
    while root_text in sys.path:sys.path.remove(root_text)
    pure=['omegaconf','antlr4','biosppy','peakutils','bidict','shortuuid','einops','huggingface_hub','transformers']
    # The shared vendor root also contains Windows .pyd extensions. Create a filtered path inside this run.
    safe_root=Path(safe_root)
    safe_root.mkdir(parents=True,exist_ok=True)
    for name in pure:
        source=root/name
        link=safe_root/name
        if source.is_dir() and not link.exists():link.symlink_to(source,target_is_directory=True)
    for name in ('huggingface_hub-0.36.0.dist-info','transformers-4.57.1.dist-info'):
        source=root/name;link=safe_root/name
        if source.is_dir() and not link.exists():link.symlink_to(source,target_is_directory=True)
    safe_text=str(safe_root)
    while safe_text in sys.path:sys.path.remove(safe_text)
    sys.path.insert(0,safe_text)
    return {'safe_vendor_root':safe_text,'pure_python_packages':pure,
            'windows_binary_packages_excluded':['tokenizers','safetensors','regex']}

def wsl_runtime_metadata(source_json,destination_json):
    from a24_shared import read,put
    value=read(source_json)
    for key in ('E2_weight','P_checkpoint','E1_upstream','vendor'):
        if key in value:value[key]=str(linux_path(value[key]))
    put(destination_json,value)
    return value

def wsl_weight_metadata(source_json,destination_json):
    from a24_shared import read,put
    value=read(source_json)
    if 'weight' in value:value['weight']=str(linux_path(value['weight']))
    put(destination_json,value)
    return value

def wsl_preprocessing_admission(shared,source_module,run_root,start_time):
    """Reuse the frozen prior admission function, changing only its Windows path conversion."""
    text=Path(source_module).read_text(encoding='utf-8')
    tree=ast.parse(text)
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='preprocessing_admission')
    body=ast.get_source_segment(text,node)
    before="meta.replace('DRIVE_ROOT/','DRIVE_ROOT/')"
    assert body.count(before)==1
    body=body.replace(before,'meta')
    namespace=dict(vars(shared));namespace.update({'REC':Path(run_root),'START':start_time})
    exec(compile(body,str(source_module)+'[WSL path-only adapter]','exec'),namespace)
    return namespace['preprocessing_admission']

def subprocess_kwargs():
    """No Windows-only subprocess flags are added in the WSL runner."""
    return {}
