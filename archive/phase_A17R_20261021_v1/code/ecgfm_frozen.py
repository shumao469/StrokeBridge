"""Inference-only ECG-FM adapter using unmodified pinned upstream modules.

No fairseq training installation or random parameter fallback. Every encoder key
must load; only pretraining-only heads can be discarded. MIT upstream retained.
"""
import ast, importlib.util, json, sys, types
from pathlib import Path
import numpy as np
import torch
from torch import nn
from omegaconf import OmegaConf

UPSTREAM=Path(__file__).resolve().parents[1]/'ecgfm_upstream'
WEIGHTS=Path('DRIVE_ROOT/脑心同查/models/ecg-fm/584219ea492cdeef2e19ffbdf9c6ecc874ba427e/mimic_iv_ecg_physionet_pretrained.pt')

def upstream_modules():
    package=types.ModuleType('fairseq_signals'); package.__path__=[str(UPSTREAM/'fairseq_signals')]
    modules=types.ModuleType('fairseq_signals.modules'); modules.__path__=[str(UPSTREAM/'fairseq_signals/modules')]
    sys.modules['fairseq_signals']=package; sys.modules['fairseq_signals.modules']=modules
    for name,attrs in [('transpose_last',['TransposeLast']),('layer_norm',['LayerNorm','Fp32LayerNorm']),('fp32_group_norm',['Fp32GroupNorm']),('same_pad',['SamePad']),('quant_noise',[]),('dropout',[]),('conv_feature_extraction',['ConvFeatureExtraction']),('conv_positional_encoding',['ConvPositionalEncoding']),('multi_head_attention',['MultiHeadAttention']),('transformer_encoder_layer',['TransformerEncoderLayer']),('transformer_encoder',['TransformerEncoder'])]:
        fullname='fairseq_signals.modules.'+name
        spec=importlib.util.spec_from_file_location(fullname,UPSTREAM/'fairseq_signals/modules'/f'{name}.py')
        mod=importlib.util.module_from_spec(spec); sys.modules[fullname]=mod; spec.loader.exec_module(mod)
        for attr in attrs: setattr(modules,attr,getattr(mod,attr))
    return modules

class FrozenECGFM(nn.Module):
    def __init__(self,checkpoint=WEIGHTS):
        super().__init__()
        # Pickle only from the pinned official asset after independent SHA verification.
        from acquire_a3 import digest,SHA
        if digest(checkpoint)!=SHA: raise ValueError('Unverified checkpoint')
        ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
        cfg=OmegaConf.create(ck['cfg']['model']);cfg.saliency=False;self.cfg=cfg
        m=upstream_modules()
        layers=[(256,2,2)]*4 if cfg.conv_feature_layers=='[(256, 2, 2)] * 4' else ast.literal_eval(cfg.conv_feature_layers)
        dim=layers[-1][0]
        self.feature_extractor=m.ConvFeatureExtraction(layers,in_d=cfg.in_d,mode=cfg.extractor_mode,conv_bias=cfg.conv_bias)
        self.layer_norm=m.LayerNorm(dim)
        self.post_extract_proj=nn.Linear(dim,cfg.encoder_embed_dim) if dim!=cfg.encoder_embed_dim else None
        self.conv_pos=m.ConvPositionalEncoding(cfg)
        self.encoder=m.TransformerEncoder(cfg)
        state=ck['model']; expected=set(self.state_dict()); converted={}
        for k,v in state.items():
            k=k.replace('weight_g','parametrizations.weight.original0').replace('weight_v','parametrizations.weight.original1')
            converted[k]=v
        missing=expected-set(converted)
        extras=set(converted)-expected
        allowed=('quantizer.','project_q.','final_proj.','mask_emb','target_glu.','input_quantizer.','project_inp.')
        illegal=[k for k in extras if not k.startswith(allowed)]
        if missing or illegal: raise ValueError(f'Encoder loading mismatch missing={missing} illegal={illegal}')
        self.load_state_dict({k:v for k,v in converted.items() if k in expected},strict=True)
        self.load_audit={'encoder_keys':len(expected),'discarded_pretraining_keys':sorted(extras),'parameters':sum(p.numel() for p in self.parameters()),'cfg':OmegaConf.to_container(cfg,resolve=False)}
        self.feature_grad_mult=0
        self.dropout_input=nn.Identity();self.dropout_features=nn.Identity();self.input_quantizer=None
        self.requires_grad_(False); self.eval()

    def forward(self,source):
        x=self.layer_norm(self.feature_extractor(source).transpose(1,2))
        if self.post_extract_proj is not None:x=self.post_extract_proj(x)
        x=x+self.conv_pos(x,channel_first=False)
        return self.encoder(x)['x'].mean(dim=1)

    def upstream_parity(self,source):
        """Run exact upstream Wav2Vec2.forward AST (mask=False/features_only).
        Training imports/dataclasses are not executed; the method body is intact.
        """
        sourcefile=UPSTREAM/'fairseq_signals/models/wav2vec2/wav2vec2.py'
        tree=ast.parse(sourcefile.read_text(encoding='utf-8'))
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Wav2Vec2Model')
        forward=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='forward')
        scope={'torch':torch};exec(compile(ast.Module(body=[forward],type_ignores=[]),str(sourcefile),'exec'),scope)
        ref=scope['forward'](self,source,padding_mask=None,mask=False,features_only=True)['x'].mean(dim=1)
        actual=self(source)
        return float((ref-actual).abs().max().item())

def preprocess(x,fs):
    """Same central 6s as baseline; official linear resample and per-lead z-score.

    Six rather than upstream 5s is a declared variable-length inference choice.
    Normalization removes positive constant amplitude scaling, not physical QC.
    """
    from ecg_transform.t.common import LinearResample
    from ecg_transform.t.scale import Standardize
    from ecg_transform.inp import ECGInput
    from ecg_transform.sample import ECGMetadata
    x=np.asarray(x,dtype=np.float64)
    good=np.flatnonzero(np.any(np.abs(x)>1e-8,axis=1))
    if not len(good):raise ValueError('all zero')
    x=x[good[0]:good[-1]+1]; start=(len(x)-int(fs*6))//2
    if start<0 or not np.isfinite(x).all():raise ValueError('bad ECG')
    x=x[start:start+int(fs*6)].T
    meta=ECGMetadata(sample_rate=int(fs),num_samples=x.shape[1],lead_names=['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6'],unit=None,input_start=0,input_end=x.shape[1])
    inp=ECGInput(x,meta)
    # Run exact library transforms; constant leads zeroed explicitly.
    from ecg_transform.t.common import HandleConstantLeads
    for t in [LinearResample(desired_sample_rate=500),HandleConstantLeads(strategy='zero'),Standardize()]:inp=t([inp])[0]
    return np.asarray(inp.signal,dtype=np.float32)

if __name__=='__main__':
    torch.set_num_threads(2)
    model=FrozenECGFM().to('cuda:0')
    with torch.inference_mode(): y=model(torch.zeros(2,12,3000,device='cuda:0'))
    print(json.dumps({'load':model.load_audit,'output_shape':list(y.shape),'finite':bool(torch.isfinite(y).all())},default=str))
