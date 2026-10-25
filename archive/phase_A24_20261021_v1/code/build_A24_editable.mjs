import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath,pathToFileURL} from 'node:url';
import {spawnSync} from 'node:child_process';
import {createHash} from 'node:crypto';

const CODE=path.dirname(fileURLToPath(import.meta.url));
const policyFile=path.join(CODE,'../FIGURE_POLICY.json');
if(!process.argv.includes('--mock')){
  const policy=await fs.readFile(policyFile,'utf8').then(JSON.parse).catch(error=>{if(error.code==='ENOENT')return null;throw error;});
  if(policy&&!policy.allow_new_PPTX){console.log('PPTX_DISABLED_BY_USER_PROMPT_ONLY_POLICY');process.exit(0);}
}
const U=path.resolve(CODE,'../../..');
const OUT=path.join(U,'manuscript_NC_A24_20261023');
const skill='DRIVE_ROOT/Users/USER_PROFILE/.codex/plugins/cache/openai-primary-runtime/presentations/26.1020.11348/skills/presentations';
const runtime='DRIVE_ROOT/Users/USER_PROFILE/.cache/codex-runtimes/codex-primary-runtime/dependencies';
process.env.RUNTIME_NODE_MODULES=path.join(runtime,'node/node_modules');
const runtimePython=path.join(runtime,'python/python.exe');
const {Presentation,PresentationFile,FileBlob}=await import(pathToFileURL(path.join(runtime,'node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs')).href);
const {finalizePresentation}=await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const mock=process.argv.includes('--mock');
const mockLabel=process.argv.includes('--mock-label')?process.argv[process.argv.indexOf('--mock-label')+1]:'v2';
if(!/^[a-zA-Z0-9_]+$/.test(mockLabel))throw Error('Invalid mock path suffix');
const root=mock?path.join(CODE,'../software_contracts_PPTX_wsl_'+mockLabel):OUT;
const build=path.join(root,'.pptx-build');const outputs=path.join(root,'pptx');
await fs.mkdir(build,{recursive:true});await fs.mkdir(outputs,{recursive:true});
const read=async p=>JSON.parse(await fs.readFile(p,'utf8'));
const digest=async p=>createHash('sha256').update(await fs.readFile(p)).digest('hex');
const toWindows=p=>p.replace(/^\/mnt\/([a-z])\//,(_,x)=>x.toUpperCase()+':/');
let inventory;
if(mock){
  const regions=Array.from({length:6},(_,j)=>({letter:String.fromCharCode(97+j),bbox:[25+(j%2)*255,50+Math.floor(j/2)*200,240+(j%2)*255,220+Math.floor(j/2)*200]}));
  const paths=[],texts=[];
  for(let j=0;j<6;j++){
    const b=regions[j].bbox,g=regions[j].letter;
    paths.push({bbox:[b[0]+10,b[1]+30,b[2]-10,b[3]-30],group:g,stroke:[.2,.5,.7],fill:null,line_width:1,
      commands:[{moveTo:{x:b[0]+10,y:b[3]-30}},{lineTo:{x:b[2]-10,y:b[1]+30}}]});
    for(let k=0;k<3;k++)texts.push({text:`${g} MOCK ${k}`,bbox:[b[0],b[1]+k*25,b[0]+160,b[1]+k*25+12],size:9,font:'LiberationSans',color:0x243b53,direction:[1,0],group:g});
  }
  let objectFile;
  if(process.argv.includes('--mock-objects')){
    objectFile=process.argv[process.argv.indexOf('--mock-objects')+1];
  }else{
    objectFile=path.join(build,'mock_objects.json');await fs.writeFile(objectFile,JSON.stringify({width:510,height:675,paths,texts,panel_regions:regions}));
  }
  inventory=[{figure:'MOCK_NOT_SCIENTIFIC',title:'Mock native vector/text contract',panels:6,objects:objectFile}];
}else{
  const done=await read(path.join(OUT,'figure_completion.json'));
  if(done.new_fits!==0||!done.SVG_native_text||!done.PDF_no_raster_layers)throw Error('Vector artifact gate failed');
  const audit=await read(path.join(CODE,'../recovery_E_wsl_20261023_v2/audit/completion.json'));
  if(audit.state!=='E_independent_zero_fit_audit_PASS'||audit.total_fits!==422)throw Error('E scientific audit gate failed');
  inventory=await read(path.join(OUT,'figure_inventory.json'));
  if(inventory.length!==7||inventory.some(x=>x.rows!==3||x.panels<6))throw Error('Figure inventory mismatch');
}
const W=1080,H=1428;
const sourcePaths=[fileURLToPath(import.meta.url),path.join(CODE,'group_A24_panels.py'),
  ...inventory.map(item=>path.isAbsolute(item.objects)?item.objects:path.join(OUT,item.objects))];
if(!mock)sourcePaths.push(path.join(OUT,'figure_inventory.json'),path.join(OUT,'figure_completion.json'));
const sourceLock=Object.fromEntries(await Promise.all(sourcePaths.map(async p=>[p,await digest(p)])));
await fs.writeFile(path.join(build,'pptx_source_lock.json'),JSON.stringify({time:new Date().toISOString(),files:sourceLock,new_fits:0},null,2));
const deck=Presentation.create({slideSize:{width:W,height:H}});
const hex=(color,opacity=1)=>{
  if(color===null||color===undefined)return 'none';
  const rgb=Array.isArray(color)?color:[((color>>16)&255)/255,((color>>8)&255)/255,(color&255)/255];
  return '#'+rgb.map(v=>Math.round(255*(v*opacity+1-opacity)).toString(16).padStart(2,'0')).join('');
};
const manifests=[];
for(const item of inventory){
  const source=path.isAbsolute(item.objects)?item.objects:path.join(OUT,item.objects);
  const o=await read(source);const sx=W/o.width,sy=H/o.height;
  const slide=deck.slides.add();slide.background.fill='#FFFFFF';
  let counter=0;
  for(const p of o.paths){
    const [x0,y0,x1,y1]=p.bbox,ww=Math.max(x1-x0,.05),hh=Math.max(y1-y0,.05);
    const commands=p.commands.map(cmd=>Object.fromEntries(Object.entries(cmd).map(([kind,values])=>[kind,
      Object.fromEntries(Object.entries(values).map(([key,value])=>[key,(value-(key.startsWith('x')?x0:y0))*(key.startsWith('x')?sx:sy)]))])));
    const dash=p.dashes&&!p.dashes.startsWith('[]')?'dashed':'solid';
    slide.shapes.add({geometry:'custom',name:`${p.group==='global'?'global':'panel_'+p.group}__path_${counter++}`,
      position:{left:x0*sx,top:y0*sy,width:ww*sx,height:hh*sy},
      fill:hex(p.fill,p.fill_opacity??1),line:{fill:hex(p.stroke,p.stroke_opacity??1),width:(p.stroke?p.line_width:0)*sx,style:dash},
      customPaths:[{width:ww*sx,height:hh*sy,commands}]});
  }
  for(const t of o.texts){
    const [x0,y0,x1,y1]=t.bbox;const angle=Math.atan2(t.direction[1],t.direction[0])*180/Math.PI;
    const vertical=Math.abs(angle)>45&&Math.abs(angle)<135;
    const width=(vertical?y1-y0:x1-x0)*sx+2;
    const height=Math.max(t.size*1.18,(vertical?x1-x0:y1-y0))*sy;
    const centerX=(x0+x1)/2*sx,centerY=(y0+y1)/2*sy;
    const shape=slide.shapes.add({geometry:'textbox',name:`${t.group==='global'?'global':'panel_'+t.group}__text_${counter++}`,
      position:{left:centerX-width/2,top:centerY-height/2,width,height,rotation:angle},
      fill:'none',line:{fill:'none',width:0,style:'solid'}});
    shape.text=[{runs:[{run:t.text,style:{typeface:'Arial',fontSize:`${t.size*sx}px`,color:hex(t.color),
      bold:/Bold/i.test(t.font),italic:/Italic|Oblique/i.test(t.font)}}]}];
    shape.text.style={fontSize:t.size*sx,color:hex(t.color),bold:/Bold/i.test(t.font),
      italic:/Italic|Oblique/i.test(t.font),alignment:'left',verticalAlignment:'middle',wrap:'none',autoFit:'none',insets:0};
  }
  slide.speakerNotes.text=`${item.figure}: ${item.title}. Source: ${toWindows(source)}. All texts and plotted paths are native editable objects; panels are grouped after export. Concurrent AF, viewed source-exposed development. Figure legends and source tables accompany this file. Evidence-design references: Brody https://www.nature.com/articles/s41467-026-76155-6 ; McCracken https://www.nature.com/articles/s41467-022-35321-2 ; Meyre https://www.nature.com/articles/s41467-025-62218-7 . No third-party artwork reused.`;
  manifests.push({figure:item.figure,objects:counter,source_SHA256:await digest(source)});
}
const candidate=path.join(build,'native_ungrouped.pptx');
const grouped=path.join(build,'native_panel_groups.pptx');
const final=path.join(outputs,mock?'MOCK_contract.pptx':'A24_NC_figures_editable.pptx');
await (await PresentationFile.exportPptx(deck)).save(candidate);
const groupedRun=spawnSync(runtimePython,[path.join(CODE,'group_A24_panels.py'),'--input',candidate,'--output',grouped,'--receipt',path.join(build,'grouping_receipt.json')],{encoding:'utf8',windowsHide:true});
if(groupedRun.status!==0)throw Error(groupedRun.stdout+'\n'+groupedRun.stderr);
const result=await finalizePresentation({workspaceDir:root,candidatePath:grouped,finalPath:final,
  pythonExecutable:runtimePython,integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
  layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
  explicitTotalSlideCount:inventory.length,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],
  layoutArgs:['--expected-slide-size-emu',`${W*9525},${H*9525}`,'--validate-bullet-geometry','--validate-heading-fit'],
  fontPolicy:{basis:'design',families:['Arial']},verifyArtifactToolImport:true,
  receiptPath:path.join(build,'finalizer_receipt.json')});
const rendered=await PresentationFile.importPptx(await FileBlob.load(final));
for(const [i,slide] of rendered.slides.items.entries()){
  const png=await rendered.export({slide,format:'png',scale:1});
  await fs.writeFile(path.join(build,`slide_${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await png.arrayBuffer()));
}
const receipt={time:new Date().toISOString(),state:mock?'mock_native_PPTX_render_PASS':'native_PPTX_rendered_visual_review_pending',
  slides:inventory.length,native_text_and_data_paths:true,movable_panel_groups:true,
  new_fits:0,source_manifests:manifests,visual_review_complete:false,output:final,SHA256:await digest(final)};
for(const [p,h] of Object.entries(sourceLock))if(await digest(p)!==h)throw Error('PPTX source changed during authoring: '+p);
await fs.writeFile(path.join(root,mock?'completion.json':'PPTX_completion.json'),JSON.stringify(receipt,null,2));
console.log(JSON.stringify(receipt));
