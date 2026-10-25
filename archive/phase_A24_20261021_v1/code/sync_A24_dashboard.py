"""Refresh project entry points from current A24 evidence; no computation or scheduling."""
from pathlib import Path
from datetime import datetime
import json,re,html
P=Path(__file__).resolve().parents[1];U=P.parents[1];REC=P/'recovery_E_wsl_20261023_v2'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def put(p,v):
    p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(p)
def main():
    now=datetime.now().astimezone().isoformat(timespec='seconds');s=read(U/'PHASE_A24_STATUS.json')
    extraction=read(REC/'extraction/status.json') if (REC/'extraction/status.json').exists() else {}
    pipeline=read(REC/'background_pipeline.json') if (REC/'background_pipeline.json').exists() else {}
    ledger=read(REC/'comparisons/fit_ledger.json') if (REC/'comparisons/fit_ledger.json').exists() else []
    completed=sum(r['status']=='complete' for r in ledger);processed=int(extraction.get('processed',0))
    active=bool(s.get('background_worker'))
    review=U/'A24_verified_review_20261023/verification.json'
    passed=review.exists() and read(review).get('state')=='PASS'
    policy=read(P/'FIGURE_POLICY.json') if (P/'FIGURE_POLICY.json').exists() else {}
    prompt_only=policy.get('mode')=='prompt_only_after_verification'
    if passed:
        s.update(time=now,state='E_verified_complete_prompt_only' if prompt_only else 'E_verified_complete',
            background_worker=False,E_fits=12,total_fits=422,new_review_fits=0,
            final_verification='A24_verified_review_20261023/verification.json',
            verification_time=read(review)['time'],new_drawing_allowed=not prompt_only)
        put(U/'PHASE_A24_STATUS.json',s);active=False
    text=f"{now} snapshot: {s['state']}; extraction {processed}/21002; E fits {completed} complete/{len(ledger)} attempts, total {410+completed}/422. "
    text+='WSL native dependencies, strict encoder gates and actual first32 preprocessing PASS. Corrective continuation retains original extraction deadline approximately19:16:36 Asia/Shanghai; combined fit/audit cap45min. R/N reused, SPH excluded, hospital/expert branches waiting. '
    text+=('Fresh zero-fit source/model-probability/metric/patient/checkpoint review PASS; ready to organize a technical manuscript. ' if passed else '')
    text+=('User policy: written prompts only; no new figures/PPTX. Existing13:11artifacts retained, final visual review pending.' if prompt_only else 'Figure/PPTX chain triggers only after E/source/metric/checkpoint audit PASS. Rendered files still require visual review.')
    project=read(U/'PROJECT_STATE.json');project['as_of']=now
    a=project['A24'];a.update(state=s['state'],fits_completed=410+completed,fits_attempted=410+len(ledger),
        E_fits_completed=completed,background_worker=active,E_current_run=str(REC.relative_to(U)),
        E_current_summary=text,E_preprocessing32='PASS exact arrays for both pipelines',
        E_corrective_authorization='允许纠正续跑，沿用原截止',
        figure_generation='Existing13:11artifacts retained; new rendering disabled; prompt-only review.' if prompt_only else 'Numerical/source gate required.',
        figure_program='runs/phase_A24_20261021_v1/code/build_A24_figures.py',
        PPTX_program='runs/phase_A24_20261021_v1/code/build_A24_editable.mjs',
        ready_to_organize_technical_manuscript=passed,new_drawing_allowed=not prompt_only,
        budget_adjustment_needed=False,NC_submission_ready=False)
    put(U/'PROJECT_STATE.json',project)
    auto=read(U/'AUTORUN_STATE.json');auto.update(current_worker=text,
        coordinator_pid=pipeline.get('host_pid') if active else None,
        current_plan='runs/phase_A24_20261021_v1/recovery_E_wsl_20261023_v2/protocol.txt')
    put(U/'AUTORUN_STATE.json',auto)
    overview=U/'研究总览.html';content=overview.read_text(encoding='utf-8')
    banner=f'<section id="a24"><h2>当前A24：WSL纠正续跑与审计门控绘图</h2><p class="scope">快照 {html.escape(now)}：全量提取 {processed:,}/21,002；E拟合完成{completed}/12，累计{410+completed}/422。WSL原生依赖、严格模型数值门及首32条实际预处理均通过。当前状态：{html.escape(s["state"])}。纠正续跑沿用原提取截止约今日19:16:36，不顺延；拟合和审计合计最多45分钟。R400/N10复用；SPH不放行、94条保留；医院和专家资料继续等待。只有E全部审计通过，才自动生成5张主图、2张补图（合计45 panels），以及SVG/PDF/600dpi PNG和原生文字、曲线可编辑且panel成组的PPTX；最终实际图件仍须视觉检查。NC尚未投稿就绪，当前为AF识别和候选测量工作。</p><p><a href="runs/phase_A24_20261021_v1/recovery_E_wsl_20261023_v2/extraction/status.json">实时提取状态</a> · <a href="PHASE_A24_STATUS.json">科学任务状态</a> · <a href="runs/phase_A24_20261021_v1/recovery_E_wsl_20261023_v2/background_pipeline.json">后台流水线状态</a> · <a href="runs/phase_A24_20261021_v1/A24_WSL修复与预检_20261023.txt">修复与预检记录</a></p><p>以下论文v2和旧图为历史交付；尚不是本次A24新图。</p></section>'
    if passed:
        banner=f'<section id="a24"><h2>当前A24：E核验通过，可整理技术文章；仅输出文字设计</h2><p class="scope">更新 {html.escape(now)}。E于13:10完成全套审计，19:56补充零拟合核验通过：21,002条/18,260患者；评价2,095条/1,822患者；六模型概率重放最大误差2.55e-15；12次E拟合完成，A24累计422/422，R/N未重跑，无需追加预算。当前没有A24计算作业。形态M在两编码器之外的条件增量区间均跨零，尚未建立稳定额外价值。可整理技术文章，NC未投稿就绪；当前证据限于AF识别及候选测量。SPH不放行、94条保留，医院/专家资料继续等待。原队列13:11已生成5主图+2补图和7页PPTX，实际视觉验收待完成；现已关闭新增绘图/PPTX，只交付文字prompt。</p><p><a href="A24_verified_review_20261023/verification.json">补充核验PASS</a> · <a href="A24_verified_review_20261023/创新性与文献对照评估.txt">创新性与文献对照</a> · <a href="A24_verified_review_20261023/主图设计prompts_基于已核验结果.txt">主图设计prompt</a> · <a href="manuscript_NC_A24_20261023/index.html">上一轮A24图件入口</a> · <a href="PHASE_A24_STATUS.json">最新状态</a></p><p>下方manuscript_NC_review_v2_20261021为历史稿件，本次核验和最新图件见以上链接。</p></section>'
    content,n=re.subn(r'<section id="a24">.*?</section>',lambda m:banner,content,count=1,flags=re.S);assert n==1
    eyebrow='2026年10月23日 · A24 E核验通过 · 仅文字设计 · NC尚未投稿就绪' if passed else '2026年10月23日 · A24 WSL纠正续跑 · NC尚未投稿就绪'
    content=re.sub(r'<p class="eyebrow">.*?</p>','<p class="eyebrow">'+eyebrow+'</p>',content,count=1,flags=re.S)
    overview.write_text(content,encoding='utf-8')
    page=P/'index.html';content=page.read_text(encoding='utf-8')
    note=f'<p class="note">最新状态快照 {html.escape(now)}：WSL原生环境及首32条实际预处理通过；全量提取{processed:,}/21,002，E拟合{completed}/12，累计{410+completed}/422。纠正续跑沿用原约19:16:36提取截止；审计通过后自动接续矢量图与可编辑PPTX。<a href="recovery_E_wsl_20261023_v2/extraction/status.json">实时进展</a> · <a href="recovery_E_wsl_20261023_v2/background_pipeline.json">后台流水线</a>。以下旧CPU/停止记录保留作历史证据。SPH及医院/专家等待不变；NC未就绪。</p>'
    if passed:
        note=f'<p class="note">最新快照 {html.escape(now)}：E全套数值来源审计及补充保存模型概率重放均PASS，21,002条提取完成，E12次拟合、A24共422次完成，无新增核验拟合。现无A24运行作业；新增图件/PPTX已关闭，只保留文字prompt。<a href="../../A24_verified_review_20261023/verification.json">完整核验</a> · <a href="../../A24_verified_review_20261023/创新性与文献对照评估.txt">创新性评估</a>。以下停止和CPU记录均为历史。SPH及医院/专家等待保留；NC未就绪。</p>'
    content,n=re.subn(r'<p class="note">.*?</p>',lambda m:note,content,count=1,flags=re.S);assert n==1
    page.write_text(content,encoding='utf-8');print(text)
if __name__=='__main__':main()
