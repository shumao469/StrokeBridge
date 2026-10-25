// Pure mock-DOM test of the ACTUAL inline demo JavaScript, not a browser QA claim.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import {fileURLToPath} from 'node:url';
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const html = fs.readFileSync(path.join(root, 'demo.html'), 'utf8');
const bundle = JSON.parse(fs.readFileSync(path.join(root, 'data/audited_aggregate_results.json'), 'utf8'));
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
const ids = {};
for (const id of ['phase', 'policy', 'seed', 'lead', 'stage', 'results', 'example']) {
  ids[id] = {value: '', textContent: '', listeners: {},
    addEventListener(name, callback) { this.listeners[name] = callback; }};
  Object.defineProperty(ids[id], 'innerHTML', {
    get() { return this.content || ''; },
    set(value) { this.content = value; if (id === 'policy') this.value = value.match(/<option>(.*?)<\/option>/)?.[1] || ''; }
  });
}
Object.assign(ids.phase, {value:'A32'});
Object.assign(ids.seed, {value:'2026102701'});
Object.assign(ids.lead, {value:'v1'});
Object.assign(ids.stage, {value:'selected'});
const context = vm.createContext({document:{getElementById(id) { return ids[id]; }}, console});
vm.runInContext(script, context);
assert.equal(ids.policy.value, 'W1_window_aware');
const pct = v => v === null || v === undefined ? '—' : (100*v).toFixed(2)+'%';
let configurations = 0;
let cells = 0;
for (const phase of ['A32', 'A33']) {
  ids.phase.value = phase;
  ids.phase.listeners.change();
  const rows = bundle[phase+'_results'];
  for (const policy of [...new Set(rows.map(r=>r.policy))])
    for (const seed of [2026102701, 2026102702])
      for (const lead of ['ii', 'v1'])
        for (const stage of ['raw', 'selected', 'reported']) {
          Object.assign(ids.policy, {value:policy});
          Object.assign(ids.seed, {value:String(seed)});
          Object.assign(ids.lead, {value:lead});
          Object.assign(ids.stage, {value:stage});
          ids.stage.listeners.change();
          const expectedRows = rows.filter(r=>r.policy===policy && r.seed===seed && r.lead===lead);
          const expectedHTML = expectedRows.map(r=>{
            const s=r.stages[stage];
            return '<tr><td>'+r.arm+'</td><td>'+pct(s.recall)+'</td><td>'+pct(s.precision_lower)+'</td><td>'+pct(s.negative_recall)+'</td><td>'+pct(s.biphasic_recall)+'</td><td>'+pct(r.strict_patient_joint_reported)+'</td></tr>';
          }).join('');
          assert.equal(ids.results.innerHTML, expectedHTML);
          configurations++;
          cells += expectedRows.length * 6;
        }
}
assert.equal(configurations, 96);
assert.ok(ids.example.textContent.includes('SYNTHETIC-ONLY-0001'));
assert.ok(!html.includes('type="file"'));
console.log(JSON.stringify({state:'PASS',actual_inline_script:true,mock_DOM_only:true,browser_render_test:false,configurations,cells_compared:cells,training_fits:0,clinical_predictions:0}));
