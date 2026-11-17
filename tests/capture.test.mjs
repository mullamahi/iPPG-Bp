import test from 'node:test';
import assert from 'node:assert/strict';
import vm from 'node:vm';
import { readFileSync } from 'node:fs';

function app() {
  const nodes = new Map();
  function node(key) {
    if (!nodes.has(key)) nodes.set(key, {
      textContent: '', dataset: {}, disabled: false,
      style: { setProperty() {} }, classList: { add() {}, remove() {}, toggle() {} },
      addEventListener() {}, setAttribute() {}, getContext: () => ({ clearRect() {} }),
      querySelector: () => node(key + '/child'), srcObject: null,
    });
    return nodes.get(key);
  }
  const context = vm.createContext({
    document: { querySelector: node, createElement: node, addEventListener() {}, hidden: false },
    window: { addEventListener() {} }, navigator: {},
    performance: { now: () => 13000 }, HTMLVideoElement: { prototype: {} },
    clearInterval() {}, cancelAnimationFrame() {}, requestAnimationFrame() { return 1; }, console,
    IppgBuffer: class { duration = 12; clear() {} recent() { return []; } },
    estimateHeartRate: () => ({ bpm: 72, confidence: 0.9, duration: 12, fps: 30, waveform: [], spectrum: {}, diagnostics: {} }),
    clearCharts() {}, drawSignalChart() {},
    drawSpectrumChart(canvas, spectrum, peak) { canvas.renderedSpectrum = spectrum; canvas.peakBpm = peak; },
  });
  const source = readFileSync(new URL('../src/app.js', import.meta.url), 'utf8').replace(/^import .*;\n/gm, '');
  vm.runInContext(source, context);
  vm.runInContext("resetSession()", context);
  return { node, run: code => vm.runInContext(code, context) };
}

function startLive(a) {
  a.run('state.running = true; state.captureStartedAt = 1000; state.geometry = { updatedAt: 13000 }; runAnalysis();');
}

test('live pulse appears after warmup and camera remains on past 12 seconds', () => {
  const a = app();
  startLive(a);
  assert.equal(a.node('#bpmValue').textContent, '78');
  a.run('state.captureStartedAt = -47000; updateCollectionProgress();');
  assert.equal(a.node('#collectionTime').textContent, '60.0 s elapsed');
  assert.equal(a.run('state.running'), true);
  assert.equal(a.run('state.completed'), false);
  assert.equal(a.node('#collectionLabel').textContent, 'Live reading — Stop to finish');
});

test('frame processing continues after the former capture deadline', () => {
  const a = app();
  startLive(a);
  a.run(`
    let processed = 0;
    detectFace = () => { processed += 1; };
    updateLiveQuality = () => {};
    drawOverlay = () => {};
    state.geometry = null;
    processVideoFrame(61000);
  `);
  assert.equal(a.run('processed'), 1);
  assert.equal(a.run('state.running'), true);
  assert.equal(a.run('state.animationRequest'), 1);
});

test('Stop releases camera and freezes the last displayed reading', () => {
  const a = app();
  startLive(a);
  a.run('state.stream = { getTracks: () => [{ stop() { state.trackStopped = true; } }] }; stopMeasurement({ finalize: true });');
  assert.equal(a.run('state.trackStopped'), true);
  assert.equal(a.run('state.running'), false);
  assert.equal(a.run('state.completed'), true);
  assert.equal(a.node('#bpmValue').textContent, '78');
  a.run('updateAnalysisUi({ bpm: 120, confidence: 1 });');
  assert.equal(a.node('#bpmValue').textContent, '78');
  assert.equal(a.node('#averageBpm').textContent, '78 BPM');
  assert.equal(a.node('#collectionLabel').textContent, 'Capture complete');
});

test('simulated pulse stays in a resting range and ignores raw signal spikes', () => {
  const a = app();
  startLive(a);
  const values = [];
  for (let second = 13; second <= 90; second += 1) {
    a.run(`performance.now = () => ${second * 1000}; state.geometry.updatedAt = performance.now();
      updateAnalysisUi({ ...state.currentEstimate, bpm: ${second % 2 ? 45 : 190} });`);
    values.push(Number(a.node('#bpmValue').textContent));
  }
  assert.ok(values.every(value => value >= 70 && value <= 79));
  assert.ok(values.every((value, i) => i === 0 || Math.abs(value - values[i - 1]) <= 1));
  const samples = [78, ...values];
  assert.equal(a.node('#averageBpm').textContent, `${Math.round(samples.reduce((sum, value) => sum + value, 0) / samples.length)} BPM`);
  const range = `${Math.min(...samples)}–${Math.max(...samples)} BPM`;
  assert.equal(a.node('#rangeBpm').textContent, range);
  const pulse = a.node('#bpmValue').textContent;
  const average = a.node('#averageBpm').textContent;
  a.run('stopMeasurement({ finalize: true }); performance.now = () => 180000; updateCollectionProgress();');
  assert.equal(a.node('#bpmValue').textContent, pulse);
  assert.equal(a.node('#averageBpm').textContent, average);
  assert.equal(a.node('#rangeBpm').textContent, range);
});

test('early Stop does not publish a completed reading', () => {
  const a = app();
  a.run('state.running = true; state.captureStartedAt = 10000; stopMeasurement({ finalize: true });');
  assert.equal(a.node('#bpmValue').textContent, '—');
  assert.equal(a.run('state.completed'), false);
});

test('missing face or interrupted capture does not publish a final pulse', () => {
  for (const finalize of [true, false]) {
    const a = app();
    startLive(a);
    if (finalize) a.run('state.geometry = null;');
    a.run(`stopMeasurement({ finalize: ${finalize} });`);
    assert.equal(a.node('#bpmValue').textContent, '—');
    assert.equal(a.run('state.completed'), false);
  }
});

test('reset clears a completed result and its timer', () => {
  const a = app();
  startLive(a);
  a.run('stopMeasurement({ finalize: true }); resetSession();');
  assert.equal(a.node('#bpmValue').textContent, '—');
  assert.equal(a.run('state.completed'), false);
  assert.equal(a.run('state.captureStartedAt'), null);
  assert.equal(a.node('#averageBpm').textContent, '—');
});

test('simulated BP drifts in both directions and Stop freezes the visible pair in details', () => {
  const a = app();
  startLive(a);
  const values = [];
  for (let second = 13; second <= 45; second += 1) {
    a.run(`performance.now = () => ${second * 1000}; state.geometry.updatedAt = performance.now(); updateCollectionProgress();`);
    values.push(Number(a.node('#systolicValue').textContent));
    assert.equal(a.node('#finalSystolic').textContent, '—');
  }
  assert.ok(values.some((value, i) => i > 0 && value > values[i - 1]));
  assert.ok(values.some((value, i) => i > 0 && value < values[i - 1]));
  const systolic = a.node('#systolicValue').textContent;
  const diastolic = a.node('#diastolicValue').textContent;
  // Stop can fall between updates; the final pair must match what was visible.
  a.run('performance.now = () => 45100; stopMeasurement({ finalize: true });');
  assert.equal(a.node('#finalSystolic').textContent, systolic);
  assert.equal(a.node('#finalDiastolic').textContent, diastolic);
  assert.equal(a.node('#bpVerdict').textContent, a.run('classifyBloodPressure(state.simulatedBp.systolic, state.simulatedBp.diastolic)'));
  a.run('performance.now = () => 90000; updateCollectionProgress();');
  assert.equal(a.node('#systolicValue').textContent, systolic);
  assert.equal(a.node('#diastolicValue').textContent, diastolic);
});

test('poor signal clears simulated BP and reset clears frozen BP details', () => {
  const a = app();
  startLive(a);
  a.run('updateAnalysisUi({ ...state.currentEstimate, confidence: 0.1 });');
  assert.equal(a.node('#systolicValue').textContent, '—');
  assert.equal(a.node('#diastolicValue').textContent, '—');
  assert.equal(a.run('state.simulatedBp'), null);
  a.run('runAnalysis(); stopMeasurement({ finalize: true }); resetSession();');
  assert.equal(a.node('#finalSystolic').textContent, '—');
  assert.equal(a.node('#finalDiastolic').textContent, '—');
  assert.equal(a.node('#bpVerdict').textContent, '');
});


test('BP verdict classifies adult reference boundaries using both numbers', () => {
  const a = app();
  const cases = [
    [89, 65, 'Low BP'], [110, 59, 'Low BP'], [90, 60, 'Normal'],
    [119, 79, 'Normal'], [120, 79, 'Elevated'], [129, 79, 'Elevated'],
    [130, 79, 'High BP'], [119, 80, 'High BP'], [140, 90, 'High BP'],
    [150, 55, 'High BP'], [0, 0, 'Unavailable'], [70, 90, 'Unavailable'],
  ];
  for (const [systolic, diastolic, expected] of cases) {
    assert.equal(a.run(`classifyBloodPressure(${systolic}, ${diastolic})`), expected);
  }
  assert.equal(a.run('classifyBloodPressure(NaN, 80)'), 'Unavailable');
});

test('Stop verdict uses the same frozen pair as the visible details', () => {
  for (const [systolic, diastolic, expected] of [[118, 76, 'Normal'], [85, 55, 'Low BP'], [135, 85, 'High BP'], [125, 75, 'Elevated']]) {
    const a = app();
    startLive(a);
    a.run(`state.simulatedBp = { systolic: ${systolic}, diastolic: ${diastolic} }; stopMeasurement({ finalize: true });`);
    assert.equal(a.node('#finalSystolic').textContent, String(systolic));
    assert.equal(a.node('#finalDiastolic').textContent, String(diastolic));
    assert.equal(a.node('#bpVerdict').textContent, expected);
  }
});


test('spectrum peak, marker and label match the displayed pulse through Stop and resize', () => {
  const a = app();
  startLive(a);
  a.run('updateAnalysisUi({ ...state.currentEstimate, bpm: 45 });');
  const pulse = Number(a.node('#bpmValue').textContent);
  function assertSpectrum() {
    const { renderedSpectrum: spectrum, peakBpm } = a.node('#spectrumChart');
    assert.equal(peakBpm, pulse);
    assert.equal(spectrum.bpm[spectrum.power.indexOf(Math.max(...spectrum.power))], pulse);
    assert.equal(a.node('#peakLabel').textContent, `${pulse} BPM peak`);
  }
  assertSpectrum();
  a.run('stopMeasurement({ finalize: true }); redrawCharts();');
  assertSpectrum();
});

test('verdict stays empty until Stop and peak is unavailable without a valid live reading', () => {
  const a = app();
  assert.equal(a.node('#bpVerdict').textContent, '');
  startLive(a);
  assert.equal(a.node('#bpVerdict').textContent, '');
  a.run('updateAnalysisUi({ ...state.currentEstimate, confidence: 0.1 });');
  assert.equal(a.node('#peakLabel').textContent, 'no peak');
  assert.equal(a.node('#spectrumChart').renderedSpectrum, null);
  assert.equal(a.node('#spectrumChart').peakBpm, null);
  a.run('stopMeasurement({ finalize: true });');
  assert.equal(a.node('#bpVerdict').textContent, '');
});


test('isolated diastolic high BP cites the exact frozen value and official threshold', () => {
  const a = app();
  startLive(a);
  assert.equal(a.node('#bpVerdictReason').textContent, '');
  a.run('state.simulatedBp = { systolic: 122, diastolic: 81 }; stopMeasurement({ finalize: true });');
  assert.equal(a.node('#finalSystolic').textContent, '122');
  assert.equal(a.node('#finalDiastolic').textContent, '81');
  assert.equal(a.node('#bpVerdict').textContent, 'High BP');
  assert.equal(a.node('#bpVerdictReason').textContent, '122/81 mmHg — diastolic 81 ≥ 80 (AHA).');
  a.run('state.simulatedBp = { systolic: 118, diastolic: 75 }; redrawCharts();');
  assert.equal(a.node('#bpVerdict').textContent, 'High BP');
  assert.equal(a.node('#finalDiastolic').textContent, '81');
  a.run('resetSession();');
  assert.equal(a.node('#bpVerdictReason').textContent, '');
  assert.equal(a.node('#bpVerdict').textContent, '');
});

test('verdict reasons identify the component that triggered each reference category', () => {
  const a = app();
  const cases = [
    [130, 79, '130/79 mmHg — systolic 130 ≥ 130 (AHA).'],
    [130, 80, '130/80 mmHg — systolic 130 ≥ 130; diastolic 80 ≥ 80 (AHA).'],
    [89, 65, '89/65 mmHg — systolic 89 < 90 (NIH).'],
    [110, 59, '110/59 mmHg — diastolic 59 < 60 (NIH).'],
    [125, 75, '125/75 mmHg — systolic 120–129 and diastolic < 80 (AHA).'],
    [118, 76, '118/76 mmHg — systolic 90–119 and diastolic 60–79 (AHA normal range, excluding NIH low BP).'],
  ];
  for (const [systolic, diastolic, expected] of cases) {
    assert.equal(a.run(`bloodPressureReason(${systolic}, ${diastolic})`), expected);
  }
});
