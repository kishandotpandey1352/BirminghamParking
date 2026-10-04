import React from 'react'
import { createRoot } from 'react-dom/client'
import { Activity, ArrowRight, BrainCircuit, CarFront, Gauge, RefreshCw, ShieldCheck, TrainFront } from 'lucide-react'
import './styles.css'

const api = async (path, options) => {
  const response = await fetch(path, options)
  const body = await response.json()
  if (!response.ok) throw new Error(body.detail || 'Request failed')
  return body
}

function Metric({ label, value, unit, accent = false }) {
  return <article className={`metric ${accent ? 'accent' : ''}`}><span>{label}</span><strong>{value ?? '--'}</strong><small>{unit}</small></article>
}

function Prediction({ options, selectedExample, setSelectedExample, result, runPrediction }) {
  const example = options.examples[selectedExample]
  return <section className="view-grid"><div className="section-heading"><div><p className="eyebrow">01 / FORECAST</p><h2>Make the next half hour legible.</h2><p>Select a real historical observation and inspect the bounded output from the versioned model.</p></div><div className="horizon"><Gauge size={18} /> 30 min horizon</div></div><div className="control-row"><label>Historical example<select value={selectedExample} onChange={event => setSelectedExample(Number(event.target.value))}>{options.examples.map((item, index) => <option value={index} key={`${item.car_park_id}-${item.observed_at}`}>{item.car_park_id} · {item.observed_at}</option>)}</select></label><button onClick={runPrediction}><ArrowRight size={17} /> Forecast availability</button></div><div className="metric-grid"><Metric label="Observed occupancy" value={example?.occupancy} unit="occupied spaces" /><Metric label="Forecast occupancy" value={result ? Math.round(result.bounded_occupancy_forecast) : null} unit="bounded spaces" /><Metric label="Available in 30 min" value={result ? Math.round(result.predicted_available_spaces) : null} unit="spaces" accent /></div>{result && <div className="trace"><span>Forecast trace</span><b>{result.observation_timestamp}</b><ArrowRight size={16} /><b>{result.forecast_timestamp}</b><em>raw {result.raw_occupancy_forecast.toFixed(1)} · bounded to capacity</em></div>}</section>
}

function Monitoring({ runReplay, report, mode }) {
  const metric = report?.mae == null ? 'Insufficient data' : report.mae.toFixed(1)
  return <section className="view-grid"><div className="section-heading"><div><p className="eyebrow">02 / MONITORING</p><h2>Watch the model after it ships.</h2><p>Drift recommends investigation. It never silently retrains or deploys a replacement.</p></div><Activity size={40} strokeWidth={1.3} /></div><div className="scenario-strip"><button className={mode === 'baseline' ? 'selected' : ''} onClick={() => runReplay('baseline')}>Historical replay</button><button className={mode === 'data_drift' ? 'selected' : ''} onClick={() => runReplay('data_drift')}>Data drift</button><button className={mode === 'concept_drift' ? 'selected' : ''} onClick={() => runReplay('concept_drift')}>Concept drift</button></div>{report && <><div className="metric-grid"><Metric label="Model MAE" value={metric} unit="occupied spaces" /><Metric label="Persistence MAE" value={report.baseline_mae == null ? 'Unavailable' : report.baseline_mae.toFixed(1)} unit="same labelled cohort" /><Metric label="Drift status" value={report.drift.status} unit={`${report.labelled_samples} labelled samples`} accent={report.drift.status === 'warning'} /></div><div className="report-panel"><div><span>Data quality</span><p>{report.data_quality?.excluded_rows} excluded rows · {report.data_quality?.duplicate_rows} duplicates</p><span>Initial label state</span><p>{report.initial_window.reason}</p></div><pre>{JSON.stringify(report.drift, null, 2)}</pre></div></>}</section>
}

function ModelOps({ metadata, versions, refresh }) {
  const [version, setVersion] = React.useState(`real-${new Date().toISOString().slice(0, 10)}`)
  const [message, setMessage] = React.useState('')
  const train = async () => { try { const result = await api('/api/models/train', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ version }) }); setMessage(`Trained ${result.model_version}`); refresh() } catch (error) { setMessage(error.message) } }
  return <section className="view-grid"><div className="section-heading"><div><p className="eyebrow">03 / MODEL OPS</p><h2>Train deliberately. Version everything.</h2><p>Retraining is guarded behind <code>PARKPULSE_ENABLE_TRAINING=1</code> and writes an immutable versioned artifact.</p></div><BrainCircuit size={40} strokeWidth={1.3} /></div><div className="ops-panel"><div><label>New model version<input value={version} onChange={event => setVersion(event.target.value)} /></label><button onClick={train}><RefreshCw size={17} /> Train and promote locally</button>{message && <p className="status">{message}</p>}</div><div><span>Current release</span><h3>{metadata.model_version || 'Loading...'}</h3><p>Trained {metadata.trained_at || '--'}</p></div></div><div className="version-list">{versions.map(item => <div key={`${item.model_version}-${item.path}`}><ShieldCheck size={17} /><b>{item.model_version}</b><span>{item.trained_at}</span><code>{item.path}</code></div>)}</div></section>
}

function App() {
  const [view, setView] = React.useState('prediction')
  const [options, setOptions] = React.useState({ car_parks: [], examples: [] })
  const [metadata, setMetadata] = React.useState({})
  const [versions, setVersions] = React.useState([])
  const [selectedExample, setSelectedExample] = React.useState(0)
  const [result, setResult] = React.useState(null)
  const [report, setReport] = React.useState(null)
  const [mode, setMode] = React.useState('baseline')
  const [error, setError] = React.useState('')
  const refresh = async () => { try { setOptions(await api('/api/options')); setMetadata(await api('/api/model')); setVersions((await api('/api/models')).versions); setError('') } catch (err) { setError(err.message) } }
  React.useEffect(() => { refresh() }, [])
  const runPrediction = async () => { const item = options.examples[selectedExample]; try { setResult(await api('/api/predict', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ car_park_id: item.car_park_id, observed_at: item.observed_at, occupancy: item.occupancy }) })); setError('') } catch (err) { setError(err.message) } }
  const runReplay = async selectedMode => { try { setMode(selectedMode); setReport(await api(`/api/replay/${selectedMode}`)); setError('') } catch (err) { setError(err.message) } }
  return <main><header className="hero"><div><p className="eyebrow">PARKPULSE / MLOPS STUDIO</p><h1>Thirty minutes ahead.</h1><p className="lede">A production-shaped parking forecast where the model, the artifact, and the evidence remain visible.</p></div><div className="hero-mark"><CarFront size={34} /><span>2016<br />HISTORICAL</span></div></header><nav className="tabs">{[['prediction', 'Forecast'], ['monitoring', 'Monitoring'], ['models', 'Model ops']].map(([key, label]) => <button className={view === key ? 'active' : ''} onClick={() => setView(key)} key={key}>{label}</button>)}</nav>{error && <div className="error">{error}</div>}{view === 'prediction' && <Prediction options={options} selectedExample={selectedExample} setSelectedExample={setSelectedExample} result={result} runPrediction={runPrediction} />}{view === 'monitoring' && <Monitoring runReplay={runReplay} report={report} mode={mode} />}{view === 'models' && <ModelOps metadata={metadata} versions={versions} refresh={refresh} />}<footer><span><TrainFront size={15} /> {metadata.model_version || 'model loading'}</span><span>FastAPI · React · Docker-ready</span></footer></main>
}

createRoot(document.getElementById('root')).render(<App />)