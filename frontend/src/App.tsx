import { useEffect, useMemo, useState } from "react";
import {
  Activity, ArrowUpRight, BarChart3, BookOpenText, BrainCircuit, Check, ChevronDown,
  CircleHelp, Database, FileText, LayoutDashboard, LoaderCircle, Menu, Newspaper,
  Search, ShieldAlert, Sparkles, X,
} from "lucide-react";
import {
  Bar, BarChart, CartesianGrid, Cell, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { api } from "./services/api";
import type { DatasetStats, ExperimentCurves, Explanation, Metrics, ModelInfo, ModelName, Prediction } from "./types/api";

type Page = "Dashboard" | "Analyze News" | "Models" | "Performance" | "Dataset" | "About";
const nav: { name: Page; icon: typeof LayoutDashboard }[] = [
  { name: "Dashboard", icon: LayoutDashboard }, { name: "Analyze News", icon: Search },
  { name: "Models", icon: BrainCircuit }, { name: "Performance", icon: BarChart3 },
  { name: "Dataset", icon: Database }, { name: "About", icon: CircleHelp },
];
const SAMPLE_HEADLINE = "City council releases annual transportation report";
const SAMPLE_ARTICLE = "The city council released its annual transportation report on Tuesday after months of public meetings. Officials said the document summarizes bus reliability, road maintenance and pedestrian safety projects completed during the year. The report includes spending totals, contract dates and recommendations for future work. Residents can review the full document on the city website and submit written comments before the next council session. Independent auditors reviewed the listed expenses and confirmed the arithmetic in the financial appendix.";
const SAMPLE_INPUTS = [
  { name: "Example 1", headline: SAMPLE_HEADLINE, article: SAMPLE_ARTICLE },
  {
    name: "Example 2",
    headline: "University publishes annual research report",
    article: "The university released its annual public research report on Monday. The report describes ongoing studies in medicine, engineering and climate science, and includes grant amounts and publication references. Researchers submitted progress summaries to a review committee. The committee will publish its findings next month after considering feedback from faculty and students. Officials said the document is available on the university website and will be updated when the next reporting period is complete.",
  },
];
const pretty = (name: string) => name.toUpperCase();
const percent = (value?: number) => value == null ? "—" : `${(value * 100).toFixed(1)}%`;
const fmt = (value?: number | null) => value == null ? "—" : new Intl.NumberFormat().format(value);

function App() {
  const [page, setPage] = useState<Page>("Dashboard");
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [metrics, setMetrics] = useState<Record<string, Metrics>>({});
  const [experiments, setExperiments] = useState<ExperimentCurves>({});
  const [stats, setStats] = useState<DatasetStats | null>(null);
  const [minimumArticleWords, setMinimumArticleWords] = useState(50);
  const [dataError, setDataError] = useState("");
  const [mobileOpen, setMobileOpen] = useState(false);

  useEffect(() => {
    Promise.allSettled([api.models(), api.metrics(), api.dataset()]).then(([modelResult, metricResult, statsResult]) => {
      if (modelResult.status === "fulfilled") {
        setModels(modelResult.value.models);
        setMinimumArticleWords(modelResult.value.minimum_article_words);
      }
      else setDataError(modelResult.reason.message);
      if (metricResult.status === "fulfilled") {
        setMetrics(metricResult.value.models);
        setExperiments(metricResult.value.experiments ?? {});
      }
      if (statsResult.status === "fulfilled") setStats(statsResult.value);
    });
  }, []);

  const availableCount = models.filter((model) => model.available).length;
  const best = useMemo(() => Object.entries(metrics).sort((a, b) => b[1].f1 - a[1].f1)[0], [metrics]);
  const viewProps = { models, metrics, stats, dataError, setPage };
  return (
    <div className="shell">
      {mobileOpen && <button className="scrim" aria-label="Close navigation" onClick={() => setMobileOpen(false)} />}
      <aside className={`sidebar ${mobileOpen ? "sidebar-open" : ""}`}>
        <a className="brand" href="#" onClick={(event) => { event.preventDefault(); setPage("Dashboard"); }}>
          <span className="brand-mark"><Activity size={20} /></span><span>TruthLens<span className="brand-ai"> AI</span></span>
        </a>
        <div className="nav-label">WORKSPACE</div>
        <nav>{nav.map(({ name, icon: Icon }) => (
          <button key={name} className={`nav-link ${page === name ? "active" : ""}`} onClick={() => { setPage(name); setMobileOpen(false); }}>
            <Icon size={17} strokeWidth={1.8} />{name}{name === "Analyze News" && <span className="nav-dot" />}
          </button>
        ))}</nav>
        <div className="side-bottom">
          <div className="status-card"><span className={`status-dot ${availableCount ? "" : "idle"}`} />
            <div><strong>{availableCount ? `${availableCount} model${availableCount === 1 ? "" : "s"} ready` : "No models trained"}</strong><small>{availableCount ? "Inference is available" : "Train a model to begin"}</small></div>
          </div>
          <div className="sidebar-foot">DEEP LEARNING PROJECT <span>·</span> 2026</div>
        </div>
      </aside>
      <main className="main">
        <header className="topbar">
          <button className="icon-button mobile-menu" aria-label="Open navigation" onClick={() => setMobileOpen(true)}><Menu size={20} /></button>
          <div className="breadcrumbs"><span>Workspace</span><span className="crumb-slash">/</span><strong>{page}</strong></div>
          <div className="top-right"><span className="api-status"><span className="status-dot idle" /> Research preview</span><span className="avatar">TL</span></div>
        </header>
        <div className="content">
          {page === "Dashboard" && <Dashboard {...viewProps} best={best} availableCount={availableCount} />}
          {page === "Analyze News" && <Analyze models={models} minimumArticleWords={minimumArticleWords} />}
          {page === "Models" && <Models models={models} metrics={metrics} />}
          {page === "Performance" && <Performance metrics={metrics} experiments={experiments} />}
          {page === "Dataset" && <Dataset stats={stats} />}
          {page === "About" && <About />}
        </div>
      </main>
    </div>
  );
}

function PageTitle({ eyebrow, title, subtitle, action }: { eyebrow: string; title: string; subtitle: string; action?: React.ReactNode }) {
  return <div className="page-heading"><div><div className="eyebrow">{eyebrow}</div><h1>{title}</h1><p>{subtitle}</p></div>{action}</div>;
}
function EmptyState({ title, detail }: { title: string; detail: string }) {
  return <div className="empty-state"><span className="empty-icon"><Database size={20} /></span><strong>{title}</strong><p>{detail}</p></div>;
}
function MetricCard({ label, value, note, icon: Icon }: { label: string; value: string; note: string; icon: typeof Activity }) {
  return <div className="metric-card"><div className="metric-top"><span>{label}</span><span className="metric-icon"><Icon size={17} /></span></div><div className="metric-value">{value}</div><div className="metric-note">{note}</div></div>;
}

function Dashboard(props: {
  models: ModelInfo[]; metrics: Record<string, Metrics>; stats: DatasetStats | null; dataError: string;
  best?: [string, Metrics]; availableCount: number; setPage: (page: Page) => void;
}) {
  const { models, metrics, stats, dataError, best, availableCount, setPage } = props;
  return <>
    <PageTitle eyebrow="OVERVIEW" title="Fake News Detection AI" subtitle="Deep learning powered text classification — analyze patterns, not truth." action={<button className="button button-primary" onClick={() => setPage("Analyze News")}><Sparkles size={16} /> Analyze an article</button>} />
    {dataError && <div className="inline-alert"><ShieldAlert size={16} />{dataError}</div>}
    <div className="metrics-grid">
      <MetricCard label="DATASET ARTICLES" value={fmt(stats?.total)} note={stats ? `${fmt(stats.real_count)} real · ${fmt(stats.fake_count)} fake` : "Statistics unavailable"} icon={Newspaper} />
      <MetricCard label="BEST MODEL" value={best ? pretty(best[0]) : "—"} note={best ? `F1 score ${percent(best[1].f1)}` : "No evaluation results"} icon={BrainCircuit} />
      <MetricCard label="BEST F1 SCORE" value={best ? percent(best[1].f1) : "—"} note={best ? "From saved test evaluation" : "Train and evaluate a model"} icon={Activity} />
      <MetricCard label="AVAILABLE MODELS" value={`${availableCount} / ${models.length || 5}`} note="Trained artifacts detected" icon={BarChart3} />
    </div>
    <div className="section-heading"><div><h2>Model workspace</h2><p>Review model readiness and test results.</p></div><button className="text-button" onClick={() => setPage("Models")}>All models <ArrowUpRight size={15} /></button></div>
    <div className="model-grid">{(models.length ? models : ["dense", "rnn", "lstm", "gru", "bert"].map((name) => ({ name: name as ModelName, available: false, parameter_count: null, description: "Waiting for trained artifacts" }))).map((model) => {
      const result = metrics[model.name];
      return <article key={model.name} className="model-card"><div className="model-card-top"><span className="model-glyph"><BrainCircuit size={18} /></span><span className={`pill ${model.available ? "pill-green" : "pill-muted"}`}>{model.available ? "READY" : "NOT TRAINED"}</span></div><h3>{pretty(model.name)}</h3><p>{model.description}</p><div className="model-metric-row"><span>Test accuracy</span><strong>{percent(result?.test_accuracy)}</strong></div><div className="model-metric-row"><span>F1 score</span><strong>{percent(result?.f1)}</strong></div></article>;
    })}</div>
    <div className="notice"><ShieldAlert size={18} /><div><strong>Classification is not fact-checking</strong><p>TruthLens estimates whether text resembles patterns learned from its training data. It does not verify claims or consult external sources.</p></div></div>
  </>;
}

function Analyze({ models, minimumArticleWords }: { models: ModelInfo[]; minimumArticleWords: number }) {
  const [headline, setHeadline] = useState(SAMPLE_HEADLINE);
  const [article, setArticle] = useState(SAMPLE_ARTICLE);
  const [activeSample, setActiveSample] = useState<number | null>(0);
  const [model, setModel] = useState<ModelName>("lstm");
  const [result, setResult] = useState<Prediction | null>(null);
  const [explanation, setExplanation] = useState<Explanation | null>(null);
  const [explanationError, setExplanationError] = useState("");
  const [comparison, setComparison] = useState<Array<Prediction | { model: string; error: string; code: string }> | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const wordCount = article.trim() ? article.trim().split(/\s+/).length : 0;
  const selectedAvailable = models.find((item) => item.name === model)?.available ?? false;
  async function analyze(compare: boolean) {
    setError(""); setResult(null); setComparison(null); setExplanation(null); setExplanationError(""); setLoading(true);
    try {
      if (compare) setComparison((await api.compare(headline, article, model)).results);
      else {
        const prediction = await api.predict(headline, article, model);
        setResult(prediction);
        try { setExplanation(await api.explain(headline, article, model)); }
        catch (e) { setExplanationError(e instanceof Error ? e.message : "Word-level explanation is unavailable."); }
      }
    } catch (e) { setError(e instanceof Error ? e.message : "Analysis failed."); }
    finally { setLoading(false); }
  }
  return <>
    <PageTitle eyebrow="TEXT CLASSIFICATION" title="Analyze news" subtitle="Submit a headline and full article to see model predictions." />
    <div className="analyze-layout">
      <section className="panel input-panel">
        <div className="panel-title"><span className="title-icon blue"><FileText size={17} /></span><div><h2>Article input</h2><p>Editable sample text is prefilled for a quick test; replace it with your article.</p></div></div>
        <div className="sample-inputs" role="group" aria-label="Load example news inputs">
          {SAMPLE_INPUTS.map((sample, index) => (
            <button
              key={sample.name}
              type="button"
              className={`sample-tab ${activeSample === index ? "sample-tab-active" : ""}`}
              aria-pressed={activeSample === index}
              onClick={() => {
                setHeadline(sample.headline);
                setArticle(sample.article);
                setActiveSample(index);
                setResult(null);
                setComparison(null);
                setError("");
              }}
            >
              {sample.name}
            </button>
          ))}
        </div>
        <label className="field-label" htmlFor="headline">Headline</label>
        <input id="headline" className="text-input" value={headline} maxLength={1000} placeholder="Enter the news headline..." onChange={(e) => { setHeadline(e.target.value); setActiveSample(null); }} />
        <div className="field-row"><label className="field-label" htmlFor="article">Article text</label><span className={`word-count ${wordCount < minimumArticleWords ? "word-count-warning" : ""}`}>{wordCount} words · {minimumArticleWords} minimum</span></div>
        <textarea id="article" className="article-input" value={article} maxLength={100000} placeholder="Paste the complete article text here for analysis..." onChange={(e) => { setArticle(e.target.value); setActiveSample(null); }} />
        <div className="input-footer"><div><label className="field-label" htmlFor="model-select">Classification model</label><div className="select-wrap"><select id="model-select" value={model} onChange={(e) => setModel(e.target.value as ModelName)}>{(models.length ? models : ["dense", "rnn", "lstm", "gru", "bert"].map((name) => ({ name, available: false }))).map((item) => <option key={item.name} value={item.name}>{pretty(item.name)}{item.available ? "" : " · not trained"}</option>)}</select><ChevronDown size={15} /></div></div><span className={`pill ${selectedAvailable ? "pill-green" : "pill-muted"}`}><span className="status-dot" />{selectedAvailable ? "Model ready" : "Model unavailable"}</span></div>
        {error && <div className="form-error"><ShieldAlert size={16} />{error}</div>}
        <div className="analyze-actions"><button className="button button-primary grow" disabled={loading || !headline.trim() || !article.trim()} onClick={() => analyze(false)}>{loading ? <><LoaderCircle className="spin" size={16} /> Analyzing text…</> : <><Sparkles size={16} /> Analyze news</>}</button><button className="button button-secondary" disabled={loading || !headline.trim() || !article.trim()} onClick={() => analyze(true)}>Compare models</button></div>
        <p className="privacy-line"><ShieldAlert size={13} /> Articles are not stored. No external fact-checking is performed.</p>
      </section>
      <section className="panel result-panel">
        {!result && !comparison && !loading && <div className="result-empty"><span className="result-empty-icon"><Sparkles size={23} /></span><h2>Analysis results</h2><p>Your classification and probability scores will appear here after a model is available.</p><div className="result-steps"><span><Check size={14} /> Headline + article</span><span><Check size={14} /> Model inference</span><span><Check size={14} /> Probabilities</span></div></div>}
        {loading && <div className="result-empty"><LoaderCircle className="spin loading-icon" size={28} /><h2>Processing article…</h2><p>Running model inference. This may take a moment on first load.</p></div>}
        {result && <PredictionResult result={result} explanation={explanation} explanationError={explanationError} />}
        {comparison && <div className="comparison-result"><div className="panel-title"><span className="title-icon purple"><BarChart3 size={17} /></span><div><h2>Model comparison</h2><p>Results from available trained models.</p></div></div>{comparison.map((item) => "error" in item ? <div key={item.model} className="compare-row"><strong>{pretty(item.model)}</strong><span className="pill pill-muted">Unavailable</span><small>{item.error}</small></div> : <div key={item.model} className="compare-row"><strong>{pretty(item.model)}</strong><span className={`pill ${item.prediction === "REAL" ? "pill-green" : item.prediction === "FAKE" ? "pill-red" : "pill-amber"}`}>{item.prediction}</span><small>{percent(item.confidence)} confidence</small></div>)}</div>}
      </section>
    </div>
    <div className="footnote"><ShieldAlert size={15} /><span>Model-based classification — not independent fact verification. Predictions are probabilistic and may be wrong.</span></div>
  </>;
}

function PredictionResult({ result, explanation, explanationError }: { result: Prediction; explanation: Explanation | null; explanationError: string }) {
  const styles = result.prediction === "REAL" ? "result-real" : result.prediction === "FAKE" ? "result-fake" : "result-uncertain";
  return <div className="prediction-result"><div className="result-kicker">MODEL CLASSIFICATION</div><div className={`prediction-label ${styles}`}>{result.prediction}</div><div className="confidence-line"><span>Model confidence</span><strong>{percent(result.confidence)}</strong></div><div className="probability"><div><span><i className="legend-dot real" />REAL</span><strong>{percent(result.real_probability)}</strong></div><div className="progress"><span className="progress-real" style={{ width: `${result.real_probability * 100}%` }} /></div></div><div className="probability"><div><span><i className="legend-dot fake" />FAKE</span><strong>{percent(result.fake_probability)}</strong></div><div className="progress"><span className="progress-fake" style={{ width: `${result.fake_probability * 100}%` }} /></div></div><div className="interpretation"><span className="interpretation-icon"><BookOpenText size={16} /></span><div><strong>Model interpretation · LIME</strong><p>{(explanation?.note ?? explanationError) || "Generating local word-level approximation…"}</p>{explanation && <div className="word-weights">{explanation.words.map(({ word, weight }) => { const leansReal = (weight >= 0) === (explanation.prediction === "REAL"); return <span className={leansReal ? "weight-real" : "weight-fake"} title={`Contribution: ${weight.toFixed(3)}`} key={`${word}-${weight}`}>{word}</span>; })}</div>}</div></div><div className="result-meta"><span>MODEL USED</span><strong>{pretty(result.model)}</strong></div></div>;
}

function Models({ models, metrics }: { models: ModelInfo[]; metrics: Record<string, Metrics> }) {
  const descriptions: Record<string, [string, string, string]> = {
    dense: ["Embedding → Global average pooling → Dense", "Compact baseline; averages learned word representations.", "Word order is mostly discarded; long-range context is limited."],
    rnn: ["Embedding → SimpleRNN → Dense", "Processes tokens sequentially with a recurrent state.", "May struggle to retain information across long sequences."],
    lstm: ["Embedding → LSTM → Dense", "Gated recurrent network designed to retain longer context.", "More compute-intensive; still sensitive to dataset shortcuts."],
    gru: ["Embedding → GRU → Dense", "Gated recurrent network with a simpler parameterization than LSTM.", "Context window and learned corpus patterns constrain results."],
    bert: ["Pretrained Transformer + classification head", "Contextual representations; requires fine-tuning to this task.", "No weights are provided; must be fine-tuned and evaluated."],
  };
  const rows = models.length ? models : ["dense", "rnn", "lstm", "gru", "bert"].map((name) => ({ name, available: false, parameter_count: null, description: `${pretty(name)} text classifier` }));
  return <><PageTitle eyebrow="MODEL REGISTRY" title="Models" subtitle="Architecture, artifact readiness, and test-set performance." /><div className="model-list">{rows.map((model) => {
    const [architecture, advantage, limitation] = descriptions[model.name] ?? ["—", "—", "—"];
    const result = metrics[model.name];
    return <article className="panel registry-card" key={model.name}><div className="registry-head"><div className="registry-name"><span className="model-glyph"><BrainCircuit size={18} /></span><div><h2>{pretty(model.name)}</h2><p>{model.description}</p></div></div><span className={`pill ${model.available ? "pill-green" : "pill-muted"}`}>{model.available ? "ARTIFACTS READY" : "NOT TRAINED"}</span></div><div className="registry-info"><div><span>ARCHITECTURE</span><p>{architecture}</p></div><div><span>TRAINABLE PARAMETERS</span><p>{fmt(result?.parameter_count ?? model.parameter_count)}</p></div><div><span>ADVANTAGE</span><p>{advantage}</p></div><div><span>LIMITATION</span><p>{limitation}</p></div></div><div className="registry-scores">{(["test_accuracy", "precision", "recall", "f1"] as const).map((key) => <div key={key}><span>{key.replace("test_", "").replace("_", " ").toUpperCase()}</span><strong>{percent(result?.[key])}</strong></div>)}</div></article>;
  })}</div></>;
}

function Performance({ metrics, experiments }: { metrics: Record<string, Metrics>; experiments: ExperimentCurves }) {
  const chartData = Object.entries(metrics).map(([name, value]) => ({ name: pretty(name), Accuracy: value.test_accuracy * 100, Precision: value.precision * 100, Recall: value.recall * 100, F1: value.f1 * 100 }));
  const metricValues = chartData.flatMap(({ Accuracy, Precision, Recall, F1 }) => [Accuracy, Precision, Recall, F1]);
  const yAxisMin = metricValues.length
    ? Math.max(0, Math.floor((Math.min(...metricValues) - 5) / 5) * 5)
    : 0;
  const yAxisTicks = Array.from({ length: Math.floor((100 - yAxisMin) / 5) + 1 }, (_, index) => yAxisMin + index * 5);
  const [selected, setSelected] = useState("");
  const chosen = metrics[selected] ?? Object.values(metrics)[0];
  useEffect(() => { if (selected && !metrics[selected]) setSelected(""); }, [metrics, selected]);
  return <><PageTitle eyebrow="EVALUATION" title="Performance" subtitle="Metrics and charts are loaded from saved held-out test results." />{!chartData.length ? <div className="panel"><EmptyState title="Evaluation results unavailable" detail="Train models and save test metrics to populate this page. No sample charts are shown." /></div> : <>
    <div className="panel chart-panel"><div className="section-heading"><div><h2>Model comparison</h2><p>Saved test metrics · zoomed scale starts at {yAxisMin}% · 5-point ticks</p></div></div><div className="chart-wrap"><ResponsiveContainer width="100%" height="100%"><BarChart data={chartData} margin={{ top: 10, right: 10, left: -12, bottom: 0 }}><CartesianGrid stroke="#26324b" vertical={false} /><XAxis dataKey="name" stroke="#8290a9" tickLine={false} axisLine={false} /><YAxis domain={[yAxisMin, 100]} ticks={yAxisTicks} tickFormatter={(value: number) => `${value}%`} stroke="#8290a9" tickLine={false} axisLine={false} width={48} fontSize={10} /><Tooltip cursor={false} formatter={(value, name) => [`${Number(value).toFixed(1)}%`, name]} contentStyle={{ background: "#111a2d", border: "1px solid #2d3a55", borderRadius: 12 }} /><Bar dataKey="Accuracy" fill="#4f8cff" radius={[3, 3, 0, 0]} /><Bar dataKey="Precision" fill="#7c5cfc" radius={[3, 3, 0, 0]} /><Bar dataKey="Recall" fill="#22c55e" radius={[3, 3, 0, 0]} /><Bar dataKey="F1" fill="#f59e0b" radius={[3, 3, 0, 0]} /></BarChart></ResponsiveContainer></div><div className="chart-legend"><span><i className="legend-dot blue-dot" />Accuracy</span><span><i className="legend-dot purple-dot" />Precision</span><span><i className="legend-dot real" />Recall</span><span><i className="legend-dot amber-dot" />F1</span></div></div>
    <div className="panel chart-panel"><div className="section-heading"><div><h2>Training evaluation</h2><p>Saved training history and confusion matrix.</p></div><div className="select-wrap compact"><select value={selected || Object.keys(metrics)[0]} onChange={(e) => setSelected(e.target.value)}>{Object.keys(metrics).map((name) => <option value={name} key={name}>{pretty(name)}</option>)}</select><ChevronDown size={15} /></div></div>
      {chosen?.history && Object.values(chosen.history).some((series) => series.length > 0) ? <div className="history-grid">{(["accuracy", "loss"] as const).map((key) => {
        const trainSeries = chosen.history?.[key] ?? [];
        const validationSeries = chosen.history?.[`val_${key}`] ?? [];
        const bertTrain = key === "loss" ? chosen.history?.train_loss ?? [] : [];
        const maxLength = Math.max(trainSeries.length, validationSeries.length, bertTrain.length);
        if (!maxLength) return <div className="history-box" key={key}><span>{key.toUpperCase()}</span><p className="subtle">No saved {key} history.</p></div>;
        const rows = Array.from({ length: maxLength }, (_, index) => ({
          epoch: `Epoch ${index + 1}`,
          Train: trainSeries[index] ?? bertTrain[index],
          Validation: validationSeries[index],
        }));
        return <div className="history-box" key={key}><span>{key.toUpperCase()} · PER EPOCH</span><div className="line-chart"><ResponsiveContainer width="100%" height="100%"><LineChart data={rows} margin={{ top: 12, right: 8, left: -23, bottom: 0 }}><CartesianGrid stroke="#26324b" vertical={false} /><XAxis dataKey="epoch" stroke="#8290a9" tickLine={false} axisLine={false} /><YAxis stroke="#8290a9" tickLine={false} axisLine={false} /><Tooltip contentStyle={{ background: "#111a2d", border: "1px solid #2d3a55", borderRadius: 12 }} /><Line type="monotone" dataKey="Train" stroke="#4f8cff" strokeWidth={2} connectNulls dot={false} /><Line type="monotone" dataKey="Validation" stroke="#f59e0b" strokeWidth={2} connectNulls dot={false} /></LineChart></ResponsiveContainer></div></div>;
      })}</div> : <EmptyState title="Training curves unavailable" detail="The selected model has no saved per-epoch history." />}
      {chosen?.confusion_matrix ? <div className="confusion"><span>CONFUSION MATRIX · ACTUAL ROWS / PREDICTED COLUMNS</span><div className="matrix-grid"><span /><b>FAKE</b><b>REAL</b><b>FAKE</b>{chosen.confusion_matrix[0]?.map((value, i) => <strong className={i === 0 ? "matrix-correct" : "matrix-wrong"} key={`f${i}`}>{value}</strong>)}<b>REAL</b>{chosen.confusion_matrix[1]?.map((value, i) => <strong className={i === 1 ? "matrix-correct" : "matrix-wrong"} key={`r${i}`}>{value}</strong>)}</div></div> : <p className="subtle">Confusion matrix unavailable.</p>}
    </div>
    </>}
    <ExperimentCharts experiments={experiments} />
  </>;
}

function ExperimentCharts({ experiments }: { experiments: ExperimentCurves }) {
  const entries = Object.entries(experiments);
  if (!entries.length) return null;
  const colors = ["#4f8cff", "#f59e0b", "#7c5cfc", "#22c55e"];
  return <div className="panel chart-panel"><div className="section-heading"><div><h2>Overfit / underfit experiments</h2><p>Saved train and validation curves; shown only when the experiment artifact exists.</p></div></div><div className="history-grid">{(["accuracy", "loss"] as const).map((metric) => {
    const series = entries.flatMap(([experiment, history]) => ["", "val_"].map((prefix) => ({
      name: `${pretty(experiment)} ${prefix ? "validation" : "train"}`,
      values: history[`${prefix}${metric}`] ?? [],
    })).filter((item) => item.values.length > 0));
    const count = Math.max(0, ...series.map((item) => item.values.length));
    if (!count) return null;
    const rows = Array.from({ length: count }, (_, index) => Object.fromEntries([
      ["epoch", `Epoch ${index + 1}`], ...series.map((item) => [item.name, item.values[index]]),
    ]));
    return <div className="history-box" key={metric}><span>{metric.toUpperCase()}</span><div className="line-chart"><ResponsiveContainer width="100%" height="100%"><LineChart data={rows} margin={{ top: 12, right: 8, left: -23, bottom: 0 }}><CartesianGrid stroke="#26324b" vertical={false} /><XAxis dataKey="epoch" stroke="#8290a9" tickLine={false} axisLine={false} /><YAxis stroke="#8290a9" tickLine={false} axisLine={false} /><Tooltip contentStyle={{ background: "#111a2d", border: "1px solid #2d3a55", borderRadius: 12 }} />{series.map((item, index) => <Line key={item.name} type="monotone" dataKey={item.name} stroke={colors[index % colors.length]} strokeWidth={2} connectNulls dot={false} />)}</LineChart></ResponsiveContainer></div></div>;
  })}</div></div>;
}

function Dataset({ stats }: { stats: DatasetStats | null }) {
  const distribution = stats ? Object.entries(stats.class_distribution).map(([name, value]) => ({ name, count: value })) : [];
  return <><PageTitle eyebrow="DATA OVERVIEW" title="Dataset" subtitle="Statistics are computed from the prepared dataset; no fabricated values are displayed." />{!stats ? <div className="panel"><EmptyState title="Dataset statistics unavailable" detail="Add the supported dataset files and run the preparation script to calculate these statistics." /></div> : <>
    <div className="metrics-grid three"><MetricCard label="TOTAL ARTICLES" value={fmt(stats.total)} note="After cleaning and deduplication" icon={Newspaper} /><MetricCard label="REAL ARTICLES" value={fmt(stats.real_count)} note="Prepared labeled articles" icon={Check} /><MetricCard label="FAKE ARTICLES" value={fmt(stats.fake_count)} note="Prepared labeled articles" icon={ShieldAlert} /></div>
    <div className="panel chart-panel"><h2>Class distribution</h2><p className="panel-subtitle">Prepared dataset counts</p><div className="dataset-chart"><ResponsiveContainer width="100%" height="100%"><BarChart data={distribution}><CartesianGrid stroke="#26324b" vertical={false} /><XAxis dataKey="name" stroke="#8290a9" tickLine={false} axisLine={false} /><YAxis stroke="#8290a9" tickLine={false} axisLine={false} /><Tooltip contentStyle={{ background: "#111a2d", border: "1px solid #2d3a55", borderRadius: 12 }} /><Bar dataKey="count" radius={[6, 6, 0, 0]}>{distribution.map((entry) => <Cell key={entry.name} fill={entry.name === "REAL" ? "#22c55e" : "#ef4444"} />)}</Bar></BarChart></ResponsiveContainer></div><div className="split-strip">{Object.entries(stats.split_counts ?? {}).map(([label, count]) => <span key={label}><b>{fmt(count)}</b>{label} split</span>)}</div></div>
    <div className="panel process-panel"><h2>Preparation pipeline</h2><div className="process-steps"><span>HTML / URL cleanup</span><span>Empty + duplicate removal</span><span>Reuters dateline cleanup</span><span>Stratified 70 / 15 / 15 split</span><span>Training-only tokenizer fit</span></div><p>Headline and article text are combined for classification. Dates, source fields, and subject metadata are not model inputs.</p></div>
  </>}</>;
}

function About() {
  return <><PageTitle eyebrow="PROJECT" title="About TruthLens AI" subtitle="A deep learning course project in natural language processing." /><div className="about-grid"><section className="panel about-card"><span className="title-icon blue"><Search size={18} /></span><h2>Problem</h2><p>Misleading news text can spread rapidly. This project studies how neural language models classify examples using patterns from labeled training data.</p></section><section className="panel about-card"><span className="title-icon purple"><BrainCircuit size={18} /></span><h2>Approach</h2><p>Dense, recurrent, and Transformer-based text classifiers can be trained and evaluated on a held-out dataset, then served through a typed API.</p></section><section className="panel about-card"><span className="title-icon amber"><ShieldAlert size={18} /></span><h2>Limitations</h2><p>This is not a fact-checker. Models can reflect dataset bias, fail on new topics, and make confident mistakes. Scores represent learned text patterns, not factual certainty.</p></section><section className="panel about-card"><span className="title-icon green"><BookOpenText size={18} /></span><h2>Technology</h2><p>React, TypeScript, Vite, FastAPI, TensorFlow/Keras, Pandas, NumPy, and scikit-learn. Training is an offline workflow; the browser never trains models.</p></section></div></>;
}

export default App;
