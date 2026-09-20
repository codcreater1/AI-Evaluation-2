# AI Evaluation Platform: Core Evaluation Engine

Çekirdek motor: şemalar, pluggable evaluator'lar, runner, LLM judge, FastAPI + PostgreSQL.
Diğer modüller (dataset, Langfuse, dashboard, CI) bunun üstüne kurulur.

## Hızlı başlangıç
```bash
cp .env.example .env
docker compose up --build        # Postgres + API  ->  http://localhost:8000/docs
# ya da lokal:
uv sync --extra dev   # veya: pip install -e ".[dev]"
pytest -q
```
Örnek çağrı: `curl -X POST localhost:8000/evaluations/run -H 'content-type: application/json' -d @examples/sample_request.json`

## Mimari
```
app/schemas/      EvaluationCase, ExecutionResult (sistemin ürettiği), EvaluationResult
app/evaluators/   base.py (BaseEvaluator + registry), deterministic/, llm_judge/
app/engine/       runner.py, aggregation.py (ortalama, eşik, karşılaştırma), classification.py (TP/TN/FP/FN, F1), adapters.py
app/config/       YAML -> RunConfig -> evaluator listesi
app/integrations/ ScoreSink protokolü + LangfuseSink (skorları trace'e yazar)
app/db, app/api/  SQLAlchemy modelleri, FastAPI router'ları
```

### Şemalar
- **EvaluationCase**: `id, system, input{}, expected_output{}, metadata{}` (serbest dict; chatbot varsayımı yok)
- **ExecutionResult**: `output{}, retrieved[], latency_ms, input/output_tokens, cost_usd, trace_id`
- **EvaluationResult**: `evaluator, evaluator_version, score(0..1|null), passed(bool|null), label, reason, metadata`.
  `score=None + label="not_applicable"` = bu case için geçerli değil (ortalamaya girmez).
  Hata durumunda `score=0, passed=False, label="error"` (muhafazakar).

### Evaluator'lar
| Ad | Tür | Parametreler |
|---|---|---|
| exact_match | det. | output_key, expected_key, case_sensitive |
| required_fields | det. | fields |
| json_schema | det. | schema |
| retrieval_recall_at_k / precision_at_k | det. | k, min_score; beklenen: `expected_output.relevant_doc_ids` |
| expected_document_present | det. | source; beklenen: `expected_output.expected_documents` |
| citation_exists | det. | citations_key |
| latency_threshold / cost_threshold / token_threshold | det. | max_ms / max_usd / max_tokens |
| answer_correctness, groundedness, citation_correctness | LLM (ATA RAG) | pass_threshold, *_key |
| decision_correctness, explanation_quality, hallucination_detection | LLM (Internship) | pass_threshold, *_key |

**Yeni evaluator eklemek** (runner'a dokunmadan):
```python
from app.evaluators import BaseEvaluator, register_evaluator

@register_evaluator("my_check")
class MyCheck(BaseEvaluator):
    def evaluate(self, case, execution):
        ok = "x" in execution.output.get("answer", "")
        return self.make_result(score=float(ok), passed=ok, reason="...")
```
Dosyayı `app/evaluators/deterministic/` altına koyup `__init__.py`'ye import ekleyin, sonra YAML'da adını yazın.

**LLM Judge**: tek generic `LLMJudgeEvaluator`; her metrik `llm_judge/prompts.py` içinde bir prompt şablonu (versiyonlu).
Judge JSON `{"score","reason"}` döner. Sonuç metadata'sında `judge_model`, `prompt_name`, `prompt_version` kaydedilir.
Provider: OpenAI uyumlu herhangi bir endpoint (`JUDGE_API_KEY/BASE_URL/MODEL`). Testlerde `FakeLLM` ile mock'lanır.
**Gizlilik:** judge'a case input/output ve retrieved context gönderilir. Sentetik/anonim veri kullanın.

### Config (YAML)
`examples/ata_rag.yaml`, `examples/internship.yaml`. `thresholds` = evaluator ortalaması için minimum; altında kalırsa `passed=false`.
`classification` bloğu ikili karar sistemlerinde confusion matrix + accuracy/precision/recall/F1 üretir.

### API
| | |
|---|---|
| POST /evaluations/run | `{config, items:[{case, execution?}], meta}`; `execution` yoksa kayıtlı sistemin `endpoint_url`'i çağrılır |
| GET /evaluations, /evaluations/{id}?only_failed=true | run'lar ve case bazlı sonuçlar |
| GET/POST /systems, GET /evaluators | sistem kaydı, kayıtlı evaluator listesi |
| GET /metrics?system=&run_id= | evaluator başına ortalama skor / pass rate |

### Langfuse
Motor sadece `ScoreSink` protokolünü bilir. `LANGFUSE_PUBLIC_KEY/SECRET_KEY` varsa `LangfuseSink`, `execution.trace_id`'ye her skoru yazar.
Dataset/experiment entegrasyonu bu sink'e (veya yeni bir sink'e) eklenir.

### Bilinen sınırlar (MVP)
- Run senkron çalışır (büyük dataset için background job gerekir).
- Tablolar `create_all` ile oluşur; Alembic migration sonraya.
- Precision@K, dönen doküman sayısına böler (k'dan az dönerse cezalandırmaz).

---

# Langfuse, Dataset ve Experiment yönetimi

## Akış
```
POST /datasets                    -> ata-rag-golden-v1 (immutable)
POST /datasets/{n}/versions       -> değişiklik = yeni versiyon (v2, v3 ...)
POST /experiments                 -> dataset versiyonunu çalıştır + Langfuse'a yaz + gate'i değerlendir
POST /experiments/compare         -> baseline vs candidate, PASS/FAIL
python -m app.cli gate ...        -> CI: exit 0 = PASS, 1 = FAIL
```

## Dataset versiyonlama
- Dataset adı `ata-rag-golden`, versiyonlar `ata-rag-golden-v1, -v2 ...` (ad `-vN` ile bitemez; suffix otomatik eklenir).
- **Immutable:** versiyon oluşunca içeriği değiştirilemez/silinemez. Bu sadece konvansiyon değil, ORM seviyesinde
  engelli (`ImmutableDatasetError`). Tek istisna: `langfuse_synced_at`.
- Değişiklik = yeni versiyon: `add_cases` / `update_cases` / `remove_case_ids` (base versiyondan türetir) veya tam `cases` listesi.
- Case ID'leri versiyonlar arasında **sabit** kalır. Her versiyonun `content_hash` (SHA-256) değeri vardır,
  experiment bu hash'i kaydeder → hangi içerikle koşulduğu kanıtlanabilir.
- Langfuse'ta dataset adı `ata-rag-golden-v1`, item id'si `ata-rag-golden-v1:rag-001` (Langfuse'ta item id'ler proje genelinde tekildir).
- **Production → dataset:** `POST /datasets/{n}/versions/from-trace` Langfuse trace'ini okuyup insanın verdiği
  `expected_output` ile yeni bir case ekler (yeni versiyon).

## Experiment
Bir experiment = bir sistem versiyonu × bir dataset versiyonu. Kaydedilenler: dataset ref + hash, `meta`
(app_version, model, prompt_version, retriever, ...), evaluator versiyonları, config, aggregate skorlar,
operasyonel ortalamalar (latency, cost, token), gate sonucu, zaman damgası. Numara sistem başına artar
(`ata-rag #1, #2 ...`), Langfuse'ta dataset run adı olarak kullanılır.

Bir sistem için bir **baseline** experiment vardır (`set_baseline: true` veya `POST /experiments/{id}/baseline`).
Yeni experiment otomatik olarak baseline'a karşı değerlendirilir.

## Langfuse'ta ne oluşur
| Nesne | Nasıl |
|---|---|
| Dataset + items | `POST /api/public/v2/datasets`, `/dataset-items` (versiyon başına) |
| Trace | Sistem kendi `trace_id`'sini verirse o kullanılır; vermezse platform trace (+ retrieval span, generation) oluşturur |
| Experiment (dataset run) | `POST /api/public/dataset-run-items` (run adı = experiment adı) |
| Scores | Her evaluator sonucu trace'e skor: sayısal (score) veya kategorik (TP/FN/error) |
Langfuse yapılandırılmamışsa (env yoksa) platform çalışmaya devam eder. Langfuse hatası değerlendirmeyi düşürmez, `experiment.langfuse` alanına yazılır.
Not: REST kullanıldı (SDK değil); sürümden bağımsız ve mock ile test edilebilir.

## Regression / quality gates (YAML)
`examples/gates.yaml`:
```yaml
gates:
  minimum:            {answer_correctness: 0.90, groundedness: 0.90}   # mutlak alt sınır
  max_drop:           {default: 0.02}                                  # baseline'a göre max düşüş (puan)
  max_increase_pct:   {latency_ms: 25, cost_usd: 25}                   # düşük-iyi metrikler, % artış
  max_newly_failing_cases: 3                                           # baseline'da geçip şimdi kalan case sayısı
```
- Gate'ler experiment config'ine (`config.gates`) konur ya da `/experiments/compare` isteğinde verilir.
- Baseline yoksa `minimum` çalışır, regresyon kontrolleri `skipped` olur.
- Her kontrol `pass / fail / skipped` + okunabilir mesaj döner; herhangi biri `fail` ise gate FAIL.
- Karşılaştırma çıktısı: metrik bazında `improved / regressed / unchanged` (eşik 0.1 puan), operasyonel metrikler,
  `newly_failing` (trace linkiyle), `fixed`, `still_failing`. Farklı dataset versiyonları ortak case'ler üzerinden kıyaslanır (`dataset_mismatch` uyarısı).

## CI kullanımı (3. teslimat için hazır)
```bash
python -m app.cli gate --api $EVAL_API_URL --candidate $EXPERIMENT_ID --gates gates.yaml
```
Çıktı örneği:
```
FAIL: AI Evaluation failed
  [FAIL ] answer_correctness: 92.4% -> 86.7% (-5.7 pts), max allowed drop 2.0 pts
Deployment should be blocked.
```
