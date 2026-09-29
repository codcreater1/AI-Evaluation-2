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

---

# Dashboard (localhost)

`uvicorn app.api.main:app` sonrası tarayıcıda **http://localhost:8000** açılır.

- **System overview:** her sistemin son experiment'i, skorlar vs eşik (yeşil/kırmızı), gate PASS/FAIL
- **Experiment history:** run adı, sistem, dataset versiyonu, model, prompt, case sayısı, pass rate'ler, baseline (★)
- **Experiment sayfası:** skorlar, latency/cost/token, baseline ile karşılaştırma, gate kontrolleri,
  başarısız case'ler ve her biri için **"Open in Langfuse"** trace linki
- **Langfuse bölümü:** Langfuse'tan canlı okunan bağlantı durumu, son skorlar ve dataset listesi
- **"Run test scenario" butonu:** demo dataset + 3 experiment (baseline, zararsız değişiklik, bozuk değişiklik) oluşturur
  ve Langfuse ayarlıysa hepsini oraya gönderir. Sunum / test için tek tıkla veri üretir.

Langfuse okuma API'leri sürümler arasında değişti (yeni platformda skorlar için `v3/scores`); istemci önce yenisini
dener, olmazsa eskisine düşer. Okuma başarısız olursa panel geri kalanı çalışmaya devam eder, hata mesajını gösterir.
Yeni gönderilen verinin Langfuse'ta görünmesi birkaç dakika sürebilir.

## Langfuse Cloud limitleri (önemli)
- **Ücretsiz (Hobby) plan:** "genel API" dakikada ~30 istek. Skorları tek tek `POST /api/public/scores` ile göndermek bu
  limiti aşıyordu (ilk ~30 skordan sonrası düşüyordu). Şimdi **trace'ler ve skorlar toplu (ingestion batch)** gönderilir;
  yalnız dataset-run bağlantısı (case başına 1 küçük istek) genel API'yi kullanır.
- **429 (rate limit):** istemci `Retry-After` kadar bekleyip yeniden dener. Panel okumaları beklemez, önbelleğe alınır ve
  anlaşılır bir mesaj gösterir.
- **Hatalar gizlenmez:** Langfuse'a gönderirken hata olursa experiment kaydında `langfuse.errors` / `last_error`
  tutulur, panelde kırmızı uyarı çıkar.
- Test senaryosu bilerek küçük (6 case) tutuldu; iki kez üst üste çalıştırmadan önce ~1 dakika bekleyin.
- **Trace gecikmesi:** Langfuse v4'te bu yolla gönderilen trace'ler arayüzde birkaç dakika (en fazla ~15 dk) gecikmeli görünebilir.
- **Bilinen risk:** Langfuse, eski *trace/observation ingestion API*'sini **16 Kasım 2026'da Cloud'da kapatacak**
  (skor eventleri devam ediyor). Bugün çalışıyor; kalıcı çözüm trace'leri OpenTelemetry (OTLP) endpoint'ine taşımak.

---

# Golden Datasets, Human Evaluation & Judge Agreement

## Golden datasets
`datasets/ata_rag_golden.json` (104 cases) and `datasets/internship_golden.json` (54 cases) — see
`datasets/README.md` for methodology and provenance. In short: the ATA RAG set was built by asking
the live `pomelo-9` assistant real questions (English, Polish, Ukrainian); the Internship Coordinator
set was built from the live `pomelo-3`/`pomelo-2` system's real (read-only) decisions plus the
thresholds it publishes at `university-rules.json`. Both were extended with paraphrase / boundary /
robustness cases and tagged via `metadata.source_note`. Load them into the platform with:
```bash
python -m scripts.load_golden_datasets --api http://localhost:8000
```
This matches the `ata-rag-golden-v1` / `internship-golden-v1` names that `examples/ata_rag.yaml` and
`examples/internship.yaml` already expect.

## Human evaluation
`POST /human-evaluations` records a human verdict (`score`, `passed`, `reason`, `reviewer`) for one
case + evaluator; it automatically matches the `trace_id` from the matching `EvaluationResultRow` in
the same run and pushes it to Langfuse as a `human_<evaluator>` score (best-effort, never fails the
evaluation). `GET /human-evaluations/queue` lists LLM-judge results that don't have a human review
yet; `GET /human-evaluations/agreement` produces a report over the matched (judge, human) pairs.
Dashboard: **Human evaluation** (review queue with an inline form) and **Judge vs human agreement**
pages (`/dashboard/human-eval`, `/dashboard/agreement`).

## LLM judge vs. human agreement
`app/human_eval/agreement.py`: pass/fail agreement rate, **Cohen's kappa** (chance-corrected), mean
`|score diff|`, a confusion matrix (both_pass/both_fail/human_pass_judge_fail/human_fail_judge_pass),
and the biggest disagreements. To validate on >=100 cases without a paid `JUDGE_API_KEY`, the
dashboard's **"Run judge-vs-human demo"** button (or `POST /dashboard/human-eval/run-demo`) seeds a
120-case synthetic-but-realistically-noisy judge+human set (`app/human_eval/demo.py`) so the report
can be exercised end to end.
