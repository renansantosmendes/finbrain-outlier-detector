# Outlier Detector (API de ML com W&B e Vercel)

API para **treino e serving** de um detector de dias atípicos no preço de
ações. O modelo é um **Isolation Forest** (scikit-learn), treinado com retornos,
volume e retorno relativo ao Ibovespa, e com a flag de dias de divulgação de
resultado. O treino roda no **GitHub Actions** e registra cada modelo como
artefato no **Weights & Biases (W&B)**. A API roda no **Vercel**, carrega os
modelos do W&B para a memória quando inicia e busca os dados de mercado a
cada requisição.

## Estrutura

```
api/index.py                          entry point do Vercel (expõe o app FastAPI)
vercel.json                           função Python e rewrite de todas as rotas
requirements.txt                      dependências de execução (gerado do uv.lock)
pyproject.toml, uv.lock               dependências do projeto (uv); extras dev e notebook
notebooks/outlier_detection_training.ipynb  notebook didático com funções (versão anterior)
notebooks/outlier_detection_simple.ipynb    notebook simples do treino, para a aula
src/outlier_detector/
  data/market_data.py                 preços, Ibovespa e datas de resultado (yfinance)
  features/price_features.py          as 7 features do modelo
  models/isolation_forest_detector.py detector (fit, score_days, save, load)
  models/walk_forward.py              scores walk-forward sem olhar o futuro
  tracking/wandb_tracker.py           run, métricas e registro do modelo no W&B
  registry/wandb_model_store.py       nome do artefato, upload e download do modelo
  training/train.py                   comando de treino (CLI), uma run por ticker
  serving/model_registry.py           modelos em memória, carregados do W&B
  serving/service.py                  busca os dados e pontua os últimos dias
  serving/schemas.py, api.py          contrato HTTP e rotas FastAPI
  serving/vercel.py                   ajuste do caminho das rotas no Vercel
tests/outlier_detector/               testes espelhando src/ (sem rede)
.github/workflows/                    CI, treino e deploy (três workflows separados)
```

## Como o modelo é treinado

Cada ticker é uma run do W&B. O treino segue o notebook simples:

- **Histórico:** desde `--start` (padrão `2020-01-01`).
- **Features:** `ret`, `ret_abs`, `ret_lag1`, `vol_20`, `vol_rel`, `ret_ab`
  (retorno menos o Ibovespa) e `is_earn` (dia de resultado).
- **Walk-forward:** primeiro treino com 756 dias (`TRAIN_MIN`), reajuste a cada
  126 dias (`STEP`), janela expansiva, sem padronização.
- **Modelo final:** ajustado com todas as linhas e registrado como artefato
  `outlier-detector-<ticker>` no W&B, com a alias `latest`.
- **Métricas da run:** linhas de treino, linhas avaliadas no walk-forward e a
  fração de dias marcados como atípicos.

## Como rodar localmente

Pré-requisito: [uv](https://docs.astral.sh/uv/) e uma chave do W&B.

```bash
uv sync --extra dev                     # dependências e ferramentas de teste
uv sync --extra dev --extra notebook    # também matplotlib e ipykernel, para o notebook

cp .env.example .env                    # depois preencha WANDB_API_KEY no .env
uv run pytest                           # testes unitários (sem rede)
uv run pylint src                       # lint do pacote

# treina e registra no W&B (uma run por ticker)
uv run python -m outlier_detector.training.train --tickers AAPL,PETR4.SA

# sobe a API em http://127.0.0.1:8000, carregando AAPL do W&B
OUTLIER_TICKERS=AAPL uv run uvicorn api.index:app --reload
```

## API

| Método | Rota | Descrição |
|---|---|---|
| GET | `/health` | Status e quantidade de modelos em memória |
| GET | `/models` | Tickers cujos modelos estão em memória |
| POST | `/predict` | Pontua os últimos dias de um ticker |

O `/predict` recebe só o ticker e o número de dias. A API baixa os preços, o
Ibovespa e as datas de resultado no yfinance:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL", "last_n_days": 5}'
```

Resposta: `days`, com `score` (quanto maior, mais atípico) e `is_atypical`.
Códigos: `404` quando o ticker não está em memória, `422` quando os dados de
mercado não geram features. A documentação interativa fica em `/docs`.

### Carregamento em memória

Ao criar a aplicação, a API baixa do W&B o último modelo de cada ticker listado
em `OUTLIER_TICKERS` e guarda em memória. Um ticker que falha é registrado no
log e deixado de fora, e a API continua no ar. Sem `OUTLIER_TICKERS`, nenhum
modelo é baixado.

## Workflows

| Arquivo | Quando roda | O que faz |
|---|---|---|
| `ci.yml` (01) | Push e pull request | Estrutura, `requirements.txt` x `uv.lock`, pylint, testes com cobertura mínima de 80%, smoke test da API, Bandit |
| `train.yml` (02) | Manual | Espera a CI do commit, treina os tickers e registra os modelos no W&B |
| `deploy.yml` (03) | Manual | Recebe o `train_run_id` de um treino bem-sucedido e publica a API no Vercel |

Os três são independentes. O deploy não copia modelos: a API os baixa do W&B
quando inicia.

### Passo a passo

1. Rode **02 - Train Model** e anote o número da execução (`Run ID`).
2. Rode **03 - Deploy API to Vercel** com esse número em `train_run_id`.

### Secrets e variáveis

No GitHub (*Settings → Secrets and variables → Actions*):

| Nome | Tipo | Uso |
|---|---|---|
| `WANDB_API_KEY` | Secret | Treino registra os modelos no W&B |
| `VERCEL_TOKEN` | Secret | Deploy na Vercel |
| `VERCEL_ORG_ID` | Secret | `orgId` em `.vercel/project.json` (depois de `vercel link`) |
| `VERCEL_PROJECT_ID` | Secret | `projectId` em `.vercel/project.json` |

Na Vercel (*Project Settings → Environment Variables*), a API lê:

| Nome | Obrigatória | Uso |
|---|---|---|
| `WANDB_API_KEY` | Sim | Baixa os modelos do W&B na inicialização |
| `OUTLIER_TICKERS` | Sim | Tickers carregados em memória, por exemplo `AAPL,PETR4.SA` |
| `WANDB_ENTITY` | Não | Usuário ou time do W&B. Padrão: conta da chave |
| `OUTLIER_WANDB_PROJECT` | Não | Projeto do W&B. Padrão: `outlier-detector` |

Nunca coloque a chave no código: ela entra só no `.env` local, no secret do
GitHub e na variável da Vercel.

## Atualizar requirements.txt

O `requirements.txt` é o que o Vercel instala. Depois de mudar dependências:

```bash
uv lock
uv export --no-dev --no-hashes --no-emit-project --frozen -o requirements.txt
```

A CI falha se o arquivo não bater com o `uv.lock`.

## Observações

- O `/predict` pontua dias com o modelo final, que foi treinado com todo o
  histórico. Para uma avaliação honesta, use o walk-forward do treino, que o
  notebook mostra.
- Diferente do notebook, o pacote não ignora features quando o Ibovespa ou as
  datas de resultado faltam: o treino e a API param com erro, para que o modelo
  nunca seja usado com features diferentes das do treino.
- Os testes não acessam a rede nem o W&B: yfinance e wandb são substituídos.
