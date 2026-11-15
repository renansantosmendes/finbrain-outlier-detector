# Outlier Detector (esqueleto de API de ML)

Esqueleto de uma API para **treino e serving** de um modelo de detecção de
dias atípicos em séries de preços. O modelo é um **Isolation Forest**
(scikit-learn) que aprende o padrão de cada ativo e marca os dias que fogem
desse padrão. A API roda no **Vercel**, e o treino roda no **GitHub Actions**.

O objetivo é servir de ponto de partida para a aula: o núcleo de detecção
(features, walk-forward e detector) foi portado do notebook de referência e
organizado em módulos com testes.

## Estrutura

```
api/index.py                       entry point do Vercel (expõe o app FastAPI)
vercel.json                        função Python e rewrite de todas as rotas
requirements.txt                   dependências de execução (gerado do uv.lock)
pyproject.toml, uv.lock            dependências do projeto (uv)
models/                            modelos treinados (gerado; fora do git)
src/outlier_detector/
  data/market_data.py              download de preços com yfinance
  features/indicators.py           RSI e MACD
  features/price_features.py       features usadas pelo modelo
  models/isolation_forest_detector.py  detector (fit, scores, save, load)
  models/walk_forward.py           scores sem olhar o futuro
  models/artifact_store.py         salva e carrega modelos por ticker
  training/train.py                comando de treino (CLI)
  serving/schemas.py               corpo das requisições e respostas
  serving/service.py               pontuação dos dias de uma requisição
  serving/api.py                   rotas FastAPI
  serving/vercel.py                ajuste do caminho das rotas no Vercel
tests/outlier_detector/            testes espelhando src/
.github/workflows/                 CI, treino e deploy (três workflows separados)
```

## Como rodar localmente

Pré-requisito: [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev                 # instala dependências e ferramentas de teste
uv run pytest                       # testes unitários
uv run pylint src                   # lint do pacote

# treina AAPL e PETR4.SA, salvando em models/ (baixa preços do Yahoo Finance)
uv run python -m outlier_detector.training.train --tickers AAPL,PETR4.SA

# sobe a API local em http://127.0.0.1:8000
uv run uvicorn api.index:app --reload
```

Variável opcional `OUTLIER_MODEL_DIRECTORY`: pasta com os modelos (padrão:
`models/` na raiz do projeto). Ver `.env.example`.

## API

| Método | Rota | Descrição |
|---|---|---|
| GET | `/health` | Confirma que o serviço está no ar |
| GET | `/models` | Lista os tickers com modelo treinado |
| POST | `/predict` | Pontua os últimos dias de uma lista de preços |

Exemplo de `/predict`. São necessários pelo menos 60 dias de preços, em
ordem crescente de data:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
        "ticker": "AAPL",
        "last_n_days": 3,
        "prices": [
          {"day": "2026-01-02", "close": 250.1, "volume": 45000000},
          {"day": "2026-01-05", "close": 251.3, "volume": 43000000}
        ]
      }'
```

Resposta: a lista `days`, com `score` (negativo quer dizer atípico) e
`is_atypical`. Códigos: `404` para ticker sem modelo, `422` para preços
inválidos.

A documentação interativa fica em `/docs`.

## Workflows

| Arquivo | Quando roda | O que faz |
|---|---|---|
| `ci.yml` (01) | Push e pull request | Estrutura, `requirements.txt` x `uv.lock`, pylint, testes, smoke test da API, Bandit |
| `train.yml` (02) | Manual | Espera a CI do commit, treina os tickers e publica o artefato `trained-models` |
| `deploy.yml` (03) | Manual | Recebe o `train_run_id`, baixa os modelos, publica no Vercel e testa `/health` e `/models` |

Os três são independentes: a CI roda sozinha, o treino espera a CI do
commit, e o deploy só aceita um treino bem-sucedido.

### Passo a passo do deploy

1. Rode o workflow **02 - Train Model** e anote o número da execução (`Run ID`).
2. Rode o workflow **03 - Deploy API to Vercel** com esse número em `train_run_id`.

### Secrets e variáveis do GitHub

Configure em *Settings → Secrets and variables → Actions*:

| Nome | Tipo | Onde obter |
|---|---|---|
| `VERCEL_TOKEN` | Secret | Vercel → Account Settings → Tokens |
| `VERCEL_ORG_ID` | Secret | `.vercel/project.json` depois de `vercel link` (campo `orgId`) |
| `VERCEL_PROJECT_ID` | Secret | `.vercel/project.json` depois de `vercel link` (campo `projectId`) |
| `DEPLOY_ENVIRONMENT` | Variable (opcional) | Nome do ambiente de deploy. Padrão: `production` |

Para criar o projeto no Vercel pela primeira vez: `npx vercel link` na raiz
do repositório, e use os valores de `.vercel/project.json`. Não envie essa
pasta para o git.

## Atualizar requirements.txt

O `requirements.txt` é o que o Vercel instala. Sempre que mudar as
dependências, gere de novo a partir do `uv.lock`:

```bash
uv lock
uv export --no-dev --no-hashes --no-emit-project --frozen -o requirements.txt
```

A CI falha se o arquivo não bater com o `uv.lock`.

## Observações

- Os modelos ficam em `models/` e não entram no git. O deploy copia os
  artefatos do treino para essa pasta antes de publicar.
- A detecção usa o `IsolationForest` com `RobustScaler`. O score de
  `walk_forward_scores` usa só dados anteriores a cada dia, então a taxa de
  dias marcados é uma medida de comportamento, não de acurácia.
- Os testes não acessam a rede: o yfinance é substituído por dados
  sintéticos.
