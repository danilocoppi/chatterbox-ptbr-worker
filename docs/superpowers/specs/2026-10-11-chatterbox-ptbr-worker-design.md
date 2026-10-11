# chatterbox-ptbr-worker: especificação

Data: 11/10/2026 · Status: aguardando revisão

## Objetivo

Worker serverless de fila no RunPod que gera fala em **português do Brasil com emoção**, usando o Chatterbox Multilingual V3 com o pacote pt-BR da Resemble AI (licença MIT). Ele substitui o `profzeller/chatterbox-wellness-worker`, que só fala inglês e ignora a emoção.

Nesta fase o worker serve para **testar qualidade e velocidade na nuvem** e comparar GPUs. Ficam para depois: uso em tempo real pelo claudinei, latência e custo de manter o worker ligado.

## Decisões já tomadas

| Tema | Decisão |
|---|---|
| Idiomas | Só pt-BR, com o pacote pt-BR do V3 (`t3_pt_br.safetensors` + `s3gen_v3.pt`) |
| Pesos | Dentro da imagem (abordagem A): baixados na montagem e sem internet ao ligar |
| Publicação | Repositório privado `danilocoppi/chatterbox-ptbr-worker`, compilado pelo RunPod a partir do GitHub |
| Uso | Testes sob demanda, com mínimo de workers ativos em 0 |
| Código | Escrito do zero; o wellness-worker não declara licença e serve só de referência de estrutura e formato |

## Componentes

```
chatterbox-ptbr-worker/
├── Dockerfile          imagem: Python 3.11, Chatterbox no commit 5de7a54, pesos e voz embutidos
├── baixar_pesos.py     roda na montagem: baixa o pacote pt-BR, os arquivos base e a voz padrão
├── ptbr.py             carrega o modelo (mesmo carregador validado localmente)
├── sintese.py          lógica sem GPU: valida a entrada, resolve a emoção, divide frases, gera WAV em base64
├── handler.py          porta de entrada do RunPod: carrega o modelo ao ligar e atende as tarefas
├── tests/              testes unitários (pytest) da sintese.py
├── scripts/            teste de integração local e teste do endpoint na nuvem
└── README.md           uso, publicação no RunPod e exemplos
```

Cada arquivo tem uma função só. A `sintese.py` não importa torch, o que permite testá-la sem GPU. A `handler.py` só faz a ligação entre o RunPod, a `sintese.py` e o modelo.

## Interface

### Entrada (`input`)

Os nomes dos campos seguem os do wellness-worker, para que ele sirva de substituto direto.

| Campo | Tipo | Obrigatório | Descrição |
|---|---|---|---|
| `text` | string | sim | Texto em pt-BR, de 1 a 1.000 caracteres depois de remover a etiqueta de emoção |
| `emotion` | string | não | Emoção (tabela abaixo); o padrão é `neutro` |
| `exaggeration` | número | não | Intensidade de 0,25 a 2,0; sobrescreve o valor da emoção |
| `cfg_weight` | número | não | Ritmo e aderência de 0,0 a 1,0; sobrescreve o valor da emoção |
| `temperature` | número | não | De 0,05 a 2,0; o padrão é 0,8 |
| `seed` | inteiro | não | Semente para resultado reproduzível; sem ela, o resultado varia |
| `reference_audio_url` | string | não | URL http(s) de um áudio (WAV, MP3 ou FLAC) com a voz a imitar |
| `reference_audio_base64` | string | não | O mesmo áudio em base64, com ou sem o prefixo `data:` |
| `health_check` | booleano | não | Se `true`, responde só o estado do worker, sem gerar áudio |

- **Etiqueta no texto:** uma etiqueta no início, como `[triste] Que pena...`, define a emoção.
  - Ela só tem letras (`[A-Za-zÀ-ÿ_]+`), então um começo como `[1] Primeiro item` é falado como texto.
  - Ela é sempre removida do texto falado, mesmo quando o campo `emotion` também vem; nesse caso, o campo tem precedência.
  - Uma etiqueta de letras com nome desconhecido gera erro de emoção desconhecida.
  - Etiquetas no meio do texto não são interpretadas e são faladas como texto.
- **Voz de referência:** se vierem URL e base64 juntos, vale a URL. Sem nenhum dos dois, vale a voz padrão `pt_br_f2`, a voz feminina pt-BR da demo oficial da Resemble.
- **Campos fora da tabela** são ignorados. A única exceção é `speed` diferente de 1,0, que vira erro, porque não há suporte a mudar a velocidade.

### Emoções

A emoção define a intensidade (`exaggeration`) e o ritmo (`cfg_weight`). Qual emoção soa de fato depende sobretudo da voz de referência e do próprio texto.

| Nome | Sinônimos aceitos | exaggeration | cfg_weight |
|---|---|---|---|
| `neutro` | `neutral` | 0,5 | 0,5 |
| `alegre` | `happy` | 0,8 | 0,3 |
| `animado` | `excited` | 0,9 | 0,3 |
| `triste` | `sad` | 0,35 | 0,5 |
| `empatico` | `empático`, `empathetic` | 0,4 | 0,5 |
| `calmo` | `calm` | 0,35 | 0,5 |
| `sussurro` | `whisper` | 0,3 | 0,5 |

Os nomes não diferenciam maiúsculas de minúsculas, e os acentos são ignorados na comparação.

### Saída

```json
{
  "audio_base64": "<WAV 24 kHz mono 16 bits>",
  "sample_rate": 24000,
  "duration_seconds": 4.08,
  "text": "Que pena...",
  "emotion": "triste",
  "metrics": {
    "generation_seconds": 2.21,
    "first_sentence_seconds": 2.21,
    "rtf": 0.54,
    "sentences": 1,
    "gpu": "NVIDIA GeForce RTX 3060 Ti",
    "cold_start": false,
    "model_load_seconds": 6.5
  }
}
```

| Medida | Definição |
|---|---|
| `generation_seconds` | Tempo de síntese de todas as frases; não inclui preparar a voz de referência nem codificar o WAV |
| `first_sentence_seconds` | Tempo de síntese da primeira frase; equivale ao tempo até o primeiro áudio se o cliente tocasse frase a frase |
| `rtf` | `generation_seconds / duration_seconds`; abaixo de 1 é mais rápido que o tempo real |
| `cold_start` | `true` no primeiro pedido atendido por aquele worker desde que ligou |
| `model_load_seconds` | Tempo para carregar o modelo quando o worker ligou |

Com `health_check`, a resposta é `{"status": "healthy", "model_loaded": true, "gpu": "..."}`.

## Funcionamento

**Ao ligar o worker**, uma vez só:
1. Carrega o modelo na GPU a partir de `/models`, sem internet (`HF_HUB_OFFLINE=1`).
2. Prepara as condições da voz padrão e guarda uma cópia.
3. Faz uma geração curta de aquecimento, para que o primeiro pedido não pague a inicialização do CUDA.
4. Se qualquer passo falhar, o processo encerra com erro, e o RunPod substitui o worker.

**A cada tarefa:**
1. `health_check` responde e termina.
2. Valida a entrada e resolve a emoção e os parâmetros (`sintese.py`).
3. Resolve a voz:
   - sem voz própria, restaura a cópia da voz padrão;
   - com voz própria, baixa ou decodifica o áudio para um arquivo temporário (até 15 s de download e 10 MB), prepara as condições e apaga o arquivo.
4. Divide o texto em frases:
   - quebra depois de `.`, `!`, `?` ou `…` seguidos de espaço;
   - uma frase com mais de 300 caracteres é quebrada em `,` ou `;`, e na falta deles, entre palavras, em pedaços de até 300 caracteres.
5. Gera cada frase com os mesmos parâmetros e com a semente aplicada antes de cada uma.
6. Junta as frases com 150 ms de silêncio entre elas e codifica em WAV base64.
7. Calcula as medidas e devolve.

O RunPod atende uma tarefa por vez em cada worker, que é o padrão, então não há concorrência dentro do processo. Toda saída leva a marca d'água Perth da Resemble, que é aplicada pelo próprio Chatterbox.

## Erros

Um erro devolve `{"error": "<mensagem em português>"}`, e o RunPod marca a tarefa como falha com essa mensagem. Casos cobertos:

- texto ausente, vazio ou com mais de 1.000 caracteres;
- emoção desconhecida, com a lista de emoções aceitas na mensagem;
- número fora da faixa ou de tipo inválido;
- `speed` diferente de 1,0;
- URL que não é http(s), que excede o tempo ou o tamanho, ou que falha ao baixar;
- base64 inválido, ou com mais de 10 MB depois de decodificado;
- áudio de referência que não decodifica.

Uma exceção inesperada durante a geração vira `{"error": ...}` com a mensagem e não derruba o worker.

## Imagem

- **Base:** `python:3.11-slim`, mais `git` e `ca-certificates`.
- **Pacotes:** `chatterbox-tts` instalado do GitHub, commit `5de7a54aa4e5e2baadb0182dde554908b48b85c2`. Ele traz o PyTorch 2.6.0 com CUDA 12.4, o mesmo conjunto validado localmente. Também entram `runpod` e `soundfile`.
- **Pesos:** com `HF_HOME=/models`, a montagem roda `baixar_pesos.py`. São baixados:
  - de `ResembleAI/chatterbox`: `ve.pt`, `conds.pt`, `grapheme_mtl_merged_expanded_v1.json` e `Cangjie5_TC.json`;
  - de `ResembleAI/Chatterbox-Multilingual-pt-br`: `t3_pt_br.safetensors`, `s3gen_v3.pt` e o tokenizador;
  - a voz `pt_br_f2.wav`.
- **Ao ligar:** `HF_HUB_OFFLINE=1` e `CMD ["python", "-u", "handler.py"]`.
- **Tamanho esperado:** uns 10 GB.
- **Requisitos do endpoint:** driver com CUDA 12.4 ou superior, e uma GPU com 8 GB de memória de vídeo ou mais (o modelo usa uns 4 GB).

## Testes

1. **Unitários** (`pytest`, sem GPU), cobrindo a `sintese.py`:
   - emoção (nomes, sinônimos, acentos, etiqueta no texto, precedência do campo, sobrescrita por números);
   - validação e mensagens de erro;
   - divisão em frases (pontuação, frase longa, texto sem pontuação);
   - ida e volta do WAV em base64.
2. **Integração local:** rodar o `handler` direto no ambiente `chatterbox-tts/.venv` da RTX 3060 Ti com três tarefas:
   - neutra;
   - com etiqueta `[alegre]`;
   - com voz em base64.

   O áudio precisa decodificar, ter duração maior que zero e, transcrito pelo Parakeet local, ter CER médio abaixo de 5%.
3. **Nuvem:** depois que o RunPod compilar e o endpoint existir, enviar as mesmas tarefas com `scripts/testar_endpoint.py`, uma fria e depois três quentes, e comparar as medidas com as da 3060 Ti.

## Publicação no RunPod (passos do usuário)

1. Em **Settings → Connections**, conectar o GitHub e dar acesso ao repositório privado.
2. Em **Serverless → New Endpoint → GitHub Repo**, escolher o repositório, o branch `main` e o `Dockerfile` da raiz.
3. **GPUs:** começar com uma de 24 GB (RTX 3090, A5000 ou L4); para comparar, criar outro endpoint com uma de 48 GB (A40 ou A6000).
4. **Workers:** mínimo de ativos em 0 e máximo em 1.
5. **Disco do container:** 20 GB.
6. **Versões de CUDA permitidas:** 12.4 ou superior.
7. Esperar a compilação, que é demorada por causa dos 10 GB da imagem.
8. Passar o ID do novo endpoint e uma chave de API com acesso a ele. A chave atual só acessa o endpoint antigo.

## Fora do escopo

- Outros idiomas.
- Streaming.
- `speed`.
- Etiquetas de emoção por frase dentro de um mesmo pedido.
- Workers sempre ligados.
- Publicação no RunPod Hub (`hub.json`).
- Saída em MP3 ou Opus.
- Autenticação além da do próprio RunPod.

## Riscos

- **Imagem grande:** a compilação e o primeiro download em cada máquina são demorados, e pode haver limites de tamanho ou de tempo na compilação do RunPod.
- **Decodificador do pacote pt-BR:** o `s3gen_v3.pt` não traz dois buffers fixos do tokenizador. O carregamento usa `strict=False`, como a demo oficial, e confere que só esses dois faltam.
- **Voz `pt_br_f2`:** é a voz da demo da Resemble e serve para testes. Para uso contínuo, o ideal é uma voz própria com consentimento.
- **Medições na nuvem:** a GPU e o processador da máquina mudam os resultados, por isso a resposta informa a GPU usada.
