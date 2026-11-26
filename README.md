# Reconhecimento facial com ArcFace

Pipeline de **ArcFace** para o TCC: cadastro 1:N, dataset LFW, avaliação e um adaptador leve em cima do `buffalo_l` congelado. CosFace e AuraFace existem só como placeholder na factory; a implementação real nesta etapa é ArcFace.

Hardware de referência do grupo: Ryzen 5700G + RTX 4060 (o adaptador usa CUDA se estiver disponível, senão CPU).

## O que é ArcFace

ArcFace (Additive Angular Margin Loss) mapeia cada rosto para um vetor numa hiperesfera e separa identidades pelo ângulo. Artigo: [ArcFace (CVPR 2019)](https://openaccess.thecvf.com/content_CVPR_2019/html/Deng_J_ArcFace_Additive_Angular_Margin_Loss_for_Deep_Face_Recognition_CVPR_2019_paper.html).

Neste projeto o backbone já vem treinado (`buffalo_l` / InsightFace). Nós não retreinamos o iResNet: o ONNX fica congelado.

## ROI, detecção e embedding

Fluxo por imagem:

1. Detector do InsightFace acha a face (bbox = ROI)
2. Alinhamento pelos landmarks
3. Embedding ArcFace 512-D, normalizado em L2

Cadastro, identificação e avaliação usam sempre essa face alinhada, não a foto inteira “crua”.

## Interface multi-modelo

`FaceEmbedder` é o contrato comum (`extract(image) -> list[FaceDetection]`). A factory `get_embedder(--model, --adapter)` devolve:

| Modelo | Status |
|--------|--------|
| `arcface` | Implementado (`buffalo_l`) |
| `arcface` + `--adapter caminho.pt` | ArcFace + `LinearAdapter` |
| `cosface` / `auraface` | Stub (`NotImplementedError`) |

`enroll` e `identify` aceitam `--model` e `--adapter` e continuam funcionando com galeria local.

## Adaptador leve

O `buffalo_l` é ONNX: não dá para fazer fine-tuning completo do backbone aqui. Em vez disso:

1. Extrai embeddings congelados no split de treino do LFW
2. Treina `LinearAdapter` (linear 512→512 + L2) com **cross-entropy** nas identidades
3. Na inferência, o mesmo detector/alinhamento roda e o vetor passa pela projeção

Checkpoint padrão: `results/adapters/arcface_linear.pt` (gitignored). O backbone já é ArcFace; o adaptador só especializa o espaço ao subconjunto LFW de treino.

## Dataset LFW

Usamos o [LFW](http://vis-www.cs.umass.edu/lfw/) (Labeled Faces in the Wild) para avaliação acadêmica. As imagens **não** entram no Git.

`prepare-lfw` baixa o LFW (primeiro pelo mirror Figshare do scikit-learn; o site da UMass costuma falhar em DNS), monta `data/lfw/raw/`, gera `data/lfw/splits.json` (train/val/test sem leak de identidade) e um layout `gallery/` + `query/` do split de teste. Identidades com poucas fotos entram como open-set (só query).

Se o download automático falhar:

```bash
# Baixe https://ndownloader.figshare.com/files/5976018 e salve como data/lfw/lfw.tgz
python -m src.main prepare-lfw
# ou
python -m src.main prepare-lfw --archive caminho/lfw.tgz
```

Licença/uso: o LFW é distribuição acadêmica de faces publicamente disponíveis; use só para pesquisa e não versione o dataset.

## Instalação

Python 3.10+. No diretório do projeto:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Dependências principais: InsightFace, ONNX Runtime, OpenCV, NumPy, Matplotlib, PyTorch, scikit-learn, tqdm, Pytest. GPU é opcional (CUDA do PyTorch na máquina com RTX 4060).

## Galeria local (enroll / identify)

```text
data/gallery/joao/foto1.jpg
data/gallery/maria/foto1.jpg
```

```bash
python -m src.main enroll --gallery data/gallery --output results/gallery.npz
python -m src.main identify --image caminho/consulta.jpg --index results/gallery.npz
```

Com modelo/adaptador:

```bash
python -m src.main enroll --model arcface --adapter none --gallery data/gallery
python -m src.main identify --model arcface --adapter results/adapters/arcface_linear.pt --image consulta.jpg --index results/gallery.npz
```

Equivalente: `python main.py ...`.

## LFW, treino do adaptador e avaliação

```bash
python -m src.main prepare-lfw
python -m src.main train-adapter --device cuda
python -m src.main evaluate --model arcface --adapter none
python -m src.main evaluate --model arcface --adapter results/adapters/arcface_linear.pt
```

O `evaluate` calibra o threshold na **validação** (maximiza F1) e fixa esse valor no **teste**. Saídas (sem embeddings):

- `results/metrics.csv` (resumo acumulado baseline vs adapter)
- `results/evaluation_summary.json`
- `results/evaluation_detail.csv` (por query)

Métricas: identification rate, closed-set accuracy, open-set rejection, rank-1, precision/recall/F1, matriz TP/FP/TN/FN, latência média de extract e de identify 1:N.

Se a máquina não tiver rede no CI, `prepare-lfw` não roda lá: os testes unitários usam pastas temporárias e mocks, sem baixar LFW nem `buffalo_l`.

## Como testar

```bash
pip install -r requirements.txt
pytest
```

Cobertura: factory/protocolo, adaptador 512→512 + L2, métricas sintéticas, prepare-lfw em filesystem fake, train/evaluate com extractor mockado, enroll/identify.

## Limitações

- Só ArcFace está implementado; CosFace/AuraFace ficam para o grupo
- Adaptador leve, não treino full iresnet50
- LFW + subconjunto: precisa de download local (não versionado)
- Threshold do `identify` manual continua podendo ser passado na mão; no `evaluate` ele vem da val
- Qualidade depende da detecção/alinhamento do `buffalo_l`
- Índice `results/gallery.npz` e checkpoints do adaptador não devem ir para o GitHub

## Referências

- Deng, J., Guo, J., Xue, N., & Zafeiriou, S. (2019). ArcFace: Additive Angular Margin Loss for Deep Face Recognition. *CVPR*.
- InsightFace: https://github.com/deepinsight/insightface
- LFW: http://vis-www.cs.umass.edu/lfw/
