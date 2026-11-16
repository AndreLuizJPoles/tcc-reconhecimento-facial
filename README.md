# Reconhecimento facial com ArcFace

Implementação do pipeline de **ArcFace** para o TCC de reconhecimento facial. O código usa o modelo pré-treinado `buffalo_l` do InsightFace, em CPU, só para inferência. O fluxo é **identificação 1:N**: uma imagem de consulta é comparada com uma galeria de identidades e pode resultar em match ou em **não encontrado**.

Não há treino, fine-tuning nem outros métodos (CosFace, FaceNet, RF-DETR).

## O que é ArcFace

ArcFace (Additive Angular Margin Loss) é uma função de perda para reconhecimento facial proposta por Deng, Guo, Xue e Zafeiriou (CVPR 2019). A ideia é mapear cada rosto para um vetor em uma hiperesfera e separar identidades pelo ângulo entre esses vetores.

O artigo original está em: [ArcFace: Additive Angular Margin Loss for Deep Face Recognition](https://openaccess.thecvf.com/content_CVPR_2019/html/Deng_ArcFace_Additive_Angular_Margin_Loss_for_Deep_Face_Recognition_CVPR_2019_paper.html).

## Embeddings faciais

Um embedding facial é um vetor numérico de dimensão fixa (no `buffalo_l`, 512) que resume a identidade da face depois da detecção e do alinhamento. Faces da mesma pessoa tendem a ficar próximas nesse espaço. Faces de pessoas diferentes tendem a ficar mais afastadas. Neste projeto os vetores são normalizados em L2 antes da comparação.

## ArcFace Loss

A perda parte do cosseno entre o embedding e o peso da classe, interpreta esse valor como um ângulo e soma uma margem angular `m` no ângulo da classe correta. Em seguida aplica uma escala `s`. Isso aperta a decisão no espaço angular e costuma melhorar a separação entre identidades em relação a perdas softmax comuns. Esta entrega **não treina** essa perda: só usa um modelo já treinado com ela.

## Similaridade de cosseno e busca 1:N

A comparação entre dois embeddings `a` e `b` é:

`cos(a, b) = (a · b) / (||a|| ||b||)`

Na identificação, a consulta é comparada com **todas** as fotos da galeria. Para cada identidade, fica o **maior** score entre as fotos dela. Se o melhor score for `>= threshold`, o sistema devolve essa identidade. Caso contrário, o resultado é **não encontrado** (cenário open-set: a pessoa pode não estar na base).

O limiar padrão `0.5` é provisório. Ele precisa ser calibrado depois com uma base de validação.

## Modelo pré-treinado `buffalo_l`

O pack [buffalo_l](https://github.com/deepinsight/insightface/tree/master/python-package#model-zoo) do [InsightFace](https://github.com/deepinsight/insightface) inclui detector (SCRFD) e reconhecedor ArcFace (ResNet-50) em ONNX. O pipeline do pacote Python detecta o rosto, alinha pelos pontos faciais e devolve o embedding.

Na primeira execução o InsightFace baixa os pesos para o cache local. Isso exige internet. Os arquivos de modelo não devem ir para o GitHub.

## Instalação

Python 3.10 ou superior. No diretório do projeto:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Dependências: InsightFace, ONNX Runtime (CPU), OpenCV, NumPy, Matplotlib e Pytest.

## Como montar a galeria

Organize as fotos por identidade:

```text
data/gallery/joao/foto1.jpg
data/gallery/joao/foto2.jpg
data/gallery/maria/foto1.jpg
```

Regras do cadastro:

- Nome da pasta = identidade
- Preferir uma face por foto; se a imagem tiver vários rostos, ela é ignorada
- Imagem sem rosto: aviso e pulo
- Use só imagens com autorização; não versione fotos pessoais

## Como executar

### 1. Cadastrar a galeria

```bash
python -m src.main enroll --gallery data/gallery --output results/gallery.npz
```

### 2. Identificar uma imagem

Com o índice já gerado:

```bash
python -m src.main identify --image caminho/consulta.jpg --index results/gallery.npz
```

Ou montando a galeria na hora:

```bash
python -m src.main identify --image caminho/consulta.jpg --gallery data/gallery
```

Se a consulta tiver mais de um rosto:

```bash
python -m src.main identify --image caminho/consulta.jpg --index results/gallery.npz --face-index 0
```

Opções úteis:

- `--threshold 0.5` limiar provisório de cosseno
- `--top-k 5` candidatos no ranking do JSON
- `--output-json results/identification.json`
- `--output-figure results/identification.png`

Equivalente pela raiz: `python main.py enroll ...` / `python main.py identify ...`.

## Como interpretar o resultado

O terminal mostra quantos rostos foram encontrados na consulta, a melhor similaridade e se houve match. O JSON traz:

- `matched` (`true` / `false`)
- `identity` (string ou `null` se não encontrado)
- `similarity` da melhor candidata
- `threshold`
- `best_gallery_image`
- `candidates` (ranking com identidade e similaridade, sem embeddings)

`matched = false` é resultado válido: a pessoa pode não estar na base, ou a similaridade ficou abaixo do limiar.

A figura mostra a consulta. Em caso de match, também mostra a melhor foto da galeria.

## Como testar

Os testes unitários mockam o InsightFace e **não** baixam `buffalo_l`:

```bash
pip install -r requirements.txt
pytest
```

Cobertura principal: cosseno, threshold, galeria por pastas, ranking 1:N, rejeição abaixo do limiar, imagem sem rostos, enroll/identify via CLI.

Para um teste real com o modelo, cadastre fotos locais em `data/gallery` e rode `enroll` + `identify`. Na primeira vez o pack `buffalo_l` será baixado.

## Limitações desta etapa

- Threshold `0.5` não foi calibrado; não há métricas experimentais neste README.
- Sem treino e sem fine-tuning.
- Execução inicial em CPU.
- A qualidade do embedding depende da detecção e do alinhamento do `buffalo_l`.
- Fotos da galeria com vários rostos são ignoradas no cadastro.
- O índice `results/gallery.npz` contém embeddings e não deve ir para o GitHub.

O arquivo `src/finetuning.py` existe só como reserva para uma etapa futura.

## Referências

- Deng, J., Guo, J., Xue, N., & Zafeiriou, S. (2019). ArcFace: Additive Angular Margin Loss for Deep Face Recognition. *CVPR*. https://openaccess.thecvf.com/content_CVPR_2019/html/Deng_ArcFace_Additive_Angular_Margin_Loss_for_Deep_Face_Recognition_CVPR_2019_paper.html
- InsightFace (código e pacote Python): https://github.com/deepinsight/insightface
- Model Zoo / `buffalo_l`: https://github.com/deepinsight/insightface/tree/master/python-package
