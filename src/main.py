"""CLI: enroll, identify, prepare-lfw, train-adapter, evaluate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.data.lfw import LFW_ROOT, load_splits, prepare_lfw
from src.evaluation import DEFAULT_THRESHOLD, FaceComparator
from src.finetuning import (
    DEFAULT_ADAPTER_PATH,
    extract_frozen_embeddings,
    train_linear_adapter,
)
from src.gallery import FaceGallery
from src.metrics import (
    append_metrics_table_row,
    calibrate_threshold,
    collect_calibration_scores,
    enroll_from_split_items,
    evaluate_identification,
    save_metrics_csv,
    save_metrics_summary,
)
from src.embedders.base import FaceEmbedder
from src.models import FaceDetection, get_embedder
from src.preprocessing import ImageLoadError, load_image
from src.visualization import save_identification_figure

RESULTS_DIR = Path("results")
DEFAULT_GALLERY_DIR = Path("data/gallery")
DEFAULT_INDEX_PATH = RESULTS_DIR / "gallery.npz"
DEFAULT_JSON_PATH = RESULTS_DIR / "identification.json"
DEFAULT_FIGURE_PATH = RESULTS_DIR / "identification.png"
DEFAULT_SPLITS_PATH = LFW_ROOT / "splits.json"
DEFAULT_METRICS_CSV = RESULTS_DIR / "metrics.csv"


def _add_model_adapter_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model",
        default="arcface",
        help="Extrator: arcface (implementado), cosface/auraface (stubs).",
    )
    parser.add_argument(
        "--adapter",
        default="none",
        help="none = buffalo_l puro; ou caminho para results/adapters/*.pt",
    )
    parser.add_argument(
        "--ctx-id",
        type=int,
        default=-1,
        help="InsightFace ctx_id (-1 CPU, 0 GPU).",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Identificação facial 1:N com ArcFace (InsightFace buffalo_l). "
            "Galeria local, LFW, adaptador leve e avaliação."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    enroll_parser = subparsers.add_parser(
        "enroll",
        help="Cadastra identidades a partir de data/gallery/<identidade>/",
    )
    enroll_parser.add_argument(
        "--gallery",
        default=str(DEFAULT_GALLERY_DIR),
        help="Pasta raiz da galeria (subpastas = identidades).",
    )
    enroll_parser.add_argument(
        "--output",
        default=str(DEFAULT_INDEX_PATH),
        help="Arquivo .npz com o índice da galeria (gitignored).",
    )
    _add_model_adapter_args(enroll_parser)

    identify_parser = subparsers.add_parser(
        "identify",
        help="Busca a identidade de uma imagem na galeria.",
    )
    identify_parser.add_argument("--image", required=True, help="Imagem de consulta.")
    identify_parser.add_argument(
        "--gallery",
        default=None,
        help="Pasta da galeria. Se omitido, use --index.",
    )
    identify_parser.add_argument(
        "--index",
        default=None,
        help="Índice .npz gerado pelo enroll. Alternativa a --gallery.",
    )
    identify_parser.add_argument(
        "--face-index",
        type=int,
        default=None,
        help="Índice do rosto na consulta quando houver mais de uma detecção.",
    )
    identify_parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Limiar de similaridade de cosseno (prefira o valor calibrado na val).",
    )
    identify_parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Quantidade de candidatos no ranking do JSON.",
    )
    identify_parser.add_argument(
        "--output-json",
        default=str(DEFAULT_JSON_PATH),
        help="Arquivo JSON de saída (sem embeddings).",
    )
    identify_parser.add_argument(
        "--output-figure",
        default=str(DEFAULT_FIGURE_PATH),
        help="Figura Matplotlib com o resultado.",
    )
    _add_model_adapter_args(identify_parser)

    prepare_parser = subparsers.add_parser(
        "prepare-lfw",
        help="Baixa LFW, gera splits train/val/test e layout gallery/query.",
    )
    prepare_parser.add_argument(
        "--root",
        default=str(LFW_ROOT),
        help="Pasta raiz do LFW (gitignored).",
    )
    prepare_parser.add_argument(
        "--min-images",
        type=int,
        default=5,
        help="Mínimo de fotos por identidade no treino do adaptador.",
    )
    prepare_parser.add_argument("--seed", type=int, default=42)
    prepare_parser.add_argument(
        "--no-download",
        action="store_true",
        help="Não baixa; espera data/lfw/raw já populado.",
    )
    prepare_parser.add_argument(
        "--force-download",
        action="store_true",
        help="Baixa de novo mesmo se raw/ já existir.",
    )
    prepare_parser.add_argument(
        "--archive",
        default=None,
        help="Caminho local para lfw.tgz (pula o download pela rede).",
    )

    train_parser = subparsers.add_parser(
        "train-adapter",
        help="Treina LinearAdapter com embeddings congelados buffalo_l (CE).",
    )
    train_parser.add_argument(
        "--splits",
        default=str(DEFAULT_SPLITS_PATH),
        help="JSON de splits gerado por prepare-lfw.",
    )
    train_parser.add_argument(
        "--output",
        default=str(DEFAULT_ADAPTER_PATH),
        help="Checkpoint do adaptador.",
    )
    train_parser.add_argument("--epochs", type=int, default=20)
    train_parser.add_argument("--batch-size", type=int, default=64)
    train_parser.add_argument("--lr", type=float, default=1e-3)
    train_parser.add_argument(
        "--device",
        default="auto",
        help="cuda, cpu ou auto (usa GPU se disponível).",
    )
    train_parser.add_argument("--seed", type=int, default=42)
    _add_model_adapter_args(train_parser)

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Avalia baseline ou adaptador no split de teste (threshold na val).",
    )
    evaluate_parser.add_argument(
        "--splits",
        default=str(DEFAULT_SPLITS_PATH),
        help="JSON de splits gerado por prepare-lfw.",
    )
    evaluate_parser.add_argument(
        "--metrics-csv",
        default=str(DEFAULT_METRICS_CSV),
        help="CSV resumo acumulado (sem embeddings).",
    )
    evaluate_parser.add_argument(
        "--output-json",
        default=str(RESULTS_DIR / "evaluation_summary.json"),
        help="JSON resumo desta execução.",
    )
    evaluate_parser.add_argument(
        "--output-detail-csv",
        default=str(RESULTS_DIR / "evaluation_detail.csv"),
        help="CSV por query.",
    )
    evaluate_parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Se informado, fixa o threshold (pula calibração na val).",
    )
    _add_model_adapter_args(evaluate_parser)

    return parser.parse_args(argv)


def build_extractor(args: argparse.Namespace) -> FaceEmbedder:
    adapter = getattr(args, "adapter", "none")
    model = getattr(args, "model", "arcface")
    ctx_id = getattr(args, "ctx_id", -1)
    return get_embedder(model_name=model, adapter=adapter, ctx_id=ctx_id)


def select_face(
    detections: list[FaceDetection],
    face_index: int | None,
    label: str = "consulta",
) -> tuple[FaceDetection, int]:
    if not detections:
        raise ValueError(f"Nenhum rosto detectado na imagem de {label}.")

    if len(detections) == 1:
        index = 0 if face_index is None else face_index
    elif face_index is None:
        raise ValueError(
            f"A imagem de {label} tem {len(detections)} rostos. "
            "Informe --face-index explicitamente."
        )
    else:
        index = face_index

    if index < 0 or index >= len(detections):
        raise ValueError(
            f"Índice {index} inválido para a imagem de {label} "
            f"({len(detections)} rosto(s) detectado(s))."
        )
    return detections[index], index


def run_enroll(
    argv: list[str] | None = None,
    extractor: FaceEmbedder | None = None,
) -> dict:
    args = parse_args(argv)
    if args.command != "enroll":
        raise ValueError("Comando esperado: enroll")

    extractor = extractor or build_extractor(args)
    gallery, report = FaceGallery.enroll(args.gallery, extractor)
    if not gallery.records:
        raise ValueError(
            "Nenhum rosto cadastrado. Verifique as pastas em "
            f"{args.gallery} (uma subpasta por identidade, uma face por foto)."
        )

    output_path = gallery.save(args.output)
    print(f"Modelo: {getattr(extractor, 'name', args.model)}")
    print(f"Identidades cadastradas: {len({record.identity for record in gallery.records})}")
    print(f"Registros (fotos) cadastrados: {report.enrolled}")
    if report.skipped:
        print(f"Arquivos ignorados: {len(report.skipped)}")
        for item in report.skipped:
            print(f"  - {item}")
    print(f"Índice salvo em: {output_path}")
    return {
        "gallery": str(Path(args.gallery)),
        "output": str(output_path),
        "enrolled": report.enrolled,
        "skipped": report.skipped,
        "identities": sorted({record.identity for record in gallery.records}),
        "model": getattr(extractor, "name", args.model),
        "adapter": args.adapter,
    }


def _load_gallery_for_identify(
    args: argparse.Namespace,
    extractor: FaceEmbedder,
) -> FaceGallery:
    if args.index:
        return FaceGallery.load(args.index)
    if args.gallery:
        gallery, report = FaceGallery.enroll(args.gallery, extractor)
        if report.skipped:
            print(f"Arquivos ignorados no cadastro sob demanda: {len(report.skipped)}")
            for item in report.skipped:
                print(f"  - {item}")
        return gallery
    raise ValueError("Informe --index ou --gallery para a busca.")


def run_identify(
    argv: list[str] | None = None,
    extractor: FaceEmbedder | None = None,
) -> dict:
    args = parse_args(argv if argv is not None else None)
    if args.command != "identify":
        raise ValueError("Comando esperado: identify")
    if args.index is None and args.gallery is None:
        args.gallery = str(DEFAULT_GALLERY_DIR)

    extractor = extractor or build_extractor(args)
    query_image = load_image(args.image)
    query_detections = extractor.extract(query_image)
    print(f"Consulta: {len(query_detections)} rosto(s) detectado(s).")
    query_face, query_index = select_face(query_detections, args.face_index)

    gallery = _load_gallery_for_identify(args, extractor)
    if not gallery.records:
        raise ValueError("A galeria está vazia. Cadastre identidades antes de buscar.")

    comparator = FaceComparator(threshold=args.threshold)
    identification = comparator.identify(
        query_face.embedding,
        gallery.records,
        top_k=args.top_k,
    )

    print(f"Modelo: {getattr(extractor, 'name', args.model)} | adapter: {args.adapter}")
    print(f"Índice selecionado na consulta: {query_index}")
    print(f"Registros na galeria: {len(gallery.records)}")
    print(f"Melhor similaridade: {identification.similarity:.4f}")
    print(f"Threshold: {identification.threshold:.4f}")
    if identification.matched:
        print(f"Resultado: match com '{identification.identity}'.")
    else:
        print("Resultado: não encontrado na galeria.")

    payload = {
        "query_image": str(Path(args.image)),
        "n_faces_query": len(query_detections),
        "face_index": query_index,
        "matched": identification.matched,
        "identity": identification.identity,
        "similarity": identification.similarity,
        "threshold": identification.threshold,
        "best_gallery_image": identification.best_gallery_path,
        "model": getattr(extractor, "name", args.model),
        "adapter": args.adapter,
        "candidates": [
            {
                "identity": candidate.identity,
                "similarity": candidate.similarity,
                "image_path": candidate.image_path,
            }
            for candidate in identification.candidates
        ],
    }

    json_path = Path(args.output_json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"JSON salvo em: {json_path}")

    gallery_image = None
    gallery_detections = None
    if identification.matched and identification.best_gallery_path:
        gallery_image = load_image(identification.best_gallery_path)
        matched_record = next(
            record
            for record in gallery.records
            if record.image_path == identification.best_gallery_path
            and record.identity == identification.identity
        )
        gallery_detections = [gallery.as_detection(matched_record)]

    figure_path = save_identification_figure(
        query_image,
        query_detections,
        query_index,
        identification,
        args.output_figure,
        gallery_image=gallery_image,
        gallery_detections=gallery_detections,
    )
    print(f"Figura salva em: {figure_path}")
    return payload


def run_prepare_lfw(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    if args.command != "prepare-lfw":
        raise ValueError("Comando esperado: prepare-lfw")

    report = prepare_lfw(
        root=args.root,
        download=not args.no_download,
        force_download=args.force_download,
        min_images=args.min_images,
        seed=args.seed,
        materialize=True,
        archive=args.archive,
    )
    print(f"LFW em: {report.root}")
    print(f"Identidades raw: {report.n_identities_raw}")
    print(f"Train adapter: {report.n_train_identities} identidades")
    print(f"Val: {report.n_val_identities} | Test gallery: {report.n_test_gallery_identities}")
    print(f"Open-set: {report.n_test_open_set_identities}")
    print(f"Galeria teste: {report.n_gallery_images} | Queries: {report.n_query_images}")
    print(f"Splits: {report.splits_path}")
    return {
        "root": report.root,
        "splits_path": report.splits_path,
        "n_identities_raw": report.n_identities_raw,
        "n_train_identities": report.n_train_identities,
        "n_val_identities": report.n_val_identities,
        "n_test_gallery_identities": report.n_test_gallery_identities,
        "n_test_open_set_identities": report.n_test_open_set_identities,
        "n_gallery_images": report.n_gallery_images,
        "n_query_images": report.n_query_images,
    }


def run_train_adapter(
    argv: list[str] | None = None,
    extractor: FaceEmbedder | None = None,
) -> dict:
    args = parse_args(argv)
    if args.command != "train-adapter":
        raise ValueError("Comando esperado: train-adapter")

    # Treino sempre sobre embeddings congelados do buffalo_l (sem adaptador).
    if extractor is None:
        extractor = get_embedder(
            model_name=args.model,
            adapter="none",
            ctx_id=args.ctx_id,
        )

    splits = load_splits(args.splits)
    train_images = splits["train_adapter"]["images"]
    if not train_images:
        raise ValueError("Split train_adapter vazio. Rode prepare-lfw antes.")

    paths = [item.get("absolute_path") or item["path"] for item in train_images]
    identities = [str(item["identity"]) for item in train_images]
    print(f"Extraindo embeddings congelados ({len(paths)} imagens)...")
    embeddings, labels, kept, skipped = extract_frozen_embeddings(
        paths, identities, extractor
    )
    print(f"Embeddings válidos: {len(kept)} | ignorados: {len(skipped)}")

    report = train_linear_adapter(
        embeddings,
        labels,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device,
        output_path=args.output,
        seed=args.seed,
    )
    print(f"Adaptador salvo em: {report.checkpoint}")
    print(
        f"Identidades: {report.n_identities} | samples: {report.n_samples} | "
        f"loss final: {report.final_loss:.4f} | device: {report.device}"
    )
    return {
        "checkpoint": report.checkpoint,
        "epochs": report.epochs,
        "final_loss": report.final_loss,
        "n_identities": report.n_identities,
        "n_samples": report.n_samples,
        "device": report.device,
        "skipped": skipped,
    }


def run_evaluate(
    argv: list[str] | None = None,
    extractor: FaceEmbedder | None = None,
) -> dict:
    args = parse_args(argv)
    if args.command != "evaluate":
        raise ValueError("Comando esperado: evaluate")

    extractor = extractor or build_extractor(args)
    splits = load_splits(args.splits)
    mode = (
        "adapter"
        if args.adapter not in (None, "", "none")
        else "baseline"
    )

    print(f"Enroll da galeria de validação ({mode})...")
    val_gallery = enroll_from_split_items(splits["val"]["gallery"], extractor)
    if not val_gallery.records:
        raise ValueError("Galeria de validação vazia após extração.")

    if args.threshold is None:
        print("Calibrando threshold na validação (maximiza F1)...")
        scores, genuine = collect_calibration_scores(
            val_gallery,
            splits["val"]["queries"],
            extractor,
        )
        if not scores:
            raise ValueError("Não foi possível calibrar: sem scores na validação.")
        threshold, val_counts = calibrate_threshold(scores, genuine, metric="f1")
        threshold_source = "val_f1"
        print(
            f"Threshold calibrado: {threshold:.4f} "
            f"(val F1={val_counts.f1():.4f}, TP={val_counts.tp} FP={val_counts.fp} "
            f"TN={val_counts.tn} FN={val_counts.fn})"
        )
    else:
        threshold = float(args.threshold)
        threshold_source = "cli"
        print(f"Threshold fixo via CLI: {threshold:.4f}")

    print("Enroll da galeria de teste...")
    test_gallery = enroll_from_split_items(splits["test"]["gallery"], extractor)
    if not test_gallery.records:
        raise ValueError("Galeria de teste vazia após extração.")

    report = evaluate_identification(
        test_gallery,
        splits["test"]["queries"],
        extractor,
        threshold=threshold,
        mode=mode,
        threshold_source=threshold_source,
    )

    detail_csv = save_metrics_csv(report, args.output_detail_csv)
    summary_json = save_metrics_summary(report, args.output_json)
    table_csv = append_metrics_table_row(report, args.metrics_csv)

    print(f"Modo: {report.mode} | queries: {report.n_queries}")
    print(
        f"Identification rate: {report.identification_rate:.4f} | "
        f"closed-set: {report.closed_set_accuracy:.4f} | "
        f"open-set reject: {report.open_set_rejection_rate:.4f}"
    )
    print(
        f"Rank-1: {report.rank1:.4f} | P/R/F1: "
        f"{report.precision:.4f}/{report.recall:.4f}/{report.f1:.4f}"
    )
    print(
        f"FP/FN matrix: TP={report.confusion.tp} FP={report.confusion.fp} "
        f"TN={report.confusion.tn} FN={report.confusion.fn}"
    )
    print(
        f"Latência média: extract={report.mean_extract_ms:.1f} ms | "
        f"identify={report.mean_identify_ms:.1f} ms"
    )
    print(f"Detalhe: {detail_csv}")
    print(f"Resumo: {summary_json}")
    print(f"Tabela: {table_csv}")

    payload = report.to_summary_dict()
    payload["detail_csv"] = str(detail_csv)
    payload["summary_json"] = str(summary_json)
    payload["metrics_csv"] = str(table_csv)
    payload["model"] = getattr(extractor, "name", args.model)
    payload["adapter"] = args.adapter
    return payload


def run(
    argv: list[str] | None = None,
    extractor: FaceEmbedder | None = None,
) -> dict:
    args = parse_args(argv)
    if args.command == "enroll":
        return run_enroll(argv, extractor=extractor)
    if args.command == "identify":
        return run_identify(argv, extractor=extractor)
    if args.command == "prepare-lfw":
        return run_prepare_lfw(argv)
    if args.command == "train-adapter":
        return run_train_adapter(argv, extractor=extractor)
    if args.command == "evaluate":
        return run_evaluate(argv, extractor=extractor)
    raise ValueError(f"Comando desconhecido: {args.command}")


def main() -> None:
    try:
        run()
    except (ImageLoadError, ValueError, NotImplementedError) as error:
        print(f"Erro: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
