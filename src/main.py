"""Cadastro da galeria e identificação 1:N com ArcFace."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.evaluation import DEFAULT_THRESHOLD, FaceComparator
from src.gallery import FaceGallery
from src.models import ArcFaceExtractor, FaceDetection
from src.preprocessing import ImageLoadError, load_image
from src.visualization import save_identification_figure

RESULTS_DIR = Path("results")
DEFAULT_GALLERY_DIR = Path("data/gallery")
DEFAULT_INDEX_PATH = RESULTS_DIR / "gallery.npz"
DEFAULT_JSON_PATH = RESULTS_DIR / "identification.json"
DEFAULT_FIGURE_PATH = RESULTS_DIR / "identification.png"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Identificação facial 1:N com ArcFace (InsightFace buffalo_l). "
            "Cadastre uma galeria e consulte com uma imagem."
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

    identify_parser = subparsers.add_parser(
        "identify",
        help="Busca a identidade de uma imagem na galeria.",
    )
    identify_parser.add_argument(
        "--image",
        required=True,
        help="Imagem de consulta.",
    )
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
        help=(
            "Limiar provisório de similaridade de cosseno. "
            "Deve ser calibrado com uma base de validação."
        ),
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
    return parser.parse_args(argv)


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
    extractor: ArcFaceExtractor | None = None,
) -> dict:
    args = parse_args(argv if argv is not None else ["enroll"])
    if args.command != "enroll":
        raise ValueError("Comando esperado: enroll")

    extractor = extractor or ArcFaceExtractor()
    gallery, report = FaceGallery.enroll(args.gallery, extractor)
    if not gallery.records:
        raise ValueError(
            "Nenhum rosto cadastrado. Verifique as pastas em "
            f"{args.gallery} (uma subpasta por identidade, uma face por foto)."
        )

    output_path = gallery.save(args.output)
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
    }


def _load_gallery_for_identify(
    args: argparse.Namespace,
    extractor: ArcFaceExtractor,
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
    extractor: ArcFaceExtractor | None = None,
) -> dict:
    args = parse_args(argv if argv is not None else None)
    if args.command != "identify":
        raise ValueError("Comando esperado: identify")
    if args.index is None and args.gallery is None:
        args.gallery = str(DEFAULT_GALLERY_DIR)

    extractor = extractor or ArcFaceExtractor()
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

    print(f"Índice selecionado na consulta: {query_index}")
    print(f"Registros na galeria: {len(gallery.records)}")
    print(f"Melhor similaridade: {identification.similarity:.4f}")
    print(f"Threshold: {identification.threshold:.2f} (provisório, não calibrado)")
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


def run(
    argv: list[str] | None = None,
    extractor: ArcFaceExtractor | None = None,
) -> dict:
    args = parse_args(argv)
    if args.command == "enroll":
        return run_enroll(argv, extractor=extractor)
    if args.command == "identify":
        return run_identify(argv, extractor=extractor)
    raise ValueError(f"Comando desconhecido: {args.command}")


def main() -> None:
    try:
        run()
    except (ImageLoadError, ValueError) as error:
        print(f"Erro: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
