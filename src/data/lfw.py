"""Download do LFW, splits train/val/test e layout galeria/query."""

from __future__ import annotations

import json
import random
import shutil
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LFW_ROOT = Path("data/lfw")
# O host oficial da UMass costuma falhar em DNS; Figshare é o mirror do scikit-learn.
LFW_URLS = (
    "https://ndownloader.figshare.com/files/5976018",  # lfw.tgz
    "https://ndownloader.figshare.com/files/5976015",  # lfw-funneled.tgz (fallback)
    "http://vis-www.cs.umass.edu/lfw/lfw.tgz",
)
LFW_URL = LFW_URLS[0]
DEFAULT_MIN_IMAGES = 5
DEFAULT_SEED = 42
DOWNLOAD_TIMEOUT_SECONDS = 120


@dataclass
class LfwPrepareReport:
    root: str
    n_identities_raw: int
    n_train_identities: int
    n_val_identities: int
    n_test_gallery_identities: int
    n_test_open_set_identities: int
    n_gallery_images: int
    n_query_images: int
    splits_path: str


def _list_identity_images(identity_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in identity_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def scan_lfw_identities(raw_dir: Path) -> dict[str, list[Path]]:
    if not raw_dir.exists():
        raise ValueError(f"Pasta LFW raw inexistente: {raw_dir}")

    identities: dict[str, list[Path]] = {}
    for identity_dir in sorted(path for path in raw_dir.iterdir() if path.is_dir()):
        images = _list_identity_images(identity_dir)
        if images:
            identities[identity_dir.name] = images
    if not identities:
        raise ValueError(f"Nenhuma identidade com imagens em {raw_dir}")
    return identities


def _download_url_to_file(url: str, destination: Path) -> None:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "tcc-reconhecimento-facial/1.0"},
    )
    with urllib.request.urlopen(request, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
        with destination.open("wb") as handle:
            shutil.copyfileobj(response, handle)


def _find_extracted_people_root(extract_root: Path) -> Path:
    for candidate_name in ("lfw", "lfw_funneled"):
        candidate = extract_root / candidate_name
        if candidate.is_dir():
            return candidate

    candidates = [path for path in extract_root.iterdir() if path.is_dir()]
    if len(candidates) == 1:
        nested = candidates[0]
        for candidate_name in ("lfw", "lfw_funneled"):
            candidate = nested / candidate_name
            if candidate.is_dir():
                return candidate
        return nested

    raise ValueError(
        "Não foi possível localizar a pasta lfw/lfw_funneled no arquivo baixado."
    )


def _extract_lfw_archive(archive_path: Path, raw_dir: Path) -> None:
    extract_root = raw_dir.parent / "_lfw_extract"
    if extract_root.exists():
        shutil.rmtree(extract_root)
    extract_root.mkdir(parents=True, exist_ok=True)

    with tarfile.open(archive_path, "r:gz") as archive:
        try:
            archive.extractall(extract_root, filter="data")
        except TypeError:
            archive.extractall(extract_root)

    extracted_lfw = _find_extracted_people_root(extract_root)

    if raw_dir.exists():
        for child in list(raw_dir.iterdir()):
            if child.is_dir():
                shutil.rmtree(child)
            elif child.name != ".downloaded":
                child.unlink()
    else:
        raw_dir.mkdir(parents=True, exist_ok=True)

    for identity_dir in sorted(path for path in extracted_lfw.iterdir() if path.is_dir()):
        target = raw_dir / identity_dir.name
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(identity_dir, target)

    marker = raw_dir / ".downloaded"
    marker.write_text("ok\n", encoding="utf-8")
    shutil.rmtree(extract_root, ignore_errors=True)


def download_lfw(
    raw_dir: Path,
    url: str | None = None,
    force: bool = False,
    archive: str | Path | None = None,
) -> Path:
    """Baixa (ou usa arquivo local) e extrai o LFW para raw_dir (identidade/*.jpg)."""
    raw_dir = Path(raw_dir)
    marker = raw_dir / ".downloaded"

    if raw_dir.exists() and any(raw_dir.iterdir()) and not force:
        if marker.exists() or any(path.is_dir() for path in raw_dir.iterdir()):
            return raw_dir

    raw_dir.mkdir(parents=True, exist_ok=True)
    archive_path = Path(archive) if archive else raw_dir.parent / "lfw.tgz"
    remove_archive_after = archive is None

    if archive is not None:
        if not archive_path.exists():
            raise ValueError(f"Arquivo LFW inexistente: {archive_path}")
        print(f"Usando arquivo local: {archive_path}")
        _extract_lfw_archive(archive_path, raw_dir)
        return raw_dir

    if archive_path.exists() and not force:
        print(f"Usando arquivo já presente: {archive_path}")
        _extract_lfw_archive(archive_path, raw_dir)
        return raw_dir

    urls = (url,) if url else LFW_URLS
    errors: list[str] = []
    for candidate in urls:
        try:
            print(f"Baixando LFW de {candidate} ...")
            _download_url_to_file(candidate, archive_path)
            _extract_lfw_archive(archive_path, raw_dir)
            if remove_archive_after:
                archive_path.unlink(missing_ok=True)
            return raw_dir
        except Exception as error:  # noqa: BLE001 - queremos tentar o próximo mirror
            errors.append(f"{candidate}: {error}")
            archive_path.unlink(missing_ok=True)
            print(f"Falha neste mirror: {error}")

    manual_hint = (
        "Não foi possível baixar o LFW automaticamente.\n"
        "Opções:\n"
        "1) Baixe manualmente https://ndownloader.figshare.com/files/5976018 "
        "e salve como data/lfw/lfw.tgz, depois rode de novo prepare-lfw.\n"
        "2) Ou: python -m src.main prepare-lfw --archive caminho/lfw.tgz\n"
        "3) Ou coloque pastas de identidade em data/lfw/raw/ e use --no-download.\n"
        "Erros:\n- " + "\n- ".join(errors)
    )
    raise ValueError(manual_hint)


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def build_splits(
    identities: dict[str, list[Path]],
    *,
    root: Path,
    min_images: int = DEFAULT_MIN_IMAGES,
    seed: int = DEFAULT_SEED,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    gallery_images_per_id: int = 1,
) -> dict:
    """Split por identidade (sem leak) + protocolo 1:N de teste.

    - train_adapter: identidades com >= min_images (maioria das fotos)
    - val: identidades distintas para calibrar threshold
    - test: galeria 1+ foto/id + queries; parte open-set (só query)
    """
    rng = random.Random(seed)
    eligible = sorted(
        name for name, images in identities.items() if len(images) >= min_images
    )
    scarce = sorted(
        name for name, images in identities.items() if len(images) < min_images
    )
    if len(eligible) < 3:
        raise ValueError(
            f"Poucas identidades com >= {min_images} imagens ({len(eligible)}). "
            "Reduza --min-images ou use o LFW completo."
        )

    shuffled = eligible[:]
    rng.shuffle(shuffled)
    n_train = max(1, int(len(shuffled) * train_ratio))
    n_val = max(1, int(len(shuffled) * val_ratio))
    if n_train + n_val >= len(shuffled):
        n_train = max(1, len(shuffled) - 2)
        n_val = 1

    train_ids = shuffled[:n_train]
    val_ids = shuffled[n_train : n_train + n_val]
    test_ids = shuffled[n_train + n_val :]
    if not test_ids:
        # Garante pelo menos uma identidade de teste fechado
        test_ids = [val_ids.pop()] if len(val_ids) > 1 else [train_ids.pop()]

    # Open-set: identidades escassas (ou sobra) só em query
    open_set_ids = scarce[:]
    rng.shuffle(open_set_ids)
    # Limita open-set para não dominar o split em demos pequenas
    open_set_ids = open_set_ids[: max(1, len(test_ids))]

    def sample_paths(names: list[str], holdout_for_query: bool) -> tuple[list[dict], list[dict]]:
        gallery_items: list[dict] = []
        query_items: list[dict] = []
        for name in names:
            images = identities[name][:]
            rng.shuffle(images)
            if holdout_for_query and len(images) > gallery_images_per_id:
                gallery_imgs = images[:gallery_images_per_id]
                query_imgs = images[gallery_images_per_id:]
            elif holdout_for_query:
                gallery_imgs = images[:1]
                query_imgs = images[1:] if len(images) > 1 else []
            else:
                # Treino do adaptador: quase todas as fotos; deixa 1 de fora se houver
                if len(images) >= min_images:
                    query_imgs = images[-1:]
                    gallery_imgs = images[:-1]
                else:
                    gallery_imgs = images
                    query_imgs = []

            for path in gallery_imgs:
                gallery_items.append(
                    {
                        "identity": name,
                        "path": _relative(path, root),
                        "absolute_path": str(path),
                    }
                )
            for path in query_imgs:
                query_items.append(
                    {
                        "identity": name,
                        "path": _relative(path, root),
                        "absolute_path": str(path),
                        "in_gallery": True,
                    }
                )
        return gallery_items, query_items

    train_gallery, train_query = sample_paths(train_ids, holdout_for_query=False)
    val_gallery, val_query = sample_paths(val_ids, holdout_for_query=True)
    test_gallery, test_query = sample_paths(test_ids, holdout_for_query=True)

    for name in open_set_ids:
        images = identities[name][:]
        rng.shuffle(images)
        for path in images:
            test_query.append(
                {
                    "identity": name,
                    "path": _relative(path, root),
                    "absolute_path": str(path),
                    "in_gallery": False,
                }
            )

    return {
        "dataset": "lfw",
        "seed": seed,
        "min_images": min_images,
        "train_adapter": {
            "identities": train_ids,
            "images": [
                {
                    "identity": item["identity"],
                    "path": item["path"],
                    "absolute_path": item["absolute_path"],
                }
                for item in train_gallery + train_query
            ],
        },
        "val": {
            "identities": val_ids,
            "gallery": val_gallery,
            "queries": val_query,
        },
        "test": {
            "identities_in_gallery": test_ids,
            "identities_open_set": open_set_ids,
            "gallery": test_gallery,
            "queries": test_query,
        },
    }


def materialize_gallery_query(
    splits: dict,
    root: Path,
    *,
    split_name: str = "test",
) -> tuple[Path, Path]:
    """Copia (via hardlink/copy) imagens do split para gallery/ e query/."""
    gallery_dir = root / "gallery"
    query_dir = root / "query"
    if gallery_dir.exists():
        shutil.rmtree(gallery_dir)
    if query_dir.exists():
        shutil.rmtree(query_dir)
    gallery_dir.mkdir(parents=True, exist_ok=True)
    query_dir.mkdir(parents=True, exist_ok=True)

    split = splits[split_name]
    for item in split["gallery"]:
        src = Path(item["absolute_path"])
        dest_dir = gallery_dir / item["identity"]
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / src.name
        if not dest.exists():
            shutil.copy2(src, dest)

    for index, item in enumerate(split["queries"]):
        src = Path(item["absolute_path"])
        # Prefixo evita colisão de nomes entre identidades
        tag = "in" if item.get("in_gallery", True) else "out"
        dest_name = f"{tag}_{item['identity']}_{index:05d}{src.suffix.lower()}"
        dest = query_dir / dest_name
        shutil.copy2(src, dest)

    return gallery_dir, query_dir


def prepare_lfw(
    root: str | Path = LFW_ROOT,
    *,
    download: bool = True,
    force_download: bool = False,
    min_images: int = DEFAULT_MIN_IMAGES,
    seed: int = DEFAULT_SEED,
    materialize: bool = True,
    archive: str | Path | None = None,
) -> LfwPrepareReport:
    root_path = Path(root)
    raw_dir = root_path / "raw"
    root_path.mkdir(parents=True, exist_ok=True)

    if download or archive is not None:
        download_lfw(
            raw_dir,
            force=force_download,
            archive=archive,
        )
    elif not raw_dir.exists():
        raise ValueError(
            f"LFW raw não encontrado em {raw_dir}. "
            "Rode prepare-lfw (com download) ou coloque identidades em data/lfw/raw/."
        )

    identities = scan_lfw_identities(raw_dir)
    splits = build_splits(
        identities,
        root=root_path,
        min_images=min_images,
        seed=seed,
    )
    splits_path = root_path / "splits.json"
    # Remove absolute_path do JSON versionável em disco? Mantemos para uso local;
    # imagens em si não vão para o git.
    splits_path.write_text(json.dumps(splits, indent=2), encoding="utf-8")

    n_gallery = len(splits["test"]["gallery"])
    n_query = len(splits["test"]["queries"])
    if materialize:
        materialize_gallery_query(splits, root_path, split_name="test")

    return LfwPrepareReport(
        root=str(root_path),
        n_identities_raw=len(identities),
        n_train_identities=len(splits["train_adapter"]["identities"]),
        n_val_identities=len(splits["val"]["identities"]),
        n_test_gallery_identities=len(splits["test"]["identities_in_gallery"]),
        n_test_open_set_identities=len(splits["test"]["identities_open_set"]),
        n_gallery_images=n_gallery,
        n_query_images=n_query,
        splits_path=str(splits_path),
    )


def load_splits(splits_path: str | Path) -> dict:
    path = Path(splits_path)
    if not path.exists():
        raise ValueError(f"Arquivo de splits inexistente: {path}")
    return json.loads(path.read_text(encoding="utf-8"))
