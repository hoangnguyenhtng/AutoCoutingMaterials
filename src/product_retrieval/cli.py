"""Command-line interface for product retrieval experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from product_retrieval.config import load_config, make_encoder, make_retriever
from product_retrieval.data import load_manifest, validate_independent_splits
from product_retrieval.retrieval import GalleryBuilder
from product_retrieval.verification import ProductVerifier, VerificationThresholds


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="product-retrieval", description=__doc__)
    parser.add_argument("--config", default="configs/experiment.yaml", help="YAML experiment config")
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate-manifest", help="Validate paths, schema and split leakage")
    validate.add_argument("--manifest")
    validate.add_argument("--allow-missing", action="store_true", help="Do not require image files")

    build = commands.add_parser("build-gallery", help="Encode gallery rows and save an NPZ gallery")
    build.add_argument("--manifest")
    build.add_argument("--output")

    search = commands.add_parser("search", help="Return top product-code candidates for one crop")
    search.add_argument("image")
    search.add_argument("--gallery")
    search.add_argument("--top-k", type=int)

    verify = commands.add_parser("verify", help="Verify one crop against an operator-selected code")
    verify.add_argument("image")
    verify.add_argument("--selected-code", required=True)
    verify.add_argument("--gallery")

    evaluate = commands.add_parser("evaluate", help="Evaluate query and unknown manifest rows")
    evaluate.add_argument("--manifest")
    evaluate.add_argument("--gallery")
    return parser


def _gallery_path(args, config: dict) -> str:
    return args.gallery or config["paths"]["gallery"]


def _ensure_compatible(encoder, gallery) -> None:
    expected = encoder.get_model_info()
    actual = gallery.model_info
    keys = ("model_name", "feature_type", "embedding_dimension", "preprocessing")
    differences = [key for key in keys if actual.get(key) != expected.get(key)]
    if differences:
        raise ValueError(
            "Gallery and current encoder config are incompatible in: " + ", ".join(differences)
        )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config = load_config(args.config)
        paths = config["paths"]

        if args.command == "validate-manifest":
            manifest_path = args.manifest or paths["manifest"]
            entries = load_manifest(manifest_path, check_files=not args.allow_missing)
            warnings = validate_independent_splits(entries)
            print(json.dumps({"rows": len(entries), "warnings": warnings}, ensure_ascii=False, indent=2))
            return 2 if warnings else 0

        encoder = make_encoder(config)
        batch_size = int(config["model"].get("batch_size", 16))
        if args.command == "build-gallery":
            builder = GalleryBuilder(
                encoder,
                gallery_path=args.output or paths["gallery"],
                batch_size=batch_size,
            )
            gallery = builder.build(args.manifest or paths["manifest"])
            print(json.dumps({"vectors": len(gallery.embeddings), "codes": len(gallery.codes)}, indent=2))
            return 0

        gallery = GalleryBuilder.load(_gallery_path(args, config))
        _ensure_compatible(encoder, gallery)
        retriever = make_retriever(config, gallery)

        if args.command == "search":
            embedding = encoder.encode_image(args.image)
            top_k = args.top_k or int(config["retrieval"].get("top_k", 5))
            print(json.dumps(retriever.rank_codes(embedding, top_k=top_k), ensure_ascii=False, indent=2))
            return 0

        if args.command == "verify":
            embedding = encoder.encode_image(args.image)
            verifier = ProductVerifier(retriever)
            result = verifier.verify(
                embedding,
                args.selected_code,
                VerificationThresholds.from_mapping(config["verification"]),
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0

        if args.command == "evaluate":
            from product_retrieval.evaluation import Evaluator

            manifest_path = args.manifest or paths["manifest"]
            all_entries = load_manifest(manifest_path)
            warnings = validate_independent_splits(all_entries)
            if warnings:
                print("Manifest leakage warnings:", file=sys.stderr)
                for warning in warnings:
                    print(f"- {warning}", file=sys.stderr)
            entries = [entry for entry in all_entries if entry.split in {"query", "unknown"}]
            evaluator = Evaluator(encoder, retriever, config["verification"])
            metrics, predictions, difficult_pairs = evaluator.evaluate(entries, batch_size=batch_size)
            evaluator.write_reports(
                metrics,
                predictions,
                difficult_pairs,
                metrics_path=paths["metrics"],
                predictions_path=paths["predictions"],
                difficult_pairs_path=paths["difficult_pairs"],
                experiment={
                    **config.get("experiment", {}),
                    "model": encoder.get_model_info(),
                    "retrieval": config["retrieval"],
                    "thresholds": config["verification"],
                },
            )
            print(json.dumps(metrics, ensure_ascii=False, indent=2))
            return 0
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

