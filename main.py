        df = compute_semantic_features(df, input_root)
        features = df[["similarity_a", "similarity_b"]].to_numpy(dtype=float)
        if not np.isfinite(features).all():
            raise ValueError("Semantic features contain non-finite values.")
    clear_gpu_memory()

    with track_stage("custom_inference", tracker):
        predict_custom_model(df, input_root, output_dir,
                             batch_size=args.custom_batch_size,
                             max_length=args.custom_max_length)
    clear_gpu_memory()

    with track_stage("ensemble", tracker):
        submission = make_submission(df, output_dir, args.ensemble_weights)
        arrays = {
            "gemma": np.load(output_dir / "prob_m0.npy", allow_pickle=False),
            # Compare all models using the original A/B class order.
            "llama": np.load(output_dir / "prob_m3.npy", allow_pickle=False)[:, [1, 0, 2]],
            "custom": np.load(output_dir / "prob_faiss.npy", allow_pickle=False),
            "ensemble": submission[CLASS_COLUMNS].to_numpy(),
        }
        for name, probabilities in arrays.items():
            tracker.log_metrics(prediction_metrics(name, probabilities, labels))

    output_manifest = {
        "class_order": CLASS_COLUMNS,
        "rows": len(df),
        "evaluation": "real_labels" if labels is not None else "skipped_no_labels",
        "files": [{"name": name, "sha256": sha256_file(output_dir / name)}
                  for name in ["submission.csv", "prob_m0.npy", "prob_m3.npy", "prob_faiss.npy"]],
    }
    write_json(output_dir / "output_manifest.json", output_manifest)
    tracker.log_artifact(output_dir / "output_manifest.json")
    if args.log_predictions:
        for item in output_manifest["files"]:
            tracker.log_artifact(output_dir / item["name"], "predictions")


def main(argv=None):
    args = parse_args(argv)
    execution_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    output_dir = args.output_dir / execution_id
    output_dir.mkdir(parents=True, exist_ok=False)

    for handler in list(LOGGER.handlers):
        handler.close()
        LOGGER.removeHandler(handler)
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False
    for handler in [logging.StreamHandler(), logging.FileHandler(output_dir / "pipeline.log", encoding="utf-8")]:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        LOGGER.addHandler(handler)

    LOGGER.info("Run directory: %s", output_dir)
    config = vars(args).copy()
    # A server URI may contain credentials. It is used, but never saved as an artifact.
    config["tracking_uri"] = "configured" if args.tracking_uri or os.environ.get("MLFLOW_TRACKING_URI") else "local_sqlite"
    write_json(output_dir / "run_config.json", config)
    started = time.perf_counter()
    tracker = RunTracker()
    try:
        with tracking_session(args, execution_id) as tracker:
            try:
                run_pipeline(args, output_dir, tracker)
                tracker.log_metrics({"total_seconds": time.perf_counter() - started})
                write_json(output_dir / "metrics.json", tracker.metrics)
                write_json(output_dir / "run_summary.json", {
                    "status": "FINISHED", "execution_id": execution_id,
                    "mlflow_run_id": tracker.run_id,
                    "submission": str(output_dir / "submission.csv"),
                })
                LOGGER.info("Pipeline completed. Submission: %s", output_dir / "submission.csv")
                for name in ["metrics.json", "run_summary.json", "pipeline.log"]:
                    tracker.log_artifact(output_dir / name)
            except BaseException as error:
                LOGGER.exception("Pipeline failed")
                failure = {
                    "status": "FAILED", "exception_type": type(error).__name__,
                    "message": str(error), "mlflow_run_id": tracker.run_id,
                }
                write_json(output_dir / "failure.json", failure)
                write_json(output_dir / "run_summary.json", failure)
                write_json(output_dir / "metrics.json", tracker.metrics)
                # Preserve the original exception even if the server becomes unavailable.
                with suppress(Exception):
                    for name in ["run_config.json", "failure.json", "run_summary.json", "metrics.json", "pipeline.log"]:
                        tracker.log_artifact(output_dir / name)
                raise
    except BaseException as error:
        # Startup failures may happen before an MLflow run can be created.
        if not (output_dir / "failure.json").exists():
            LOGGER.exception("Tracking setup failed")
            failure = {"status": "FAILED", "exception_type": type(error).__name__,
                       "message": str(error), "mlflow_run_id": tracker.run_id}
            write_json(output_dir / "failure.json", failure)
            write_json(output_dir / "run_summary.json", failure)
        LOGGER.error("Execution failed; diagnostic files are in %s", output_dir)
        raise
    return output_dir


if __name__ == "__main__":
    main()
