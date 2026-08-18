import pytest

from rag_workbench.experiments.configs import (
    DatasetIdentity,
    ExperimentConfig,
    GenerationConfig,
    IngestionConfig,
    RetrievalConfig,
)


def config(*, chunk_size: int = 180, top_k: int = 5) -> ExperimentConfig:
    return ExperimentConfig(
        name="test",
        ingestion=IngestionConfig(chunk_size=chunk_size),
        retrieval=RetrievalConfig(top_k=top_k),
        generation=GenerationConfig(),
        identity=DatasetIdentity(corpus_version="v1", evaluation_dataset_version="eval-v1"),
    )


def test_retrieval_change_does_not_require_reindex() -> None:
    assert not config(top_k=10).requires_reindex(config(top_k=5))


def test_threshold_change_does_not_require_reindex() -> None:
    original = config()
    changed = original.model_copy(
        update={"retrieval": original.retrieval.model_copy(update={"score_threshold": 0.4})}
    )
    assert not changed.requires_reindex(original)
    assert changed.experiment_config_hash != original.experiment_config_hash
    assert changed.index_identity == original.index_identity


def test_chunk_change_requires_reindex() -> None:
    assert config(chunk_size=200).requires_reindex(config(chunk_size=180))


def test_same_config_has_same_stable_identity() -> None:
    assert config().experiment_config_hash == config().experiment_config_hash
    assert config().index_identity == config().index_identity


def test_embedding_change_requires_reindex() -> None:
    original = config()
    changed = original.model_copy(
        update={
            "ingestion": original.ingestion.model_copy(
                update={"embedding_provider": "openai-compatible", "embedding_model": "model-x"}
            )
        }
    )
    assert changed.requires_reindex(original)


def test_embedding_dimension_and_version_are_part_of_index_identity() -> None:
    original = config()
    changed_dimension = original.model_copy(
        update={
            "ingestion": original.ingestion.model_copy(update={"embedding_dimension": 128})
        }
    )
    changed_version = original.model_copy(
        update={"ingestion": original.ingestion.model_copy(update={"embedding_version": "2"})}
    )
    assert changed_dimension.index_identity != original.index_identity
    assert changed_version.index_identity != original.index_identity


def test_overlap_must_be_strictly_smaller_than_chunk_size() -> None:
    with pytest.raises(ValueError, match="chunk_overlap"):
        IngestionConfig(chunk_size=300, chunk_overlap=300)
