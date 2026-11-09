import numpy as np

from trace.belief.appearance import AppearanceSource, sigmoid, softmax
from trace.relation import EpisodeMemory, OPEN, infer_relation
from trace.types import Candidate, Observation


def test_softmax_and_none_mass():
    body = softmax(np.array([10.0, 0.0]), temperature=0.05)
    assert body[0] > 0.99
    assert abs(body.sum() - 1.0) < 1e-12
    # cosine well below the 0.6 match threshold -> high p_none
    assert sigmoid((0.6 - 0.1) / 0.1) > 0.9
    # cosine well above it -> low p_none
    assert sigmoid((0.6 - 0.95) / 0.1) < 0.1


def test_dinov2_source_uses_injected_embedder():
    reference = np.zeros((32, 32, 3), dtype=np.uint8)
    reference[:, :] = (200, 30, 30)
    match = reference.copy()
    other = np.zeros((32, 32, 3), dtype=np.uint8)
    other[:, :] = (30, 30, 200)

    def embedder(images):
        vectors = []
        for image in images:
            mean = image.reshape(-1, 3).mean(axis=0).astype(np.float64)
            norm = np.linalg.norm(mean) or 1.0
            vectors.append(mean / norm)
        out = np.stack(vectors)
        embedder.backend = "fake"
        return out

    embedder.backend = "fake"
    source = AppearanceSource(embedder=embedder)
    memory = EpisodeMemory()
    memory.record(OPEN, "Mug", True, 1)
    relation = infer_relation("CloseObject", "Mug", memory)
    candidates = [
        Candidate(0, "mug_a", "Mug", crop=match),
        Candidate(1, "mug_b", "Mug", crop=other),
    ]
    observation = Observation(reference_frame=reference)
    belief = source.identify(relation, candidates, observation)
    assert belief.source == "dinov2"
    assert belief.raw["backend"] == "fake"
    assert belief.p_identity.argmax() == 0
    assert belief.p_identity[0] > belief.p_identity[1]
    assert abs(belief.p_identity.sum() - 1.0) < 1e-12
