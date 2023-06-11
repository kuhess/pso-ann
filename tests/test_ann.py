from numpy.testing import assert_array_equal

from psoann.ann import MultiLayerPerceptronWeights


def test_roundtrip():
    shape = [5, 4, 3]
    mlp_w = MultiLayerPerceptronWeights.create_random(shape)

    new_mlp_w = MultiLayerPerceptronWeights.from_particle_position(
        mlp_w.to_particle_position(), shape
    )

    assert mlp_w.shape == new_mlp_w.shape
    for w1, w2 in zip(mlp_w.weights, new_mlp_w.weights):
        assert_array_equal(w1, w2)
