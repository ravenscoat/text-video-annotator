import numpy as np

from video_annotator.semantic_encoder import SemanticEncoder


class MockEncoder:
    def encode_images(self, images):
        return np.asarray([[float(image.mean()), 1.0] for image in images], dtype=np.float32)

    def encode_text(self, texts):
        return np.asarray([[float(len(text)), 1.0] for text in texts], dtype=np.float32)


def test_mock_encoder_contract_is_batchable():
    encoder: SemanticEncoder = MockEncoder()
    assert encoder.encode_images([np.zeros((2, 2, 3)), np.ones((2, 2, 3))]).shape == (2, 2)
    assert encoder.encode_text(["dog", "person running"]).shape == (2, 2)
