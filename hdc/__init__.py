"""hdc-watch 공용 HDC 부품. 라이브러리 없이 직접 구현한다."""

from .vectors import D, make_vector, flip_order, level_boundaries, level_vectors
from .ops import bind, unbind, hamming, similarity, noise_level, Memory, nearest
